# inference.jl - MCMC Sweep Engine & Inference Pipeline
#
# Generative Role:
# - Coordinates the complete joint MCMC transition kernel (Theta^(t), Z^(t)) -> (Theta^(t+1), Z^(t+1))
# - Executes the active m-mechanisms in mathematically sound conditional order:
#   m=2 (birth/prune occurrences) -> Sampler (spike assignment) ->
#   m=4 (coordinates u_e, loadings W, ARD) -> m=3 (gauge projection) ->
#   m=1 (packet latencies, variances, masses, splits) ->
#   m=5 (Continuous OU network arousal, single-cell drift, vigor, Student-t) ->
#   m=0 (Continuous background rates).

"""
    PurePacketModel

Encapsulates the complete generative state of the Pure Packet marked point-process:
- `motifs`: Carrier motifs holding spatiotemporal emission packets (m=1, m=2)
- `occurrences`: Discovered sequence occurrences marked by (tau_e, u_e, c_e)
- `assign`: Discrete dual-index spike assignments z_i = (e, p) vs (0, 0)
- `bg_rates`: Continuous baseline rate parameters lambda_{0,c} (m=0)
- `counts_bg`: Pre-allocated buffer for background spike counts
- `ard_state`: Continuous kinematic manifold & ARD precisions (m=4)
- `hyperstate`: Discrete regime tracking hyperstate
- `ou_ctx`: Continuous OU Conjugate Triad context (m=5)
- `audit`: Stratified audit channels for Time-Rescaling theorem validation
- `R_max`: Maximum latent spatial manifold dimension
- `N_channels`: Total physical units (strictly 411)
- `T_total`: Recording span in seconds
- `energy_history`: Vector tracking complete-data joint posterior energy U_t
- `ks_history`: Vector tracking Time-Rescaling Kolmogorov-Smirnov statistics
"""
mutable struct PurePacketModel
    motifs::Dict{Int, PurePacketMotif}
    occurrences::Dict{Int, PureOccurrence}
    assign::Vector{SpikeAssignment}
    bg_rates::Vector{Float64}
    counts_bg::Vector{Int}
    ard_state::ARDManifoldState
    hyperstate::Level5Hyperstate
    ou_ctx::ContinuousOUContext
    audit::NamedTuple
    R_max::Int
    N_channels::Int
    T_total::Float64
    energy_history::Vector{Float64}
    ks_history::Vector{Any}
end

"""
    estimate_isi_tail_baseline(
        data::SpikeDataset;
        sigma_prior::Float64=0.015,
        alpha_bg::Float64=1.0,
        beta_bg::Float64=1.0
    ) -> Vector{Float64}

Memoryless ISI Tail Estimator (Exact Renewal Floor):
Conditioning on inter-spike intervals exceeding the packet kernel span (Delta t > 4 * sigma_prior)
strips out intra-packet bursts while leaving excess intervals strictly exponential with the true
background rate. Yields parameter-free, closed-form MLE without checkpoint leakage:
    hat{lambda}_{0,c} = |{k: Delta t_k > 4 * sigma_prior}| / sum_{k: Delta t_k > 4 * sigma_prior} (Delta t_k - 4 * sigma_prior)
"""
function estimate_isi_tail_baseline(
    data::SpikeDataset;
    sigma_prior::Float64=0.015,
    alpha_bg::Float64=1.0,
    beta_bg::Float64=1.0
)::Vector{Float64}
    N_channels = data.N_channels
    tau_burst = 4.0 * sigma_prior
    lambda_init = zeros(Float64, N_channels)

    for c in 1:N_channels
        spk_idx = data.chan_to_spikes[c]
        if length(spk_idx) < 2
            lambda_init[c] = max(0.001, Float64(length(spk_idx)) / max(data.T_total, 1.0))
            continue
        end
        t_c = data.t[spk_idx]
        excess_count = 0
        excess_sum = 0.0
        for i in 1:(length(t_c) - 1)
            dt = t_c[i + 1] - t_c[i]
            if dt > tau_burst
                excess_count += 1
                excess_sum += (dt - tau_burst)
            end
        end
        if excess_count > 0 && excess_sum > 0.0
            lambda_init[c] = max(0.0001, excess_count / excess_sum)
        else
            # Implementation guard: fast-spiking unit with zero excess intervals
            lambda_init[c] = alpha_bg / beta_bg
        end
    end
    return lambda_init
end

"""
    initialize_scratch_model(
        data::SpikeDataset,
        motifs::Dict{Int, PurePacketMotif};
        bg_rates::Union{Nothing, Vector{Float64}}=nothing,
        R_max::Int=4,
        alpha_c::Float64=10.0,
        nu_student::Float64=4.0
    ) -> PurePacketModel

Initializes a cold-scratch model instance (0 active occurrences, all spikes in background).
If `bg_rates` is provided, it sets the initial Poisson rate floor; otherwise, the closed-form
Memoryless ISI Tail Estimator is used to break the E=0 absorbing boundary adaptively per unit.
"""
function initialize_scratch_model(
    data::SpikeDataset,
    motifs::Dict{Int, PurePacketMotif};
    bg_rates::Union{Nothing, Vector{Float64}}=nothing,
    ard_state::Union{Nothing, ARDManifoldState}=nothing,
    R_max::Int=4,
    R_init::Union{Nothing, Int}=nothing,
    alpha_c::Float64=10.0,
    nu_student::Float64=4.0
)::PurePacketModel
    R_current = R_init !== nothing ? R_init : R_max
    N_channels = data.N_channels
    N_spikes = length(data.t)
    T_total = data.T_total

    # Ensure all motifs have W dimension matching R_current
    for m in values(motifs)
        if size(m.W, 1) != R_current
            P = length(m.packets)
            new_W = zeros(Float64, R_current, P)
            n_copy = min(size(m.W, 1), R_current)
            if n_copy > 0
                @views new_W[1:n_copy, :] .= m.W[1:n_copy, :]
            end
            m.W = new_W
        end
    end

    # Cold scratch state: 0 occurrences, all spikes assigned to background (0, 0)
    occurrences = Dict{Int, PureOccurrence}()
    assign = [SpikeAssignment(0, 0) for _ in 1:N_spikes]
    counts_bg = zeros(Int, N_channels)

    # Initialize baseline background rates lambda_{0, c}
    init_bg = zeros(Float64, N_channels)
    if bg_rates !== nothing
        @assert length(bg_rates) == N_channels "Dimension mismatch: provided bg_rates has length $(length(bg_rates)), expected $N_channels"
        init_bg .= max.(0.0001, bg_rates)
    else
        # Closed-form Memoryless ISI Tail Estimator (Exact Renewal Floor)
        init_bg .= estimate_isi_tail_baseline(data; sigma_prior=0.015)
    end

    # Setup ARD Manifold (alpha_prune = 6,500 s^-2) & Continuous OU Triad
    ard_state_obj = if ard_state !== nothing
        deepcopy(ard_state)
    else
        ARDManifoldState(R_current; a0=1e-3, b0=1.0e-6, prune_threshold=6500.0)
    end
    M_capacity = max(maximum(keys(motifs); init=1) + 150, 250)
    hyperstate = Level5Hyperstate(1, R_current, M_capacity; force_thresh=0.01)
    ou_ctx = ContinuousOUContext(
        N_channels, T_total;
        M_motifs=M_capacity,
        cp_range=99:301,
        dt_net=0.05, tau_net=0.5, sigma_net=0.4,
        dt_cell=10.0, tau_cell=60.0, sigma_cell=0.15,
        dt_motif=5.0, tau_motif=20.0, sigma_motif=0.3,
        alpha_c=alpha_c, nu_student=nu_student
    )

    # Stratified Audit Channels for Brown et al. (2002) Time-Rescaling Theorem
    audit = select_stratified_audit_channels(motifs, init_bg, N_channels; n_signal=10, n_quiet=10)

    println(@sprintf("Cold Scratch Model Initialized: %d Motifs, 0 Occurrences, %d Background Spikes (mean bg_rate = %.4f Hz).",
            length(motifs), N_spikes, mean(init_bg)))
    return PurePacketModel(
        motifs, occurrences, assign, init_bg, counts_bg,
        ard_state_obj, hyperstate, ou_ctx, audit,
        R_max, N_channels, T_total,
        Float64[], Any[]
    )
end

"""
    resume_model_from_checkpoint(ck_path::String, data::SpikeDataset; R_max::Int=4) -> PurePacketModel

Restores a complete PurePacketModel state from a serialized checkpoint .jls file.
"""
function resume_model_from_checkpoint(ck_path::String, data::SpikeDataset; R_max::Int=4)::PurePacketModel
    raw = open(ck_path, "r") do io
        deserialize(io)
    end
    N_channels = data.N_channels
    T_total = data.T_total

    counts_bg = zeros(Int, N_channels)
    for (i, as) in enumerate(raw.assign)
        if as.event_id == 0
            counts_bg[data.neuron[i]] += 1
        end
    end

    audit = select_stratified_audit_channels(raw.motifs, raw.bg_rates, N_channels; n_signal=10, n_quiet=10)

    println(@sprintf("Model Resumed from Checkpoint: %d Motifs, %d Occurrences, %d Background Spikes (mean bg_rate = %.4f Hz).",
            length(raw.motifs), length(raw.occurrences), sum(counts_bg), mean(raw.bg_rates)))

    return PurePacketModel(
        raw.motifs,
        raw.occurrences,
        raw.assign,
        raw.bg_rates,
        counts_bg,
        raw.ard_state,
        raw.hyperstate,
        raw.ou_ctx,
        audit,
        R_max,
        N_channels,
        T_total,
        raw.energy_history,
        raw.ks_history
    )
end

"""
    allocate_thread_buffers(max_threads::Int=Threads.maxthreadid(); rng=Random.default_rng())

Allocates zero-allocation thread-local buffers and PRNGs for parallel spike resampling.
"""
function allocate_thread_buffers(max_threads::Int=Threads.maxthreadid(); rng::AbstractRNG=Random.default_rng())
    tl_occ = [Int[] for _ in 1:max_threads]
    tl_pkt = [Int[] for _ in 1:max_threads]
    tl_logw = [Float64[] for _ in 1:max_threads]
    for tid in 1:max_threads
        sizehint!(tl_occ[tid], 128)
        sizehint!(tl_pkt[tid], 128)
        sizehint!(tl_logw[tid], 128)
    end
    tl_rng = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:max_threads]
    return (tl_occ = tl_occ, tl_pkt = tl_pkt, tl_logw = tl_logw, tl_rng = tl_rng)
end

"""
    allocate_birth_buffers(chan_to_spikes::Vector{Vector{Int}})

Pre-allocates reusable buffers for continuous mixture occurrence birth proposals.
"""
function allocate_birth_buffers(chan_to_spikes::Vector{Vector{Int}})
    N_channels = length(chan_to_spikes)
    bg_times = [Float64[] for _ in 1:N_channels]
    bg_indices = [Int[] for _ in 1:N_channels]
    for c in 1:N_channels
        sizehint!(bg_times[c], length(chan_to_spikes[c]))
        sizehint!(bg_indices[c], length(chan_to_spikes[c]))
    end
    return (bg_times = bg_times, bg_indices = bg_indices)
end

"""
    step_pure_packet_sweep!(
        sweep::Int,
        motifs::Dict{Int, PurePacketMotif},
        occurrences::Dict{Int, PureOccurrence},
        assign::Vector{SpikeAssignment},
        spike_t::Vector{Float64},
        spike_neuron::Vector{Int},
        chan_to_spikes::Vector{Vector{Int}},
        bg_rates::Vector{Float64},
        counts_bg::Vector{Int},
        ard_state::ARDManifoldState,
        ou_ctx::ContinuousOUContext,
        T_total::Float64,
        R_max::Int,
        tl_occ, tl_pkt, tl_logw, tl_rng;
        bg_times, bg_indices,
        gamma_m::Float64=100.0,
        alpha_c::Float64=10.0,
        beta_c::Float64=10.0,
        delta_margin::Float64=3.0,
        rng::AbstractRNG=Random.default_rng()
    ) -> (n_birthed::Int, n_overwritten::Int, n_pruned::Int)

Executes one complete MCMC sweep across all active universal m-mechanisms.
"""
function step_pure_packet_sweep!(
    sweep::Int,
    motifs::Dict{Int, PurePacketMotif},
    occurrences::Dict{Int, PureOccurrence},
    assign::Vector{SpikeAssignment},
    spike_t::Vector{Float64},
    spike_neuron::Vector{Int},
    chan_to_spikes::Vector{Vector{Int}},
    bg_rates::Vector{Float64},
    counts_bg::Vector{Int},
    ard_state::ARDManifoldState,
    ou_ctx::ContinuousOUContext,
    T_total::Float64,
    R_max::Int,
    tl_occ, tl_pkt, tl_logw, tl_rng;
    bg_times, bg_indices,
    gamma_m::Float64=100.0,
    alpha_c::Float64=10.0,
    beta_c::Float64=10.0,
    alpha_a::Float64=1.0,
    beta_a::Float64=1.0,
    alpha_m::Float64=0.05,
    beta_m::Float64=1.0,
    delta_margin::Float64=3.0,
    frozen_motif_ids::Set{Int}=Set{Int}(),
    allow_motif_birth::Bool=false,
    rng::AbstractRNG=Random.default_rng()
)
    N_channels = length(bg_rates)

    # 0. Mechanism m=3: Dynamic Motif Archetype Birth (M -> M+1) from Background Spikes
    n_motifs_birthed = 0
    if allow_motif_birth
        n_motifs_birthed = detect_and_birth_new_motifs!(
            motifs, occurrences, assign, spike_t, spike_neuron,
            bg_rates, T_total, length(ard_state.active_mask);
            N_channels=N_channels, occam_motif=nothing, gamma_motif=gamma_m,
            alpha_a=alpha_a, beta_a=beta_a, alpha_c=alpha_c, beta_c=beta_c,
            alpha_m=alpha_m, beta_m=beta_m,
            counts_bg=counts_bg, ou_ctx=ou_ctx, rng=rng
        )
    end

    # 1. Mechanism m=2: Occurrence Candidate Birth Proposals
    n_birthed, n_overwritten = detect_and_birth_occurrences!(
        occurrences, motifs, spike_t, spike_neuron, chan_to_spikes,
        assign, bg_rates, T_total, R_max;
        active_mask=ard_state.active_mask,
        alpha_m=alpha_m, beta_m=beta_m,
        gamma_m=gamma_m, alpha_c=alpha_c, beta_c=beta_c, delta_margin=delta_margin,
        bg_times=bg_times, bg_indices=bg_indices,
        ou_ctx=ou_ctx, rng=rng
    )

    # 2. Mechanism m=2: Marginal Negative Binomial Occurrence Pruning
    pruned_occ_ids = prune_dead_occurrences!(
        occurrences, motifs, spike_t, assign, bg_rates, T_total, ard_state.active_mask;
        chan_to_spikes=chan_to_spikes, counts_bg=counts_bg,
        alpha_c=alpha_c, beta_c=beta_c, alpha_m=alpha_m, beta_m=beta_m, gamma_m=gamma_m,
        ou_ctx=ou_ctx, rng=rng
    )
    n_pruned = length(pruned_occ_ids)

    # 3. Sampler: Channel-First Routed Dual-Index Spike Resampling (Multi-threaded)
    chan_to_motifs = update_chan_to_motifs(motifs, N_channels)
    motif_occs = Dict{Int, Vector{PureOccurrence}}()
    motif_taus = Dict{Int, Vector{Float64}}()
    for mid in keys(motifs)
        occs_m = [occ for occ in values(occurrences) if occ.motif == mid]
        sort!(occs_m, by=occ -> occ.tau)
        motif_occs[mid] = occs_m
        motif_taus[mid] = [occ.tau for occ in occs_m]
    end

    resample_all_spikes_pure_packet_threaded!(
        assign, spike_t, spike_neuron,
        chan_to_motifs, motif_occs, motif_taus,
        occurrences, motifs, bg_rates, ard_state.active_mask,
        tl_occ, tl_pkt, tl_logw, tl_rng;
        ou_ctx=ou_ctx
    )

    # Clean up dead and hollowed-out occurrences (N_active_pkts < 2)
    prune_dead_occurrences!(
        occurrences, motifs, spike_t, assign, bg_rates, T_total, ard_state.active_mask;
        chan_to_spikes=chan_to_spikes, counts_bg=counts_bg,
        alpha_c=alpha_c, beta_c=beta_c, alpha_m=alpha_m, beta_m=beta_m,
        gamma_m=gamma_m, ou_ctx=ou_ctx, rng=rng
    )

    # 4. Mechanism m=4: Occurrence Coordinates u_e with Continuous Student-t Precisions
    for (eid, occ) in occurrences
        pm = motifs[occ.motif]
        P = length(pm.packets)
        res = zeros(Float64, P)
        prec = zeros(Float64, P)
        for p in 1:P
            c = occ.counts[p]
            if c > 0
                mean_t = occ.sum_t[p] / c
                res[p] = mean_t - (occ.tau + pm.packets[p].mu)
                prec[p] = c / pm.packets[p].var
            end
        end
        tau_prec = get(ou_ctx.tau_precisions, eid, 1.0)
        sample_occurrence_coordinate!(occ.u, pm, ard_state.active_mask, res, prec; tau_precision=tau_prec, rng=rng)
    end

    # 5. Mechanism m=4 & m=3: Translational Gauge Loadings W
    for (mid, pm) in motifs
        occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
        isempty(occ_list) && continue
        P = length(pm.packets)
        E_m = length(occ_list)
        u_list = [occ.u for occ in occ_list]
        res_list = Vector{Vector{Float64}}(undef, E_m)
        prec_list = Vector{Vector{Float64}}(undef, E_m)
        for (idx, occ) in enumerate(occ_list)
            res = zeros(Float64, P)
            prec = zeros(Float64, P)
            for p in 1:P
                c = occ.counts[p]
                if c > 0
                    mean_t = occ.sum_t[p] / c
                    res[p] = mean_t - (occ.tau + pm.packets[p].mu)
                    prec[p] = c / pm.packets[p].var
                end
            end
            res_list[idx] = res
            prec_list[idx] = prec
        end
        sample_packet_loadings!(pm, ard_state.alpha, ard_state.active_mask, u_list, res_list, prec_list; rng=rng)
    end

    # 6. Mechanism m=4: Global ARD Precisions, Pruning, & Dynamic Birth
    all_pm = collect(values(motifs))
    sample_global_ard_precisions!(ard_state, all_pm; rng=rng)
    prune_inactive_axes!(ard_state, all_pm)
    if sweep >= 2
        check_and_birth_axis!(ard_state, motifs, collect(values(occurrences)); rng=rng)
    end

    # 7. Mechanism m=2: Occurrence Onsets tau_e
    for occ in values(occurrences)
        pm = motifs[occ.motif]
        resample_occurrence_onset!(occ, pm, ard_state.active_mask; T_max=T_total, rng=rng)
    end

    # 8. Mechanism m=1 & m=3: Corridor Latencies mu_p & Jitter var_p with Origin Anchor
    for (mid, pm) in motifs
        mid in frozen_motif_ids && continue
        occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
        isempty(occ_list) && continue
        update_packet_corridors!(pm, occ_list, ard_state.active_mask; anchor_origin=true, damp=0.2)
        project_translational_gauge!(pm, ard_state.active_mask)
    end

    # 9. Mechanism m=1: Packet Masses a_{m,p} (Conjugate Gamma)
    for (mid, pm) in motifs
        mid in frozen_motif_ids && continue
        occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
        isempty(occ_list) && continue
        sample_packet_masses!(pm, occ_list; alpha_a=1.0, beta_a=1.0, rng=rng)
    end

    # 10. Mechanism m=1: Reversible-Jump Pruning of Inactive Packets (Detailed Balance Reverse Jump)
    for (mid, pm) in motifs
        mid in frozen_motif_ids && continue
        occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
        isempty(occ_list) && continue
        prune_dead_packets!(
            pm, occ_list, bg_rates, N_channels;
            spike_t=spike_t, spike_chan=spike_neuron, assign=assign,
            ou_ctx=ou_ctx, active_mask=ard_state.active_mask, ard_alphas=ard_state.alpha,
            sigma_prior=0.015, alpha_a=alpha_a, beta_a=beta_a,
            counts_bg=counts_bg, rng=rng
        )
        project_translational_gauge!(pm, ard_state.active_mask)
    end

    # 11. Mechanism m=1 & m=3: Nonparametric Continuous Bayes Factor Birth & Split
    if sweep >= 2
        for (mid, pm) in motifs
            mid in frozen_motif_ids && continue
            occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
            isempty(occ_list) && continue
            detect_and_birth_or_split_packets!(
                pm, occ_list, spike_t, spike_neuron, assign, bg_rates, N_channels;
                chan_spikes=chan_to_spikes, ou_ctx=ou_ctx, sigma_prior=0.015,
                alpha_a=alpha_a, beta_a=beta_a,
                active_mask=ard_state.active_mask, ard_alphas=ard_state.alpha
            )
            project_translational_gauge!(pm, ard_state.active_mask)
        end
    end

    # 11b. Mechanism m=3: Reversible-Jump Pruning of Dead & Sub-Marginal Singleton Motifs
    pruned_mids = prune_dead_motifs!(
        motifs, occurrences, frozen_motif_ids;
        spike_t=spike_t, spike_neuron=spike_neuron, assign=assign,
        bg_rates=bg_rates, counts_bg=counts_bg, active_mask=ard_state.active_mask,
        ou_ctx=ou_ctx, T_total=T_total,
        gamma_motif=gamma_m, alpha_c=alpha_c, beta_c=beta_c, beta_a=beta_a,
        alpha_m=alpha_m, beta_m=beta_m, sigma_prior=0.015, rng=rng
    )

    # 12. Mechanism m=5 & m=0: Conjugate Triad Continuous Adaptation
    update_regional_ou!(ou_ctx, assign, spike_t, spike_neuron, bg_rates; max_iter=2)
    update_cell_ou!(ou_ctx, assign, spike_t, spike_neuron, bg_rates; max_iter=2)
    update_motif_ou!(ou_ctx, occurrences, motifs; max_iter=2)

    update_continuous_background_rates!(
        bg_rates, ou_ctx, counts_bg, assign, spike_neuron, T_total;
        alpha_bg=1.0, beta_bg=1.0, rng=rng
    )

    update_vigor_shape!(ou_ctx, occurrences; max_iter=10, tol=1e-6)
    sample_continuous_occurrence_amplitudes!(occurrences, motifs, ou_ctx.alpha_c; rng=rng)
    sample_student_t_precisions!(ou_ctx, occurrences, count(ard_state.active_mask); rng=rng)

    # Dynamic Bounding Corridor Update
    update_all_motif_bounds!(motifs)

    return (n_birthed = n_birthed, n_overwritten = n_overwritten, n_pruned = n_pruned)
end

"""
    persist_checkpoint!(path::String, model::PurePacketModel)

Serializes the complete PurePacketModel state to a native .jls file.
"""
function persist_checkpoint!(path::String, model::PurePacketModel)
    isempty(path) && return
    open(path, "w") do io
        serialize(io, (
            motifs = model.motifs,
            occurrences = model.occurrences,
            assign = model.assign,
            bg_rates = model.bg_rates,
            ard_state = model.ard_state,
            hyperstate = model.hyperstate,
            ou_ctx = model.ou_ctx,
            energy_history = model.energy_history,
            ks_history = model.ks_history
        ))
    end
end

"""
    step_pure_packet_sweep!(
        model::PurePacketModel,
        data::SpikeDataset,
        sweep::Int;
        thread_bufs=nothing,
        birth_bufs=nothing,
        gamma_m::Float64=100.0,
        alpha_c::Float64=10.0,
        beta_c::Float64=10.0,
        delta_margin::Float64=3.0,
        frozen_motif_ids::Set{Int}=Set{Int}(),
        allow_motif_birth::Bool=false,
        rng::AbstractRNG=Random.default_rng()
    ) -> (n_birthed::Int, n_overwritten::Int, n_pruned::Int)

Convenience dispatch taking `PurePacketModel` and `SpikeDataset`.
"""
function step_pure_packet_sweep!(
    model::PurePacketModel,
    data::SpikeDataset,
    sweep::Int;
    thread_bufs=nothing,
    birth_bufs=nothing,
    gamma_m::Float64=100.0,
    alpha_c::Float64=10.0,
    beta_c::Float64=10.0,
    alpha_a::Float64=1.0,
    beta_a::Float64=1.0,
    alpha_m::Float64=0.05,
    beta_m::Float64=1.0,
    delta_margin::Float64=3.0,
    frozen_motif_ids::Set{Int}=Set{Int}(),
    allow_motif_birth::Bool=false,
    rng::AbstractRNG=Random.default_rng()
)
    t_bufs = thread_bufs !== nothing ? thread_bufs : allocate_thread_buffers(Threads.maxthreadid(); rng=rng)
    b_bufs = birth_bufs !== nothing ? birth_bufs : allocate_birth_buffers(data.chan_to_spikes)

    return step_pure_packet_sweep!(
        sweep, model.motifs, model.occurrences, model.assign,
        data.t, data.neuron, data.chan_to_spikes,
        model.bg_rates, model.counts_bg, model.ard_state, model.ou_ctx,
        data.T_total, model.R_max,
        t_bufs.tl_occ, t_bufs.tl_pkt, t_bufs.tl_logw, t_bufs.tl_rng;
        bg_times=b_bufs.bg_times, bg_indices=b_bufs.bg_indices,
        gamma_m=gamma_m, alpha_c=alpha_c, beta_c=beta_c,
        alpha_a=alpha_a, beta_a=beta_a, alpha_m=alpha_m, beta_m=beta_m,
        delta_margin=delta_margin,
        frozen_motif_ids=frozen_motif_ids, allow_motif_birth=allow_motif_birth,
        rng=rng
    )
end

"""
    fit!(
        model::PurePacketModel,
        data::SpikeDataset;
        sweeps::Int=50,
        checkpoint_path::String="",
        checkpoint_interval::Int=5,
        frozen_motif_ids::Set{Int}=Set{Int}(),
        allow_motif_birth::Bool=false,
        rng::AbstractRNG=Random.default_rng()
    )

High-level MCMC fitting loop:
- Allocates thread and birth buffers.
- Prints telemetry progress table.
- Runs `step_pure_packet_sweep!` across `1:sweeps`.
- Periodically evaluates Brown et al. (2002) Time-Rescaling theorem and serializes checkpoints.
- Prints final summary report across all active universal mechanisms.
"""
function fit!(
    model::PurePacketModel,
    data::SpikeDataset;
    sweeps::Int=50,
    start_sweep::Int=1,
    checkpoint_path::String="",
    checkpoint_interval::Int=5,
    frozen_motif_ids::Set{Int}=Set{Int}(),
    allow_motif_birth::Bool=false,
    rng::AbstractRNG=Random.default_rng()
)
    # Pre-allocate sweep buffers
    thread_bufs = allocate_thread_buffers(Threads.maxthreadid(); rng=rng)
    birth_bufs = allocate_birth_buffers(data.chan_to_spikes)

    println(@sprintf("\nStarting Sweeps %d to %d on marked point-process (T = %.2f s)...", start_sweep, sweeps, data.T_total))
    print_diagnostics_header()

    try
        for sweep in start_sweep:sweeps
            t0_sw = time()

            # Execute single complete MCMC sweep
            res = step_pure_packet_sweep!(
                sweep, model.motifs, model.occurrences, model.assign,
                data.t, data.neuron, data.chan_to_spikes,
                model.bg_rates, model.counts_bg, model.ard_state, model.ou_ctx,
                data.T_total, model.R_max,
                thread_bufs.tl_occ, thread_bufs.tl_pkt, thread_bufs.tl_logw, thread_bufs.tl_rng;
                bg_times=birth_bufs.bg_times, bg_indices=birth_bufs.bg_indices,
                frozen_motif_ids=frozen_motif_ids, allow_motif_birth=allow_motif_birth,
                rng=rng
            )

            dt_sw = time() - t0_sw

            # Compute objective sweep diagnostics
            diag = compute_sweep_diagnostics(
                sweep, model.motifs, model.occurrences, model.assign,
                model.bg_rates, model.ard_state, data.T_total,
                data.total_spikes_per_chan, dt_sw;
                n_birthed=res.n_birthed, n_pruned=res.n_pruned, ou_ctx=model.ou_ctx
            )
            push!(model.energy_history, diag.joint_energy)
            print_diagnostics_row(diag)

            if sweep <= 2 || sweep % checkpoint_interval == 0
                println(@sprintf("   --> [OU Gauge Diagnostic] mean(bg_rates)=%.4f Hz | x_reg=[%.3f, %.3f] | dx_cell=[%.3f, %.3f] | alpha_c=%.2f",
                        mean(model.bg_rates), minimum(model.ou_ctx.x_region), maximum(model.ou_ctx.x_region),
                        minimum(model.ou_ctx.delta_x), maximum(model.ou_ctx.delta_x), model.ou_ctx.alpha_c))
            end

            # Periodic Time-Rescaling check and checkpoint persistence
            if sweep % checkpoint_interval == 0 || sweep == sweeps
                # Unbiased global census across 100% of channels and spikes (Brown et al. 2002)
                all_channels = collect(1:data.N_channels)
                ks_res = compute_time_rescaling_ks(
                    model.motifs, model.occurrences, data.t, data.chan_to_spikes,
                    model.bg_rates, all_channels, model.ard_state.active_mask;
                    ou_ctx=model.ou_ctx
                )
                push!(model.ks_history, (sweep = sweep, D_ks = ks_res.D_ks, D_95 = ks_res.D_95, is_valid = ks_res.is_valid, N = ks_res.N_intervals))
                println(@sprintf("   --> [Global Time-Rescaling Audit] D_KS: %.5f | 95%% Envelope: %.5f (N=%d across %d units) | Status: %s",
                        ks_res.D_ks, ks_res.D_95, ks_res.N_intervals, data.N_channels, ks_res.is_valid ? "SATURATED (Inside 95% Envelope)" : "EXPLORING"))

                if !isempty(checkpoint_path)
                    persist_checkpoint!(checkpoint_path, model)
                end

                if sweep >= 25 && length(model.ks_history) >= 3
                    d1 = abs(model.ks_history[end].D_ks - model.ks_history[end-1].D_ks) / 5.0
                    d2 = abs(model.ks_history[end-1].D_ks - model.ks_history[end-2].D_ks) / 5.0
                    max_deriv = max(d1, d2)
                    if max_deriv < 0.0003
                        println(@sprintf("   --> [Saturation Monitor] Stationary biological floor reached: D_KS = %.5f (|d D_KS/dt| = %.6f)", ks_res.D_ks, max_deriv))
                    end
                end
            end
        end
    finally
        if !isempty(checkpoint_path)
            persist_checkpoint!(checkpoint_path, model)
            println("\nCheckpoint safely persisted to: ", checkpoint_path)
        end
    end

    println("\n=================================================================")
    println("MCMC Execution Completed!")
    println("=================================================================")
    final_spikes = count(as -> as.event_id > 0, model.assign)
    N_spikes = length(data.t)
    println(@sprintf("Occurrences Discovered from Scratch: %d", length(model.occurrences)))
    println(@sprintf("Spikes Assigned: %d (%.2f%% foreground mass)", final_spikes, (final_spikes / max(N_spikes, 1)) * 100.0))

    println("\nLevel m=5 Conjugate Triad Final State:")
    println(@sprintf("  Leg 1 Network Arousal: MOp range [%.3f, %.3f] (var=%.4f) | CP range [%.3f, %.3f] (var=%.4f)",
            minimum(model.ou_ctx.x_region[1, :]), maximum(model.ou_ctx.x_region[1, :]), var(model.ou_ctx.x_region[1, :]),
            minimum(model.ou_ctx.x_region[2, :]), maximum(model.ou_ctx.x_region[2, :]), var(model.ou_ctx.x_region[2, :])))
    println(@sprintf("  Leg 1 Cell Micro-Drift: Mean single-cell var = %.4f across %d units",
            mean([var(model.ou_ctx.delta_x[c, :]) for c in 1:model.N_channels]), model.N_channels))
    println(@sprintf("  Leg 2 Motor Vigor Pin: Learned shape alpha_c = %.4f (beta_c == alpha_c => E[c_e] == 1.0)", model.ou_ctx.alpha_c))
    if !isempty(model.ou_ctx.tau_precisions)
        prec_vals = collect(values(model.ou_ctx.tau_precisions))
        println(@sprintf("  Leg 3 Student-t Kinematics: Precisions tau_e: mean=%.3f, min=%.3f, max=%.3f (nu=%.1f)",
                mean(prec_vals), minimum(prec_vals), maximum(prec_vals), model.ou_ctx.nu_student))
    end
    println(@sprintf("Active Kinematic Axes: %d (alpha = %s)", count(model.ard_state.active_mask), string(round.(model.ard_state.alpha[model.ard_state.active_mask], digits=1))))

    return (model = model, energy_history = model.energy_history, ks_history = model.ks_history)
end
