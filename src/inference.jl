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
    delta_margin::Float64=3.0,
    rng::AbstractRNG=Random.default_rng()
)
    N_channels = length(bg_rates)

    # 1. Mechanism m=2: Occurrence Candidate Birth Proposals
    n_birthed, n_overwritten = detect_and_birth_occurrences!(
        occurrences, motifs, spike_t, spike_neuron, chan_to_spikes,
        assign, bg_rates, T_total, R_max;
        active_mask=ard_state.active_mask,
        gamma_m=gamma_m, alpha_c=alpha_c, beta_c=beta_c, delta_margin=delta_margin,
        bg_times=bg_times, bg_indices=bg_indices,
        rng=rng
    )

    # 2. Mechanism m=2: Marginal Negative Binomial Occurrence Pruning
    pruned_occ_ids = prune_dead_occurrences!(
        occurrences, motifs, spike_t, assign, bg_rates, T_total, ard_state.active_mask;
        alpha_c=alpha_c, beta_c=beta_c, gamma_m=gamma_m
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

    # Clean up empty occurrences
    empty_eids = Int[]
    for (eid, occ) in occurrences
        if isempty(occ.members)
            push!(empty_eids, eid)
        end
    end
    for eid in empty_eids
        delete!(occurrences, eid)
    end

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
        occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
        isempty(occ_list) && continue
        update_packet_corridors!(pm, occ_list, ard_state.active_mask; anchor_origin=true, damp=0.2)
        project_translational_gauge!(pm, ard_state.active_mask)
    end

    # 9. Mechanism m=1: Packet Masses a_{m,p} (Conjugate Gamma)
    for (mid, pm) in motifs
        occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
        isempty(occ_list) && continue
        sample_packet_masses!(pm, occ_list; alpha_a=1.0, beta_a=1.0, rng=rng)
    end

    # 10. Mechanism m=1 & m=3: Nonparametric Pruning with Anchor Handover
    for (mid, pm) in motifs
        occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
        isempty(occ_list) && continue
        prune_dead_packets!(pm, occ_list, bg_rates, N_channels; assign=assign, occam_tax=2.0)
        project_translational_gauge!(pm, ard_state.active_mask)
    end

    # 11. Mechanism m=1 & m=3: Nonparametric Continuous Bayes Factor Birth & Split
    if sweep >= 2
        for (mid, pm) in motifs
            occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
            isempty(occ_list) && continue
            detect_and_birth_or_split_packets!(
                pm, occ_list, spike_t, spike_neuron, assign, bg_rates, N_channels;
                chan_spikes=chan_to_spikes, sigma_prior=0.015, alpha_a=1.0, beta_a=1.0,
                active_mask=ard_state.active_mask, ard_alphas=ard_state.alpha
            )
            project_translational_gauge!(pm, ard_state.active_mask)
        end
    end

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
