# run_awake_scratch.jl
# High-Performance Runner for Pure Packet Marked Point Process Engine
# Scratch Start (0 Initial Occurrences) on Pure-Neural N=411 Units (Awake Task: T=2114.28s)

using Random, Printf, NPZ, Serialization, Statistics, LinearAlgebra

include(joinpath(@__DIR__, "..", "src", "pure_packet_engine.jl"))

const DATA_PATH = joinpath(@__DIR__, "..", "data", "awake.npz")
const CHECKPOINT_PATH = joinpath(@__DIR__, "..", "checkpoints", "checkpoint_pure_packet_scratch_50sweeps.jls")

neural_spike_indices = findall(c -> 1 <= c <= N_channels, b.neuron)
spike_t = b.t[neural_spike_indices]
spike_neuron = b.neuron[neural_spike_indices]
N_spikes = length(spike_t)
rng = MersenneTwister(42)

println(@sprintf("Domain Hygiene Enforced: Strictly N=%d pure neural units (MOp: 208, CP: 203)", N_channels))
println(@sprintf("Spike stream stripped of behavioral channels: %d / %d spikes retained (%.2f%%)",
        N_spikes, length(b.t), (N_spikes / length(b.t)) * 100.0))

# Pre-allocate zero-allocation background buffers
counts_bg = zeros(Int, N_channels)
bg_rates = [b.bg_rate[b.channel_group[k]] * b.bg_pi[k] for k in 1:N_channels]

# Precompute channel to spike indices map
chan_to_spikes = [Int[] for _ in 1:N_channels]
for (idx, neuron_id) in enumerate(spike_neuron)
    push!(chan_to_spikes[neuron_id], idx)
end

# 1. Convert Motifs to PurePacketMotif (Strictly Pure Neural Units <= 411)
motifs = Dict{Int, PurePacketMotif}()
motif_origin_shifts = Dict{Int, Float64}()

for (mid, m) in b.motifs
    active_chans = Int[]
    for k in 1:N_channels
        if m.pi[k] >= 0.001
            push!(active_chans, k)
        end
    end
    sort!(active_chans, by=k -> m.mu[k])
    
    orig_shift = !isempty(active_chans) ? m.mu[active_chans[1]] : 0.0
    motif_origin_shifts[mid] = orig_shift

    pkts = Packet[]
    for (p, k) in enumerate(active_chans)
        g = b.channel_group[k]
        mass = m.pi[k] * (b.amp_a[g] / b.amp_b[g])
        push!(pkts, Packet(p, k, m.mu[k] - orig_shift, m.var[k], max(mass, 0.01)))
    end

    pm = PurePacketMotif(mid, pkts, R_max, N_channels)
    motifs[mid] = pm
end

# COLD SCRATCH INITIALIZATION: 0 Occurrences, ALL Spikes in Background
occurrences = Dict{Int, PureOccurrence}()
assign = [SpikeAssignment(0, 0) for _ in 1:N_spikes]

println("Cold Scratch Initialized: 0 Occurrences, 0 Assigned Spikes, $(N_spikes) Background Spikes.")

# Setup ARD Manifold State (Cramér-Rao Manifold Pruning alpha_prune = 6,500 s^-2)
ard_state = ARDManifoldState(R_max; a0=1e-3, b0=1.0e-6, prune_threshold=6500.0)
hyperstate = Level5Hyperstate(1, R_max, max(maximum(keys(motifs)), 50); force_thresh=0.01)

# Level m=5 Continuous Ornstein-Uhlenbeck Context (Legs 1, 2, 3)
ou_ctx = ContinuousOUContext(
    N_channels, b.T;
    M_motifs=max(maximum(keys(motifs)), 50),
    cp_range=99:301,
    dt_net=0.05, tau_net=0.5, sigma_net=0.4,
    dt_cell=10.0, tau_cell=60.0, sigma_cell=0.15,
    dt_motif=5.0, tau_motif=20.0, sigma_motif=0.3,
    alpha_c=10.0, nu_student=4.0
)

# Thread-local buffers & PRNGs for parallel spike resampling
max_t = Threads.maxthreadid()
tl_occ = [Int[] for _ in 1:max_t]
tl_pkt = [Int[] for _ in 1:max_t]
tl_logw = [Float64[] for _ in 1:max_t]
for tid in 1:max_t
    sizehint!(tl_occ[tid], 128)
    sizehint!(tl_pkt[tid], 128)
    sizehint!(tl_logw[tid], 128)
end
tl_rng = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:max_t]
println(@sprintf("Multithreading Initialized: %d threads active (max thread ID: %d)", Threads.nthreads(), max_t))

N_sweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 50
total_spikes_per_chan = [length(chan_to_spikes[k]) for k in 1:N_channels]
energy_history = Float64[]
ks_history = []

# Initialize Stratified Audit Channels for Time-Rescaling Theorem (Brown et al., 2002)
audit = select_stratified_audit_channels(motifs, bg_rates, N_channels; n_signal=10, n_quiet=10)
println("Stratified Audit Channels Initialized (Time-Rescaling Theorem):")
println("  Signal Channels (10 High-Density): ", audit.signal_channels)
println("  Quiet Channels  (10 Baseline):     ", audit.quiet_channels)

println("\nStarting $(N_sweeps) Sweeps from Cold Scratch Start...")
println("Time-Rescaling Stationarity Auditing (Continuous OU Integration)")
println(@sprintf("%-5s | %-6s | %-7s | %-7s | %-8s | %-12s | %-8s | %-12s | %-12s | %-7s",
        "Sweep", "Events", "Pruned", "Birth", "Spikes", "Total Pkts", "Mean c_e", "Energy U_t", "Max Gauge", "Time(s)"))
println("-"^116)

save_path = joinpath(@__DIR__, "checkpoint_pure_packet_scratch_$(N_sweeps)sweeps.jls")

function persist_checkpoint!(path)
    open(path, "w") do io
        serialize(io, (
            motifs = motifs,
            occurrences = occurrences,
            assign = assign,
            bg_rates = bg_rates,
            ard_state = ard_state,
            hyperstate = hyperstate,
            ou_ctx = ou_ctx,
            energy_history = energy_history,
            ks_history = ks_history
        ))
    end
end

# Preallocated background buffers for continuous mixture birth proposals
bg_times = [Float64[] for _ in 1:N_channels]
bg_indices = [Int[] for _ in 1:N_channels]
for c in 1:N_channels
    sizehint!(bg_times[c], length(chan_to_spikes[c]))
    sizehint!(bg_indices[c], length(chan_to_spikes[c]))
end

try
for sweep in 1:N_sweeps
    t0_sw = time()

    # Step 0 (Cold Start): Birth initial occurrences via Continuous Mixture Proposal
    local n_occ_birthed = 0
    local n_occ_overwritten = 0
    n_occ_birthed, n_occ_overwritten = detect_and_birth_occurrences!(
        occurrences, motifs, spike_t, spike_neuron, chan_to_spikes,
        assign, bg_rates, b.T, R_max;
        active_mask=ard_state.active_mask,
        gamma_m=100.0, alpha_c=10.0, beta_c=10.0, delta_margin=3.0,
        bg_times=bg_times, bg_indices=bg_indices,
        rng=rng
    )

    # Step 1: Exact Marginal Negative Binomial Occurrence Pruning
    local pruned_occ_ids = prune_dead_occurrences!(
        occurrences, motifs, spike_t, assign, bg_rates, b.T, ard_state.active_mask;
        alpha_c=10.0, beta_c=10.0, gamma_m=100.0
    )
    local n_pruned_occ = length(pruned_occ_ids)

    # Step 2: Channel-First Routed Dual-Index Spike Resampling with Continuous Background Intensity
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

    # Step 3: Occurrence Coordinates u_e (Level m=4) with Continuous Student-t Precision (Leg 3)
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

    # Step 4: Translational Gauge Loadings W
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

    # Global ARD Precisions, Pruning, & Dynamic Birth (Level m=4)
    all_pm = collect(values(motifs))
    sample_global_ard_precisions!(ard_state, all_pm; rng=rng)
    prune_inactive_axes!(ard_state, all_pm)
    if sweep >= 2
        check_and_birth_axis!(ard_state, motifs, collect(values(occurrences)); rng=rng)
    end

    # Occurrence Onsets tau_e
    for occ in values(occurrences)
        pm = motifs[occ.motif]
        resample_occurrence_onset!(occ, pm, ard_state.active_mask; T_max=b.T, rng=rng)
    end

    # Corridor Latencies mu_p & Jitter var_p with Origin Anchor
    for (mid, pm) in motifs
        occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
        isempty(occ_list) && continue
        update_packet_corridors!(pm, occ_list, ard_state.active_mask; anchor_origin=true, damp=0.2)
        project_translational_gauge!(pm, ard_state.active_mask)
    end

    # Packet Masses a_{m,p} (Conjugate Gamma)
    for (mid, pm) in motifs
        occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
        isempty(occ_list) && continue
        sample_packet_masses!(pm, occ_list; alpha_a=1.0, beta_a=1.0, rng=rng)
    end

    # Nonparametric Pruning with Anchor Handover
    for (mid, pm) in motifs
        occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
        isempty(occ_list) && continue
        prune_dead_packets!(pm, occ_list, bg_rates, N_channels; assign=assign, occam_tax=2.0)
        project_translational_gauge!(pm, ard_state.active_mask)
    end

    # Nonparametric Continuous Bayes Factor Birth & Split
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

    # Step 6: Level m=5 Conjugate Triad & Continuous OU Adaptation
    # 6a. Leg 1: Regional arousal x_region(t) (MOp & CP) and Single-cell micro-drift delta_x_c(t)
    update_regional_ou!(ou_ctx, assign, spike_t, spike_neuron, bg_rates; max_iter=2)
    update_cell_ou!(ou_ctx, assign, spike_t, spike_neuron, bg_rates; max_iter=2)
    update_motif_ou!(ou_ctx, occurrences, motifs, hyperstate; max_iter=2)

    # 6b. Leg 1: Continuous background rate draws with exact analytical T_eff integrals
    update_continuous_background_rates!(
        bg_rates, ou_ctx, counts_bg, assign, spike_neuron, b.T;
        alpha_bg=1.0, beta_bg=1.0, rng=rng
    )

    # 6c. Leg 2: Continuous motor vigor shape alpha_c (beta_c == alpha_c => E[c_e] == 1.0) & amplitudes
    update_vigor_shape!(ou_ctx, occurrences; max_iter=10, tol=1e-6)
    sample_continuous_occurrence_amplitudes!(occurrences, motifs, ou_ctx.alpha_c; rng=rng)

    # 6d. Leg 3: Continuous Student-t kinematics precision mixture tau_e | u_e
    sample_student_t_precisions!(ou_ctx, occurrences, count(ard_state.active_mask); rng=rng)

    # Update dynamic motif bounding envelopes
    update_all_motif_bounds!(motifs, hyperstate)

    # Diagnostics
    max_gauge_err = 0.0
    total_packets = 0
    for pm in values(motifs)
        P = length(pm.packets)
        total_packets += P
        for r in 1:R_max
            if ard_state.active_mask[r]
                g_err = abs(sum((1.0 / pm.packets[p].var) * pm.W[r, p] for p in 1:P))
                max_gauge_err = max(max_gauge_err, g_err)
            end
        end
    end

    dt_sw = time() - t0_sw
    n_spk = count(as -> as.event_id > 0, assign)
    mean_c_e = isempty(occurrences) ? 0.0 : mean(occ.amplitude for occ in values(occurrences))
    energy_u = compute_joint_energy(motifs, occurrences, total_spikes_per_chan, bg_rates, ard_state, hyperstate, b.T; ou_ctx=ou_ctx)
    push!(energy_history, energy_u)

    println(@sprintf("%-5d | %-6d | %-7d | %-7d | %-8d | %-12d | %-8.4f | %-12.2e | %-12.2e | %-7.2f",
            sweep, length(occurrences), n_pruned_occ, n_occ_birthed, n_spk, total_packets,
            mean_c_e, energy_u, max_gauge_err, dt_sw))

    if sweep <= 2 || sweep % 5 == 0
        println(@sprintf("   --> [OU Gauge Diagnostic] mean(bg_rates)=%.4f Hz | x_reg=[%.3f, %.3f] | dx_cell=[%.3f, %.3f] | alpha_c=%.2f",
                mean(bg_rates), minimum(ou_ctx.x_region), maximum(ou_ctx.x_region),
                minimum(ou_ctx.delta_x), maximum(ou_ctx.delta_x), ou_ctx.alpha_c))
    end

    if sweep % 5 == 0 || sweep == N_sweeps
        ks_res = compute_time_rescaling_ks(
            motifs, occurrences, spike_t, chan_to_spikes, bg_rates, audit.all_audit, ard_state.active_mask;
            ou_ctx=ou_ctx
        )
        push!(ks_history, (sweep = sweep, D_ks = ks_res.D_ks, D_95 = ks_res.D_95, is_valid = ks_res.is_valid, N = ks_res.N_intervals))
        println(@sprintf("   --> [Audit Time-Rescaling] D_KS: %.5f | 95%% Envelope: %.5f (N=%d) | Status: %s",
                ks_res.D_ks, ks_res.D_95, ks_res.N_intervals, ks_res.is_valid ? "SATURATED (Inside 95% Envelope)" : "EXPLORING"))

        # Persist checkpoint continuously
        persist_checkpoint!(save_path)

        # Saturation Stationarity Monitor
        if sweep >= 25 && length(ks_history) >= 3
            d1 = abs(ks_history[end].D_ks - ks_history[end-1].D_ks) / 5.0
            d2 = abs(ks_history[end-1].D_ks - ks_history[end-2].D_ks) / 5.0
            max_deriv = max(d1, d2)
            if max_deriv < 0.0003
                println(@sprintf("   --> [Saturation Monitor] Stationary biological floor reached: D_KS = %.5f (|d D_KS/dt| = %.6f)", ks_res.D_ks, max_deriv))
            end
        end
    end
end
finally
    persist_checkpoint!(save_path)
    println("\nCheckpoint safely persisted to: ", save_path)
end

println("\n=================================================================")
println("From-Scratch Execution Completed!")
println("=================================================================")
final_spikes = count(as -> as.event_id > 0, assign)
println(@sprintf("Occurrences Discovered from Scratch: %d", length(occurrences)))
println(@sprintf("Spikes Assigned: %d (%.2f%% foreground mass)", final_spikes, (final_spikes / N_spikes) * 100.0))

println("\nLevel m=5 Conjugate Triad Final State:")
println(@sprintf("  Leg 1 Network Arousal: MOp range [%.3f, %.3f] (var=%.4f) | CP range [%.3f, %.3f] (var=%.4f)",
        minimum(ou_ctx.x_region[1, :]), maximum(ou_ctx.x_region[1, :]), var(ou_ctx.x_region[1, :]),
        minimum(ou_ctx.x_region[2, :]), maximum(ou_ctx.x_region[2, :]), var(ou_ctx.x_region[2, :])))
println(@sprintf("  Leg 1 Cell Micro-Drift: Mean single-cell var = %.4f across %d units",
        mean([var(ou_ctx.delta_x[c, :]) for c in 1:N_channels]), N_channels))
println(@sprintf("  Leg 2 Motor Vigor Pin: Learned shape alpha_c = %.4f (beta_c == alpha_c => E[c_e] == 1.0)", ou_ctx.alpha_c))
if !isempty(ou_ctx.tau_precisions)
    prec_vals = collect(values(ou_ctx.tau_precisions))
    println(@sprintf("  Leg 3 Student-t Kinematics: Precisions tau_e: mean=%.3f, min=%.3f, max=%.3f (nu=%.1f)",
            mean(prec_vals), minimum(prec_vals), maximum(prec_vals), ou_ctx.nu_student))
end
println(@sprintf("Active Kinematic Axes: %d (alpha = %s)", count(ard_state.active_mask), string(round.(ard_state.alpha[ard_state.active_mask], digits=1))))
