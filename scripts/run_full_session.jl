using Serialization, LinearAlgebra, Statistics, Random, Distributions, StaticArrays, Printf, NPZ

include(joinpath(@__DIR__, "..", "src", "pure_packet_engine.jl"))

println("=================================================================")
println("Level m=0..5 Pure Packet Engine: FULL SESSION (13,833s / 3.84 hrs)")
println("Pure-Neural N=411 & Level m=5 Conjugate Triad Continuous Engine")
println("Spanning Pre-Task, Awake Task, Post-Task Off-Task & Sleep Replay")
println("=================================================================")

# 1. Load Phase 1 Completed Scratch Checkpoint (Pure Neural N=411)
awake_ck_path = joinpath(@__DIR__, "..", "checkpoints", "checkpoint_pure_packet_scratch_50sweeps.jls")
if !isfile(awake_ck_path)
    # Fallback to test directory or Downloads
    awake_ck_path = "/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/manifold_sampler/checkpoint_pure_packet_scratch_50sweeps.jls"
end

println("Loading Phase 1 Motif Dictionary & ARD Manifold from: ", basename(awake_ck_path))
ck_data = open(deserialize, awake_ck_path)

motifs = ck_data.motifs
ard_state = ck_data.ard_state
R_max = ard_state.R_max

# Domain Hygiene: Filter all motif packets to strictly physical neurons <= 411
for (mid, pm) in motifs
    filter!(p -> p.channel <= 411, pm.packets)
    rebuild_inverted_index!(pm, 411)
end

println(@sprintf("Loaded %d Motifs with %d Active ARD Dimensions (alpha_prune = %.1f s^-2):",
        length(motifs), count(ard_state.active_mask), ard_state.prune_threshold))
for (mid, pm) in motifs
    println(@sprintf("  Motif %d: %d packets | Duration W_m = %.3fs", mid, length(pm.packets), pm.W_window))
end

# 2. Load Full Session Pure Neural Spike Dataset (Strictly N=411 Neural Clusters, 0 Behavioral Channels)
data_path = "/home/foaleng/Downloads/mop_awake_ppseq/data/mop_basal_ganglia_all_clusters/spikes_full.npz"

println("\nLoading Full Session Pure Neural Spikes from: ", data_path)
raw_neural = npzread(data_path)
ts_neural = Float64.(raw_neural["timestamp"])
ch_neural = Int.(raw_neural["neuron"])

AWAKE_OFFSET = 3850.3770465001
AWAKE_END = 5964.662297610364

N_channels = 411
neural_mask = (ch_neural .>= 1) .& (ch_neural .<= N_channels)
spike_t = ts_neural[neural_mask]
spike_neurons = ch_neural[neural_mask]

perm = sortperm(spike_t)
spike_t = spike_t[perm]
spike_neurons = spike_neurons[perm]

N_spikes = length(spike_t)
T_total = maximum(spike_t)

println(@sprintf("Full Recording Domain Hygiene: %d Spikes strictly across %d Pure Neural Units | Duration T = %.2f s (%.2f hours)",
        N_spikes, N_channels, T_total, T_total / 3600.0))

rng = MersenneTwister(42)

# Precompute channel to spike indices map
chan_to_spikes = [Int[] for _ in 1:N_channels]
for (idx, neuron_id) in enumerate(spike_neurons)
    push!(chan_to_spikes[neuron_id], idx)
end

# 3. Initialize Background Rates across Full Recording
bg_rates = zeros(Float64, N_channels)
for c in 1:N_channels
    bg_rates[c] = max(0.001, Float64(length(chan_to_spikes[c])) / T_total)
end
counts_bg = zeros(Int, N_channels)

# 4. Initialize Occurrences: Transfer Awake Discoveries + Prepare for Replay Detection
occurrences = Dict{Int, PureOccurrence}()
assign = [SpikeAssignment(0, 0) for _ in 1:N_spikes]

# Transfer occurrences from Phase 1 if available
if haskey(ck_data, :occurrences) && !isempty(ck_data.occurrences)
    println(@sprintf("\nTransferring %d occurrences from Awake Task to Full Coordinates...", length(ck_data.occurrences)))
    for (eid, occ) in ck_data.occurrences
        shifted_tau = occ.tau + AWAKE_OFFSET
        if 0.0 <= shifted_tau <= T_total
            occ_trans = PureOccurrence(eid, occ.motif, shifted_tau, R_max, length(motifs[occ.motif].packets);
                                       amplitude=occ.amplitude, regime_id=occ.regime_id)
            occ_trans.u .= occ.u
            occurrences[eid] = occ_trans
        end
    end
    println(@sprintf("Successfully transferred %d occurrences into [%.1fs, %.1fs]",
            length(occurrences), AWAKE_OFFSET, AWAKE_END))
end

# 5. Level m=5 Continuous Ornstein-Uhlenbeck Context across Full Session (3.84 hrs)
hyperstate = Level5Hyperstate(1, R_max, max(maximum(keys(motifs)), 50); force_thresh=0.01)
ou_ctx = ContinuousOUContext(
    N_channels, T_total;
    M_motifs=max(maximum(keys(motifs)), 50),
    cp_range=99:301,
    dt_net=0.05, tau_net=0.5, sigma_net=0.4,
    dt_cell=10.0, tau_cell=60.0, sigma_cell=0.15,
    dt_motif=5.0, tau_motif=20.0, sigma_motif=0.3,
    alpha_c=10.0, nu_student=4.0
)

# 6. Thread-local Buffers for 16-Thread Dual-Index Resampling
max_t = Threads.maxthreadid()
tl_occ = [Int[] for _ in 1:max_t]
tl_pkt = [Int[] for _ in 1:max_t]
tl_logw = [Float64[] for _ in 1:max_t]
for tid in 1:max_t
    sizehint!(tl_occ[tid], 256)
    sizehint!(tl_pkt[tid], 256)
    sizehint!(tl_logw[tid], 256)
end
tl_rng = [Random.Xoshiro(rand(rng, UInt64)) for _ in 1:max_t]
println(@sprintf("Multithreading Initialized: %d threads active (max thread ID: %d)", Threads.nthreads(), max_t))

# Preallocated background buffers for continuous mixture proposals
bg_times = [Float64[] for _ in 1:N_channels]
bg_indices = [Int[] for _ in 1:N_channels]
for c in 1:N_channels
    sizehint!(bg_times[c], length(chan_to_spikes[c]))
    sizehint!(bg_indices[c], length(chan_to_spikes[c]))
end

total_spikes_per_chan = [length(chan_to_spikes[k]) for k in 1:N_channels]

N_sweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 200
energy_history = Float64[]

println("\n=================================================================")
println(@sprintf("Starting %d Sweeps on FULL SESSION Recording (T = %.1fs)...", N_sweeps, T_total))
println("=================================================================")
println(@sprintf("%-5s | %-6s | %-7s | %-7s | %-8s | %-12s | %-8s | %-12s | %-12s | %-7s",
        "Sweep", "Events", "Pruned", "Birth", "Spikes", "Total Pkts", "Mean c_e", "Energy U_t", "Max Gauge", "Time(s)"))
println("-"^116)

save_path = joinpath(@__DIR__, "checkpoint_pure_packet_full_session_$(N_sweeps)sweeps.jls")

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
            energy_history = energy_history
        ))
    end
end

for sweep in 1:N_sweeps
    t0_sw = time()

    # Step 0: Birth sequence occurrences across full session (Pre-Task + Awake + Off-Task + Sleep Replay)
    local n_occ_birthed = 0
    local n_occ_overwritten = 0
    n_occ_birthed, n_occ_overwritten = detect_and_birth_occurrences!(
        occurrences, motifs, spike_t, spike_neurons, chan_to_spikes,
        assign, bg_rates, T_total, R_max;
        active_mask=ard_state.active_mask,
        gamma_m=100.0, alpha_c=10.0, beta_c=10.0, delta_margin=3.0,
        bg_times=bg_times, bg_indices=bg_indices,
        rng=rng
    )

    # Step 1: Prune dead occurrences
    local pruned_occ_ids = prune_dead_occurrences!(
        occurrences, motifs, spike_t, assign, bg_rates, T_total, ard_state.active_mask;
        alpha_c=10.0, beta_c=10.0, gamma_m=100.0
    )
    local n_pruned_occ = length(pruned_occ_ids)

    # Step 2: Threaded dual-index spike resampling with continuous background log-intensity
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
        assign, spike_t, spike_neurons,
        chan_to_motifs, motif_occs, motif_taus,
        occurrences, motifs, bg_rates, ard_state.active_mask,
        tl_occ, tl_pkt, tl_logw, tl_rng;
        ou_ctx=ou_ctx
    )

    # Clean empty occurrences
    empty_eids = Int[]
    for (eid, occ) in occurrences
        if isempty(occ.members)
            push!(empty_eids, eid)
        end
    end
    for eid in empty_eids
        delete!(occurrences, eid)
    end

    # Step 3: Occurrence Coordinates u_e with Leg 3 Continuous Student-t Precisions
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

    # ARD Manifold Precision Updates
    all_pm = collect(values(motifs))
    sample_global_ard_precisions!(ard_state, all_pm; rng=rng)
    prune_inactive_axes!(ard_state, all_pm)

    # Occurrence Onsets tau_e
    for occ in values(occurrences)
        pm = motifs[occ.motif]
        resample_occurrence_onset!(occ, pm, ard_state.active_mask; T_max=T_total, rng=rng)
    end

    # Packet Corridors mu_p & Jitter var_p
    for (mid, pm) in motifs
        occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
        isempty(occ_list) && continue
        update_packet_corridors!(pm, occ_list, ard_state.active_mask; anchor_origin=true, damp=0.2)
        project_translational_gauge!(pm, ard_state.active_mask)
    end

    # Packet Masses
    for (mid, pm) in motifs
        occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
        isempty(occ_list) && continue
        sample_packet_masses!(pm, occ_list; alpha_a=1.0, beta_a=1.0, rng=rng)
    end

    # Nonparametric Continuous Bayes Factor Birth & Split
    if sweep >= 2
        for (mid, pm) in motifs
            occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
            isempty(occ_list) && continue
            detect_and_birth_or_split_packets!(
                pm, occ_list, spike_t, spike_neurons, assign, bg_rates, N_channels;
                chan_spikes=chan_to_spikes, sigma_prior=0.015, alpha_a=1.0, beta_a=1.0,
                active_mask=ard_state.active_mask, ard_alphas=ard_state.alpha
            )
            project_translational_gauge!(pm, ard_state.active_mask)
        end
    end

    # Step 6: Level m=5 Conjugate Triad & Continuous OU Adaptation
    # 6a. Leg 1: Regional arousal x_region(t) (MOp & CP) and Single-cell micro-drift delta_x_c(t)
    update_regional_ou!(ou_ctx, assign, spike_t, spike_neurons, bg_rates; max_iter=2)
    update_cell_ou!(ou_ctx, assign, spike_t, spike_neurons, bg_rates; max_iter=2)
    update_motif_ou!(ou_ctx, occurrences, motifs, hyperstate; max_iter=2)

    # 6b. Leg 1: Continuous background rate draws with exact analytical T_eff integrals
    update_continuous_background_rates!(
        bg_rates, ou_ctx, counts_bg, assign, spike_neurons, T_total;
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
    total_packets = sum(length(pm.packets) for pm in values(motifs))
    for pm in values(motifs)
        P = length(pm.packets)
        tot_prec = sum(1.0 / pkt.var for pkt in pm.packets)
        for r in 1:R_max
            if ard_state.active_mask[r]
                err = abs(sum((1.0 / pm.packets[p].var) * pm.W[r, p] for p in 1:P) / tot_prec)
                max_gauge_err = max(max_gauge_err, err)
            end
        end
    end

    assigned_spikes = count(as -> as.event_id > 0, assign)
    mean_ce = isempty(occurrences) ? 0.0 : mean(occ.amplitude for occ in values(occurrences))
    energy_val = compute_joint_energy(motifs, occurrences, total_spikes_per_chan, bg_rates, ard_state, hyperstate, T_total; ou_ctx=ou_ctx)
    push!(energy_history, energy_val)

    dt_sw = time() - t0_sw
    println(@sprintf("%-5d | %-6d | %-7d | %-7d | %-8d | %-12d | %-8.4f | %-12.2e | %-12.2e | %-7.2f",
            sweep, length(occurrences), n_pruned_occ, n_occ_birthed,
            assigned_spikes, total_packets, mean_ce, energy_val, max_gauge_err, dt_sw))

    # Persist every 10 sweeps
    if sweep % 10 == 0 || sweep == N_sweeps
        persist_checkpoint!(save_path)
    end
end

println("\n=================================================================")
println("Full Session Execution (200 Sweeps) Completed!")
println("=================================================================")
final_spikes = count(as -> as.event_id > 0, assign)
println(@sprintf("Total Occurrences Discovered across Full 3.84 hrs: %d", length(occurrences)))
println(@sprintf("Spikes Assigned: %d (%.2f%% foreground mass across %d spikes)",
        final_spikes, (final_spikes / N_spikes) * 100.0, N_spikes))

# Sleep Replay Analysis: Check occurrences during NREM intervals
sleep_json_path = "/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/results/overnight_20261006/data/sleep_intervals.json"
if isfile(sleep_json_path)
    println("\nSleep Replay Occurrence Breakdown:")
    import JSON
    sleep_intervals = JSON.parsefile(sleep_json_path)
    let total_sleep_occs = 0
        for (idx, interval) in enumerate(sleep_intervals)
            lo = Float64(interval["start_s"])
            hi = Float64(interval["end_s"])
            st = interval["state"]
            occs_in_sleep = count(occ -> lo <= occ.tau < hi, values(occurrences))
            total_sleep_occs += occs_in_sleep
            println(@sprintf("  Interval %2d [%s]: %7.1fs - %7.1fs (%4.1fs) | Occurrences: %d (Rate: %.2f Hz)",
                    idx, st, lo, hi, hi - lo, occs_in_sleep, occs_in_sleep / (hi - lo)))
        end
        println(@sprintf("Total Replay Occurrences during Annotated Sleep: %d", total_sleep_occs))
    end
end

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

persist_checkpoint!(save_path)
println("\nFinal Full Session Checkpoint Saved: ", save_path)
