# run_full_session.jl
# High-Performance Full Session Pipeline (3.84 hrs / 13,833s)
# Level m=0..5 Pure Packet Engine across Pre-Task, Awake Task, Off-Task & Sleep Replay

using Serialization, LinearAlgebra, Statistics, Random, Distributions, StaticArrays, Printf, NPZ
using PurePacketPPSeq

println("=================================================================")
println("Level m=0..5 Pure Packet Engine: FULL SESSION (13,833s / 3.84 hrs)")
println("Pure-Neural N=411 & Level m=5 Conjugate Triad Continuous Engine")
println("Spanning Pre-Task, Awake Task, Post-Task Off-Task & Sleep Replay")
println("=================================================================")

# 1. Load Phase 1 Completed Scratch Checkpoint (Pure Neural N=411)
awake_ck_path = joinpath(@__DIR__, "..", "checkpoints", "checkpoint_pure_packet_scratch_50sweeps.jls")
if !isfile(awake_ck_path)
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
    println(@sprintf("  Motif %d: %d packets | Physical Span W_eff = %.3fs [%.3fs, %.3fs]", mid, length(pm.packets), pm.delta_right - pm.delta_left, pm.delta_left, pm.delta_right))
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
total_spikes_per_chan = [length(chan_to_spikes[k]) for k in 1:N_channels]

# 3. Initialize Background Rates across Full Recording
bg_rates = zeros(Float64, N_channels)
for c in 1:N_channels
    bg_rates[c] = max(0.001, Float64(length(chan_to_spikes[c])) / T_total)
end
counts_bg = zeros(Int, N_channels)

# 4. Initialize Occurrences: Transfer Awake Discoveries + Prepare for Replay Detection
occurrences = Dict{Int, PureOccurrence}()
assign = [SpikeAssignment(0, 0) for _ in 1:N_spikes]

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

# 6. Allocate Thread & Birth Buffers
thread_bufs = allocate_thread_buffers(Threads.maxthreadid(); rng=rng)
birth_bufs = allocate_birth_buffers(chan_to_spikes)
println(@sprintf("Multithreading Initialized: %d threads active (max thread ID: %d)", Threads.nthreads(), Threads.maxthreadid()))

N_sweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 200
energy_history = Float64[]

println("\n=================================================================")
println(@sprintf("Starting %d Sweeps on FULL SESSION Recording (T = %.1fs)...", N_sweeps, T_total))
println("=================================================================")
print_diagnostics_header()

save_path = joinpath(@__DIR__, "checkpoint_pure_packet_full_session_$(N_sweeps)sweeps.jls")

function persist_checkpoint!(path)
    open(path, "w") do io
        serialize(io, (
            motifs = motifs, occurrences = occurrences, assign = assign,
            bg_rates = bg_rates, ard_state = ard_state, hyperstate = hyperstate,
            ou_ctx = ou_ctx, energy_history = energy_history
        ))
    end
end

for sweep in 1:N_sweeps
    t0_sw = time()

    # Step: Execute Complete MCMC Sweep
    n_birthed, n_overwritten, n_pruned = step_pure_packet_sweep!(
        sweep, motifs, occurrences, assign, spike_t, spike_neurons, chan_to_spikes,
        bg_rates, counts_bg, ard_state, ou_ctx, T_total, R_max,
        thread_bufs.tl_occ, thread_bufs.tl_pkt, thread_bufs.tl_logw, thread_bufs.tl_rng;
        bg_times=birth_bufs.bg_times, bg_indices=birth_bufs.bg_indices,
        gamma_m=100.0, alpha_c=10.0, beta_c=10.0, delta_margin=3.0,
        rng=rng
    )

    dt_sw = time() - t0_sw

    # Objective Telemetry & Diagnostics from metrics.jl
    diag = compute_sweep_diagnostics(
        sweep, motifs, occurrences, assign, bg_rates, ard_state, T_total,
        total_spikes_per_chan, dt_sw;
        n_birthed=n_birthed, n_pruned=n_pruned, ou_ctx=ou_ctx
    )
    push!(energy_history, diag.joint_energy)
    print_diagnostics_row(diag)

    # Persist every 10 sweeps
    if sweep % 10 == 0 || sweep == N_sweeps
        persist_checkpoint!(save_path)
    end
end

println("\n=================================================================")
println("Full Session Execution ($(N_sweeps) Sweeps) Completed!")
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
