# run_awake_scratch.jl
# High-Performance Runner for Pure Packet Marked Point Process Engine
# Scratch Start (0 Initial Occurrences) on Pure-Neural N=411 Units (Awake Task: T=2114.28s)

using Random, Printf, NPZ, Serialization, Statistics, LinearAlgebra

# Load legacy struct definitions for initial seed dictionary checkpoint
include("/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/regime_triad_sampler/run_regime_sampler.jl")
using PurePacketPPSeq

const BASE_CK_PATH = "/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/results/top4_task_20261008/mh_screening_experiment/regime_triad_reproduction_100s/checkpoint.jls"
ck = open(deserialize, BASE_CK_PATH)
s = ck.state
b = s.base

R_max = 4
N_channels = 411
neural_spike_indices = findall(c -> 1 <= c <= N_channels, b.neuron)
spike_t = b.t[neural_spike_indices]
spike_neuron = b.neuron[neural_spike_indices]
N_spikes = length(spike_t)
rng = MersenneTwister(42)

println(@sprintf("Domain Hygiene Enforced: Strictly N=%d pure neural units (MOp: 208, CP: 203)", N_channels))
println(@sprintf("Spike stream stripped of behavioral channels: %d / %d spikes retained (%.2f%%)",
        N_spikes, length(b.t), (N_spikes / length(b.t)) * 100.0))

# Pre-allocate zero-allocation background buffers & channel maps
counts_bg = zeros(Int, N_channels)
bg_rates = [b.bg_rate[b.channel_group[k]] * b.bg_pi[k] for k in 1:N_channels]
chan_to_spikes = [Int[] for _ in 1:N_channels]
for (idx, neuron_id) in enumerate(spike_neuron)
    push!(chan_to_spikes[neuron_id], idx)
end
total_spikes_per_chan = [length(chan_to_spikes[k]) for k in 1:N_channels]

# 1. Convert Motifs to PurePacketMotif (Strictly Pure Neural Units <= 411)
motifs = Dict{Int, PurePacketMotif}()
for (mid, m) in b.motifs
    active_chans = filter(k -> m.pi[k] >= 0.001, 1:N_channels)
    sort!(active_chans, by=k -> m.mu[k])
    orig_shift = !isempty(active_chans) ? m.mu[active_chans[1]] : 0.0
    pkts = Packet[
        Packet(p, k, m.mu[k] - orig_shift, m.var[k], max(m.pi[k] * (b.amp_a[b.channel_group[k]] / b.amp_b[b.channel_group[k]]), 0.01))
        for (p, k) in enumerate(active_chans)
    ]
    motifs[mid] = PurePacketMotif(mid, pkts, R_max, N_channels)
end

# Cold Scratch Initialization: 0 Occurrences, ALL Spikes in Background
occurrences = Dict{Int, PureOccurrence}()
assign = [SpikeAssignment(0, 0) for _ in 1:N_spikes]
println("Cold Scratch Initialized: 0 Occurrences, 0 Assigned Spikes, $(N_spikes) Background Spikes.")

# Setup Model States: ARD Manifold (alpha_prune = 6500 s^-2) & Continuous OU Triad
ard_state = ARDManifoldState(R_max; a0=1e-3, b0=1.0e-6, prune_threshold=6500.0)
hyperstate = Level5Hyperstate(1, R_max, max(maximum(keys(motifs)), 50); force_thresh=0.01)
ou_ctx = ContinuousOUContext(
    N_channels, b.T;
    M_motifs=max(maximum(keys(motifs)), 50),
    cp_range=99:301,
    dt_net=0.05, tau_net=0.5, sigma_net=0.4,
    dt_cell=10.0, tau_cell=60.0, sigma_cell=0.15,
    dt_motif=5.0, tau_motif=20.0, sigma_motif=0.3,
    alpha_c=10.0, nu_student=4.0
)

# Allocate Thread & Birth Buffers
thread_bufs = allocate_thread_buffers(Threads.maxthreadid(); rng=rng)
birth_bufs = allocate_birth_buffers(chan_to_spikes)
println(@sprintf("Multithreading Initialized: %d threads active (max thread ID: %d)", Threads.nthreads(), Threads.maxthreadid()))

# Stratified Audit Channels for Brown et al. (2002) Time-Rescaling Theorem
audit = select_stratified_audit_channels(motifs, bg_rates, N_channels; n_signal=10, n_quiet=10)
println("Stratified Audit Channels Initialized (Time-Rescaling Theorem):")
println("  Signal Channels (10 High-Density): ", audit.signal_channels)
println("  Quiet Channels  (10 Baseline):     ", audit.quiet_channels)

N_sweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 50
println("\nStarting $(N_sweeps) Sweeps from Cold Scratch Start...")
print_diagnostics_header()

save_path = joinpath(@__DIR__, "checkpoint_pure_packet_scratch_$(N_sweeps)sweeps.jls")
energy_history = Float64[]
ks_history = []

function persist_checkpoint!(path)
    open(path, "w") do io
        serialize(io, (
            motifs = motifs, occurrences = occurrences, assign = assign,
            bg_rates = bg_rates, ard_state = ard_state, hyperstate = hyperstate,
            ou_ctx = ou_ctx, energy_history = energy_history, ks_history = ks_history
        ))
    end
end

try
    for sweep in 1:N_sweeps
        t0_sw = time()

        # Execute complete MCMC sweep across all active universal m-mechanisms
        n_birthed, n_overwritten, n_pruned = step_pure_packet_sweep!(
            sweep, motifs, occurrences, assign, spike_t, spike_neuron, chan_to_spikes,
            bg_rates, counts_bg, ard_state, ou_ctx, b.T, R_max,
            thread_bufs.tl_occ, thread_bufs.tl_pkt, thread_bufs.tl_logw, thread_bufs.tl_rng;
            bg_times=birth_bufs.bg_times, bg_indices=birth_bufs.bg_indices,
            gamma_m=100.0, alpha_c=10.0, beta_c=10.0, delta_margin=3.0,
            rng=rng
        )

        dt_sw = time() - t0_sw

        # Compute and print objective sweep diagnostics from metrics.jl
        diag = compute_sweep_diagnostics(
            sweep, motifs, occurrences, assign, bg_rates, ard_state, b.T,
            total_spikes_per_chan, dt_sw;
            n_birthed=n_birthed, n_pruned=n_pruned, ou_ctx=ou_ctx
        )
        push!(energy_history, diag.joint_energy)
        print_diagnostics_row(diag)

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

            persist_checkpoint!(save_path)

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
