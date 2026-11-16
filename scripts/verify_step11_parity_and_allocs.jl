# verify_step11_parity_and_allocs.jl
# Validates Point 3: Evaluates allocations, wall-clock time, and birth/split parity under a real fit_sweep!

include("/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/regime_triad_sampler/run_regime_sampler.jl")
using PurePacketPPSeq
using Printf, Serialization

data = load_awake_spikes()
ck_path = joinpath(@__DIR__, "checkpoint_pure_packet_cold_start_100sweeps.jls")
model = resume_model_from_checkpoint(ck_path, data; R_max=4)

println("\n=================================================================")
println("PART A: ISOLATED STEP 11 TIMING & HEAP ALLOCATION AUDIT")
println("=================================================================")
@printf("%-8s | %-5s | %-5s | %-12s | %-14s | %-8s\n",
        "Motif", "Occs", "Pkts", "Time (s)", "Allocations", "Birthed")
println("-"^65)

total_step11_time = 0.0
total_step11_bytes = 0
total_step11_births = 0

for (mid, pm) in sort(collect(model.motifs), by=x->x[1])
    occ_list = [occ for occ in values(model.occurrences) if occ.motif == mid]
    isempty(occ_list) && continue
    P_before = length(pm.packets)
    
    # Warm-up / exact measurement with @timed
    stats = @timed PurePacketPPSeq.detect_and_birth_or_split_packets!(
        pm, occ_list, data.t, data.neuron, model.assign, model.bg_rates, data.N_channels;
        chan_spikes=data.chan_to_spikes, ou_ctx=model.ou_ctx, sigma_prior=0.015,
        alpha_a=1.0, beta_a=1.0,
        active_mask=model.ard_state.active_mask, ard_alphas=model.ard_state.alpha
    )
    
    P_after = length(pm.packets)
    n_birthed = P_after - P_before
    
    global total_step11_time += stats.time
    global total_step11_bytes += stats.bytes
    global total_step11_births += n_birthed
    
    @printf("M%-7d | %5d | %5d | %9.3f s  | %10.2f MB  | %+4d\n",
            mid, length(occ_list), P_after, stats.time, stats.bytes / (1024^2), n_birthed)
end
println("-"^65)
@printf("TOTAL STEP 11: %9.3f s | %10.2f MB | %+4d packets\n",
        total_step11_time, total_step11_bytes / (1024^2), total_step11_births)

println("\n=================================================================")
println("PART B: REAL ENGINE fit_sweep! (SWEEP 101) EXECUTION & PARITY")
println("=================================================================")

# Setup thread buffers for full sweep
max_threads = Threads.maxthreadid()
tl_occ = [Int[] for _ in 1:max_threads]
tl_pkt = [Int[] for _ in 1:max_threads]
tl_logw = [Float64[] for _ in 1:max_threads]
tl_rng = [Random.Xoshiro(42 + tid) for tid in 1:max_threads]
rng = Random.Xoshiro(101)

bg_times = [Float64[] for _ in 1:data.N_channels]
bg_indices = [Int[] for _ in 1:data.N_channels]
for ch in 1:data.N_channels
    sizehint!(bg_times[ch], 1024)
    sizehint!(bg_indices[ch], 1024)
end

println("Executing real full-engine fit_sweep! for Sweep 101...")
sweep_stats = @timed PurePacketPPSeq.step_pure_packet_sweep!(
    101, model.motifs, model.occurrences, model.assign,
    data.t, data.neuron, data.chan_to_spikes,
    model.bg_rates, model.counts_bg,
    model.ard_state, model.ou_ctx, data.T_total, model.R_max,
    tl_occ, tl_pkt, tl_logw, tl_rng;
    bg_times=bg_times, bg_indices=bg_indices,
    gamma_m=100.0, alpha_c=10.0, beta_c=10.0,
    alpha_a=1.0, beta_a=1.0, alpha_m=0.05, beta_m=1.0,
    frozen_motif_ids=Set{Int}(), allow_motif_birth=true,
    rng=rng
)

P_total = sum(length(m.packets) for m in values(model.motifs))
assigned_spikes = count(as -> as.event_id != 0, model.assign)
E_total = length(model.occurrences)
M_total = length(model.motifs)
mean_ce = mean(occ.amplitude for occ in values(model.occurrences))
mean_bg = mean(model.bg_rates)

# Diagnostic energy and gauge check
U_t = PurePacketPPSeq.compute_joint_energy(
    model.motifs, model.occurrences, data.total_spikes_per_chan,
    model.bg_rates, model.ard_state, data.T_total;
    ou_ctx=model.ou_ctx
)

max_gauge = 0.0
for pm in values(model.motifs)
    if !isempty(pm.packets)
        err = PurePacketPPSeq.compute_gauge_error(pm, model.ard_state.active_mask)
        global max_gauge = max(max_gauge, err)
    end
end

println("\n--- SWEEP 101 FULL ENGINE RESULTS ---")
@printf("  Wall-Clock Time:    %.3f s\n", sweep_stats.time)
@printf("  Heap Allocations:   %.2f MB\n", sweep_stats.bytes / (1024^2))
@printf("  Active Motifs M:    %d\n", M_total)
@printf("  Total Occurrences E: %d\n", E_total)
@printf("  Total Packets P:    %d\n", P_total)
@printf("  Assigned Spikes:    %d (%.2f%% foreground mass)\n", assigned_spikes, (assigned_spikes / length(data.t)) * 100)
@printf("  Mean Vigor E[c_e]:  %.4f\n", mean_ce)
@printf("  Mean Baseline rate: %.4f Hz\n", mean_bg)
@printf("  Joint Potential U_t: %.4e nats\n", U_t)
@printf("  Max Gauge Residual: %.2e\n", max_gauge)
