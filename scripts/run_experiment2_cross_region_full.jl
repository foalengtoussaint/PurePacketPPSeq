using PurePacketPPSeq
using Serialization
using Statistics
using NPZ
using DelimitedFiles
using LinearAlgebra
using SpecialFunctions
using Printf

# Load channel regions
data_dir = "/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/results/top4_task_20261008/data/lgcy0288__20230519_114115__ephys"
units_raw, _ = readdlm(joinpath(data_dir, "units.csv"), Char(44), header=true)
raw_regions = String.(units_raw[:, 3])
mop_channels = Set(findall(r -> startswith(r, "MOp"), raw_regions))
cp_channels  = Set(findall(r -> !startswith(r, "MOp"), raw_regions))

println("Channel counts: MOp = $(length(mop_channels)), CP = $(length(cp_channels))")

# Load spikes
raw_npz = npzread(joinpath(data_dir, "joint.npz"))
ts_all = Float64.(raw_npz["timestamp"])
ch_all = Int.(raw_npz["channel"])

# Checkpoint
ckpt = deserialize("/home/foaleng/Documents/PurePacketPPSeq/scripts/checkpoint_pure_packet_cold_start_100sweeps.jls")
ou_full = ckpt[:ou_ctx]
bg_rates_full = ckpt[:bg_rates]

function evaluate_prediction(
    inferred_occurrences,
    frozen_motifs,
    target_channels::Set{Int},
    ts_target_eval::Vector{Float64},
    ch_target_eval::Vector{Int},
    t_start_abs::Float64,
    t_eval_start::Float64,
    t_eval_end::Float64,
    ou_full,
    bg_rates_full
)
    active_instances = Dict{Int, Vector{Tuple{Float64, Float64, Float64}}}()
    compensator_penalty = 0.0

    for occ in values(inferred_occurrences)
        m = occ.motif
        tau_e = occ.tau
        c_e   = occ.amplitude
        u_e   = occ.u
        
        motif = frozen_motifs[m]
        R_common = min(size(motif.W, 1), length(u_e))
        
        for (p_idx, p) in enumerate(motif.packets)
            chan = p.channel
            if chan in target_channels
                a_p   = p.mass
                mu_p  = p.mu
                var_p = p.var
                sigma_p = sqrt(max(var_p, 1e-6))
                
                w_shift = (R_common > 0) ? dot(view(motif.W, 1:R_common, p_idx), view(u_e, 1:R_common)) : 0.0
                mu_tilde = tau_e + mu_p + w_shift
                
                # Compensator integral across [t_eval_start, t_eval_end]
                z2 = (t_eval_end - mu_tilde) / (sqrt(2.0) * sigma_p)
                z1 = (t_eval_start - mu_tilde) / (sqrt(2.0) * sigma_p)
                p_mass_in_window = 0.5 * (erf(z2) - erf(z1))
                compensator_penalty += c_e * a_p * p_mass_in_window

                # Packet instance for spike intensity evaluation
                if mu_tilde + 4.0 * sigma_p >= t_eval_start && mu_tilde - 4.0 * sigma_p <= t_eval_end
                    if !haskey(active_instances, chan)
                        active_instances[chan] = Tuple{Float64, Float64, Float64}[]
                    end
                    push!(active_instances[chan], (c_e * a_p, mu_tilde, var_p))
                end
            end
        end
    end

    spike_gain = 0.0
    for (spk_idx, t_spk) in enumerate(ts_target_eval)
        c_spk = ch_target_eval[spk_idx]
        t_abs = t_start_abs + t_spk
        lam0 = max(exp(PurePacketPPSeq.evaluate_background_log_intensity(ou_full, bg_rates_full, c_spk, t_abs)), 1e-12)
        
        lam_motif = 0.0
        if haskey(active_instances, c_spk)
            for (amp, mu_t, var_p) in active_instances[c_spk]
                dt = t_spk - mu_t
                if abs(dt) < 4.0 * sqrt(var_p)
                    lam_motif += amp * (1.0 / sqrt(2.0 * pi * var_p)) * exp(-dt^2 / (2.0 * var_p))
                end
            end
        end
        
        spike_gain += log(1.0 + lam_motif / lam0)
    end

    delta_ll = spike_gain - compensator_penalty
    return delta_ll, spike_gain, compensator_penalty
end

# 4 Stratified 130-second blocks
blocks = [
    (500.0, 630.0,  "Block 1 (Early Operant)"),
    (900.0, 1030.0, "Block 2 (Mid Operant)"),
    (1300.0, 1430.0,"Block 3 (Late Operant)"),
    (1700.0, 1830.0,"Block 4 (Terminal Operant)")
]

results = []

for (blk_idx, (t_start, t_end, blk_name)) in enumerate(blocks)
    t_dur = t_end - t_start
    idx_blk = findall(i -> (ch_all[i] <= 411) && (t_start <= ts_all[i] <= t_end), 1:length(ts_all))
    ts_blk = ts_all[idx_blk] .- t_start
    ch_blk = ch_all[idx_blk]

    mop_idx = findall(c -> c in mop_channels, ch_blk)
    cp_idx  = findall(c -> c in cp_channels,  ch_blk)

    ts_mop = ts_blk[mop_idx]
    ch_mop = ch_blk[mop_idx]

    ts_cp  = ts_blk[cp_idx]
    ch_cp  = ch_blk[cp_idx]

    t_eval_start = 3.5
    t_eval_end   = t_dur - 3.5

    cp_eval_idx = findall(t -> t_eval_start <= t <= t_eval_end, ts_cp)
    ts_cp_eval  = ts_cp[cp_eval_idx]
    ch_cp_eval  = ch_cp[cp_eval_idx]
    N_cp_eval   = length(ts_cp_eval)

    mop_eval_idx = findall(t -> t_eval_start <= t <= t_eval_end, ts_mop)
    ts_mop_eval  = ts_mop[mop_eval_idx]
    ch_mop_eval  = ch_mop[mop_eval_idx]
    N_mop_eval   = length(ts_mop_eval)

    println("\n" * "="^80)
    println("EVALUATING $(blk_name): t = [$t_start, $t_end s] (Eval window: 123.0 s)")
    println("="^80)

    # 1. MOp -> CP
    data_mop = SpikeDataset(ts_mop, ch_mop; N_channels=411, T_total=t_dur)
    model_mop = initialize_scratch_model(
        data_mop, deepcopy(ckpt[:motifs]);
        bg_rates=ckpt[:bg_rates],
        ard_state=ckpt[:ard_state],
        R_max=length(ckpt[:ard_state].active_mask)
    )
    fit!(model_mop, data_mop; sweeps=15, allow_motif_birth=false, frozen_motif_ids=Set(keys(model_mop.motifs)))
    dll_m2c, spk_m2c, comp_m2c = evaluate_prediction(
        model_mop.occurrences, ckpt[:motifs], cp_channels, ts_cp_eval, ch_cp_eval,
        t_start, t_eval_start, t_eval_end, ou_full, bg_rates_full
    )
    gain_nat_m2c = dll_m2c / N_cp_eval
    gain_bit_m2c = gain_nat_m2c / log(2.0)

    # 2. CP -> MOp
    data_cp = SpikeDataset(ts_cp, ch_cp; N_channels=411, T_total=t_dur)
    model_cp = initialize_scratch_model(
        data_cp, deepcopy(ckpt[:motifs]);
        bg_rates=ckpt[:bg_rates],
        ard_state=ckpt[:ard_state],
        R_max=length(ckpt[:ard_state].active_mask)
    )
    fit!(model_cp, data_cp; sweeps=15, allow_motif_birth=false, frozen_motif_ids=Set(keys(model_cp.motifs)))
    dll_c2m, spk_c2m, comp_c2m = evaluate_prediction(
        model_cp.occurrences, ckpt[:motifs], mop_channels, ts_mop_eval, ch_mop_eval,
        t_start, t_eval_start, t_eval_end, ou_full, bg_rates_full
    )
    gain_nat_c2m = dll_c2m / N_mop_eval
    gain_bit_c2m = gain_nat_c2m / log(2.0)

    asym_diff = gain_nat_m2c - gain_nat_c2m
    asym_ratio = gain_nat_m2c / max(gain_nat_c2m, 1e-12)

    @printf("  MOp -> CP: +%.2f nats across %d CP spikes (+%.4f nats/spk, +%.4f bits/spk)\n",
            dll_m2c, N_cp_eval, gain_nat_m2c, gain_bit_m2c)
    @printf("  CP -> MOp: +%.2f nats across %d MOp spikes (+%.4f nats/spk, +%.4f bits/spk)\n",
            dll_c2m, N_mop_eval, gain_nat_c2m, gain_bit_c2m)
    @printf("  Directional Delta: +%.4f nats/spk (MOp drive is %.2fx stronger)\n", asym_diff, asym_ratio)

    push!(results, (
        name=blk_name,
        N_cp=N_cp_eval,
        dll_m2c=dll_m2c,
        nat_m2c=gain_nat_m2c,
        bit_m2c=gain_bit_m2c,
        N_mop=N_mop_eval,
        dll_c2m=dll_c2m,
        nat_c2m=gain_nat_c2m,
        bit_c2m=gain_bit_c2m,
        diff=asym_diff,
        ratio=asym_ratio
    ))
end

# Aggregate summary
println("\n" * "="^95)
println("EXPERIMENT 2 COMPLETE SUMMARY ACROSS 4 STRATIFIED TEST BLOCKS")
println("="^95)
println("Block                     | N_CP   | MOp->CP (nats) | CP->MOp (nats) | Delta (nats) | Asym Ratio")
println("-"^95)
for r in results
    @printf("%-25s | %6d | %+14.4f | %+14.4f | %+12.4f | %9.2fx\n",
            r.name, r.N_cp, r.nat_m2c, r.nat_c2m, r.diff, r.ratio)
end
println("-"^95)

total_N_cp = sum(r.N_cp for r in results)
total_N_mop = sum(r.N_mop for r in results)
total_dll_m2c = sum(r.dll_m2c for r in results)
total_dll_c2m = sum(r.dll_c2m for r in results)

pooled_nat_m2c = total_dll_m2c / total_N_cp
pooled_bit_m2c = pooled_nat_m2c / log(2.0)

pooled_nat_c2m = total_dll_c2m / total_N_mop
pooled_bit_c2m = pooled_nat_c2m / log(2.0)

mean_nat_m2c = mean([r.nat_m2c for r in results])
sem_nat_m2c  = std([r.nat_m2c for r in results]) / sqrt(length(results))

mean_nat_c2m = mean([r.nat_c2m for r in results])
sem_nat_c2m  = std([r.nat_c2m for r in results]) / sqrt(length(results))

pooled_diff = pooled_nat_m2c - pooled_nat_c2m
pooled_ratio = pooled_nat_m2c / pooled_nat_c2m

@printf("POOLED TOTAL (%d s):   | %6d | %+14.4f | %+14.4f | %+12.4f | %9.2fx\n",
        Int(4 * 123), total_N_cp, pooled_nat_m2c, pooled_nat_c2m, pooled_diff, pooled_ratio)
@printf("MEAN ± SEM (N=4 blks):    |        | %+.4f±%.4f | %+.4f±%.4f | %+.4f      | %9.2fx\n",
        mean_nat_m2c, sem_nat_m2c, mean_nat_c2m, sem_nat_c2m, mean_nat_m2c - mean_nat_c2m, mean_nat_m2c / mean_nat_c2m)
println("="^95)

serialize("/home/foaleng/Documents/PurePacketPPSeq/scripts/exp2_cross_region_results.jls", results)
println("Results serialized to /home/foaleng/Documents/PurePacketPPSeq/scripts/exp2_cross_region_results.jls")
