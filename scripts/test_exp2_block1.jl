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

# Block 1: [500s, 630s]
t_start = 500.0
t_end   = 630.0
t_dur   = t_end - t_start

idx_blk = findall(i -> (ch_all[i] <= 411) && (t_start <= ts_all[i] <= t_end), 1:length(ts_all))
ts_blk = ts_all[idx_blk] .- t_start
ch_blk = ch_all[idx_blk]

# MOp and CP spikes
mop_idx = findall(c -> c in mop_channels, ch_blk)
cp_idx  = findall(c -> c in cp_channels,  ch_blk)

ts_mop = ts_blk[mop_idx]
ch_mop = ch_blk[mop_idx]

ts_cp  = ts_blk[cp_idx]
ch_cp  = ch_blk[cp_idx]

println("Block 1 total spikes: MOp = $(length(ts_mop)), CP = $(length(ts_cp))")

# Evaluation window [3.5s, 126.5s]
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

# Load checkpoint
ckpt = deserialize("/home/foaleng/Documents/PurePacketPPSeq/scripts/checkpoint_pure_packet_cold_start_100sweeps.jls")
ou_full = ckpt[:ou_ctx]
bg_rates_full = ckpt[:bg_rates]

# -------------------------------------------------------------------------
# Helper function: Evaluate predictive log-likelihood
# -------------------------------------------------------------------------
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

    # Evaluate spike intensity sum: sum_{i} ln(1 + lam_motif(t_i) / lam0(t_i))
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

# -------------------------------------------------------------------------
# Part 2a: MOp -> CP Generative Prediction
# -------------------------------------------------------------------------
println("\n" * "="^75)
println("EXPERIMENT 2a: MOp -> CP Generative Prediction (Infer from MOp, Predict CP)")
println("="^75)

data_mop = SpikeDataset(ts_mop, ch_mop; N_channels=411, T_total=t_dur)
model_mop = initialize_scratch_model(
    data_mop, deepcopy(ckpt[:motifs]);
    bg_rates=ckpt[:bg_rates],
    ard_state=ckpt[:ard_state],
    R_max=length(ckpt[:ard_state].active_mask)
)

println("Running 15 inference sweeps on MOp spikes...")
fit!(
    model_mop, data_mop;
    sweeps=15,
    allow_motif_birth=false,
    frozen_motif_ids=Set(keys(model_mop.motifs))
)
println("Occurrences inferred from MOp: $(length(model_mop.occurrences))")

delta_ll_mop2cp, spk_gain_cp, comp_cp = evaluate_prediction(
    model_mop.occurrences,
    ckpt[:motifs],
    cp_channels,
    ts_cp_eval,
    ch_cp_eval,
    t_start,
    t_eval_start,
    t_eval_end,
    ou_full,
    bg_rates_full
)

gain_nat_cp = delta_ll_mop2cp / N_cp_eval
gain_bit_cp = gain_nat_cp / log(2.0)

@printf("  Held-out CP test spikes:        %d\n", N_cp_eval)
@printf("  Spike emission log-gain:        +%.2f nats\n", spk_gain_cp)
@printf("  Compensator penalty:            -%.2f nats\n", comp_cp)
@printf("  Net Predictive Log-Likelihood:  +%.2f nats\n", delta_ll_mop2cp)
@printf("  Predictive Gain per Spike:      +%.4f nats / spike\n", gain_nat_cp)
@printf("  Information Gain per Spike:     +%.4f bits / spike\n", gain_bit_cp)

# -------------------------------------------------------------------------
# Part 2b: CP -> MOp Generative Prediction
# -------------------------------------------------------------------------
println("\n" * "="^75)
println("EXPERIMENT 2b: CP -> MOp Generative Prediction (Infer from CP, Predict MOp)")
println("="^75)

data_cp = SpikeDataset(ts_cp, ch_cp; N_channels=411, T_total=t_dur)
model_cp = initialize_scratch_model(
    data_cp, deepcopy(ckpt[:motifs]);
    bg_rates=ckpt[:bg_rates],
    ard_state=ckpt[:ard_state],
    R_max=length(ckpt[:ard_state].active_mask)
)

println("Running 15 inference sweeps on CP spikes...")
fit!(
    model_cp, data_cp;
    sweeps=15,
    allow_motif_birth=false,
    frozen_motif_ids=Set(keys(model_cp.motifs))
)
println("Occurrences inferred from CP: $(length(model_cp.occurrences))")

delta_ll_cp2mop, spk_gain_mop, comp_mop = evaluate_prediction(
    model_cp.occurrences,
    ckpt[:motifs],
    mop_channels,
    ts_mop_eval,
    ch_mop_eval,
    t_start,
    t_eval_start,
    t_eval_end,
    ou_full,
    bg_rates_full
)

gain_nat_mop = delta_ll_cp2mop / N_mop_eval
gain_bit_mop = gain_nat_mop / log(2.0)

@printf("  Held-out MOp test spikes:       %d\n", N_mop_eval)
@printf("  Spike emission log-gain:        +%.2f nats\n", spk_gain_mop)
@printf("  Compensator penalty:            -%.2f nats\n", comp_mop)
@printf("  Net Predictive Log-Likelihood:  +%.2f nats\n", delta_ll_cp2mop)
@printf("  Predictive Gain per Spike:      +%.4f nats / spike\n", gain_nat_mop)
@printf("  Information Gain per Spike:     +%.4f bits / spike\n", gain_bit_mop)

# -------------------------------------------------------------------------
# Part 2c: Asymmetric Directionality Index
# -------------------------------------------------------------------------
println("\n" * "="^75)
println("EXPERIMENT 2c: Directional Asymmetry Analysis")
println("="^75)
asym_ratio = gain_nat_cp / max(gain_nat_mop, 1e-12)
diff_gain  = gain_nat_cp - gain_nat_mop
@printf("  MOp -> CP Gain:  +%.4f nats / spike (+%.4f bits)\n", gain_nat_cp, gain_bit_cp)
@printf("  CP -> MOp Gain:  +%.4f nats / spike (+%.4f bits)\n", gain_nat_mop, gain_bit_mop)
@printf("  Net Asymmetry (MOp -> CP minus CP -> MOp): +%.4f nats / spike\n", diff_gain)
@printf("  Relative Asymmetry Factor: %.2fx stronger predictive coupling\n", asym_ratio)
println("="^75)
