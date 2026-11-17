using PurePacketPPSeq
using Serialization
using Statistics
using NPZ
using DelimitedFiles
using LinearAlgebra
using SpecialFunctions
using Printf

println("="^80)
println("EXPERIMENT 1: JOINT VS. INDEPENDENT PPSEQ ON HELD-OUT DATA")
println("="^80)

# 1. Load data
data_dir = "/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/results/top4_task_20261008/data/lgcy0288__20230519_114115__ephys"
units_raw, _ = readdlm(joinpath(data_dir, "units.csv"), Char(44), header=true)
raw_regions = String.(units_raw[:, 3])
mop_channels = Set(findall(r -> startswith(r, "MOp"), raw_regions))
cp_channels  = Set(findall(r -> !startswith(r, "MOp"), raw_regions))

raw_npz = npzread(joinpath(data_dir, "joint.npz"))
ts_all = Float64.(raw_npz["timestamp"])
ch_all = Int.(raw_npz["channel"])

# Checkpoint for OU background dynamics
ckpt = deserialize("/home/foaleng/Documents/PurePacketPPSeq/scripts/checkpoint_pure_packet_cold_start_100sweeps.jls")
ou_full = ckpt[:ou_ctx]
bg_rates_full = ckpt[:bg_rates]

# Forward temporal split
t_tr_start = 400.0
t_tr_end   = 1200.0
t_tr_dur   = t_tr_end - t_tr_start

t_te_start = 1200.0
t_te_end   = 1700.0
t_te_dur   = t_te_end - t_te_start

# Evaluation inner window (margins of 3.5s)
t_eval_start = 3.5
t_eval_end   = t_te_dur - 3.5

# Filter Training spikes
idx_tr = findall(i -> (ch_all[i] <= 411) && (t_tr_start <= ts_all[i] <= t_tr_end), 1:length(ts_all))
ts_tr = ts_all[idx_tr] .- t_tr_start
ch_tr = ch_all[idx_tr]

# Filter Test spikes
idx_te = findall(i -> (ch_all[i] <= 411) && (t_te_start <= ts_all[i] <= t_te_end), 1:length(ts_all))
ts_te = ts_all[idx_te] .- t_te_start
ch_te = ch_all[idx_te]

# Channel masks
tr_mop_idx = findall(c -> c in mop_channels, ch_tr)
tr_cp_idx  = findall(c -> c in cp_channels,  ch_tr)

te_mop_idx = findall(c -> c in mop_channels, ch_te)
te_cp_idx  = findall(c -> c in cp_channels,  ch_te)

println("Training Window:  t = [$t_tr_start, $t_tr_end s] ($t_tr_dur s)")
println("  MOp Spikes: $(length(tr_mop_idx)) | CP Spikes: $(length(tr_cp_idx)) | Joint: $(length(ch_tr))")
println("Test Window:      t = [$t_te_start, $t_te_end s] ($t_te_dur s)")
println("  MOp Spikes: $(length(te_mop_idx)) | CP Spikes: $(length(te_cp_idx)) | Joint: $(length(ch_te))")

# Helper: evaluate point-process log-likelihood gain of inferred occurrences
function evaluate_model_ll(
    inferred_occurrences,
    motifs,
    target_channels::Set{Int},
    ts_eval::Vector{Float64},
    ch_eval::Vector{Int},
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
        
        motif = motifs[m]
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
                
                z2 = (t_eval_end - mu_tilde) / (sqrt(2.0) * sigma_p)
                z1 = (t_eval_start - mu_tilde) / (sqrt(2.0) * sigma_p)
                p_mass_in_window = 0.5 * (erf(z2) - erf(z1))
                compensator_penalty += c_e * a_p * p_mass_in_window

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
    for (spk_idx, t_spk) in enumerate(ts_eval)
        c_spk = ch_eval[spk_idx]
        if !(c_spk in target_channels)
            continue
        end
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

# Filter evaluation test spikes
te_eval_mask = findall(t -> t_eval_start <= t <= t_eval_end, ts_te)
ts_te_eval = ts_te[te_eval_mask]
ch_te_eval = ch_te[te_eval_mask]

N_eval_total = length(ts_te_eval)
N_eval_mop   = count(c -> c in mop_channels, ch_te_eval)
N_eval_cp    = count(c -> c in cp_channels,  ch_te_eval)

println("\nHeld-out Test Spikes in Evaluation Window [$(t_eval_start)s, $(t_eval_end)s]:")
println("  Total: $N_eval_total | MOp: $N_eval_mop | CP: $N_eval_cp")

# =========================================================================
# 1. TRAIN MODEL A: MOp NEURONS ONLY
# =========================================================================
println("\n" * "="^75)
println("STEP 1: Training Model A (MOp Neurons Only, N=208)...")
println("="^75)

data_tr_mop = SpikeDataset(ts_tr[tr_mop_idx], ch_tr[tr_mop_idx]; N_channels=411, T_total=t_tr_dur)
model_a = initialize_scratch_model(data_tr_mop, Dict{Int, PurePacketMotif}(); R_init=1)
fit!(model_a, data_tr_mop; sweeps=25, allow_motif_birth=true)
println("Model A Trained: $(length(model_a.motifs)) motifs, $(sum(length(m.packets) for m in values(model_a.motifs))) packets.")

# Test Model A on held-out MOp test spikes
data_te_mop = SpikeDataset(ts_te[te_mop_idx], ch_te[te_mop_idx]; N_channels=411, T_total=t_te_dur)
model_a_test = initialize_scratch_model(
    data_te_mop, deepcopy(model_a.motifs);
    bg_rates=model_a.bg_rates,
    ard_state=model_a.ard_state,
    R_max=length(model_a.ard_state.active_mask)
)
fit!(model_a_test, data_te_mop; sweeps=15, allow_motif_birth=false, frozen_motif_ids=Set(keys(model_a_test.motifs)))
dll_mop, spk_mop, comp_mop = evaluate_model_ll(
    model_a_test.occurrences, model_a.motifs, mop_channels,
    ts_te_eval, ch_te_eval, t_te_start, t_eval_start, t_eval_end, ou_full, bg_rates_full
)
println("Model A Held-out Test Delta LL (MOp): +$(round(dll_mop, digits=2)) nats (+$(round(dll_mop / N_eval_mop, digits=4)) nats/spk)")

# =========================================================================
# 2. TRAIN MODEL B: CP NEURONS ONLY
# =========================================================================
println("\n" * "="^75)
println("STEP 2: Training Model B (CP Neurons Only, N=203)...")
println("="^75)

data_tr_cp = SpikeDataset(ts_tr[tr_cp_idx], ch_tr[tr_cp_idx]; N_channels=411, T_total=t_tr_dur)
model_b = initialize_scratch_model(data_tr_cp, Dict{Int, PurePacketMotif}(); R_init=1)
fit!(model_b, data_tr_cp; sweeps=25, allow_motif_birth=true)
println("Model B Trained: $(length(model_b.motifs)) motifs, $(sum(length(m.packets) for m in values(model_b.motifs))) packets.")

# Test Model B on held-out CP test spikes
data_te_cp = SpikeDataset(ts_te[te_cp_idx], ch_te[te_cp_idx]; N_channels=411, T_total=t_te_dur)
model_b_test = initialize_scratch_model(
    data_te_cp, deepcopy(model_b.motifs);
    bg_rates=model_b.bg_rates,
    ard_state=model_b.ard_state,
    R_max=length(model_b.ard_state.active_mask)
)
fit!(model_b_test, data_te_cp; sweeps=15, allow_motif_birth=false, frozen_motif_ids=Set(keys(model_b_test.motifs)))
dll_cp, spk_cp, comp_cp = evaluate_model_ll(
    model_b_test.occurrences, model_b.motifs, cp_channels,
    ts_te_eval, ch_te_eval, t_te_start, t_eval_start, t_eval_end, ou_full, bg_rates_full
)
println("Model B Held-out Test Delta LL (CP): +$(round(dll_cp, digits=2)) nats (+$(round(dll_cp / N_eval_cp, digits=4)) nats/spk)")

# Sum of independent models
dll_indep = dll_mop + dll_cp
println("\nTotal Independent Held-out Delta LL (MOp + CP): +$(round(dll_indep, digits=2)) nats (+$(round(dll_indep / N_eval_total, digits=4)) nats/spk)")

# =========================================================================
# 3. TRAIN MODEL C: JOINT MOp + CP NEURONS
# =========================================================================
println("\n" * "="^75)
println("STEP 3: Training Model C (Joint MOp + CP Neurons, N=411)...")
println("="^75)

data_tr_joint = SpikeDataset(ts_tr, ch_tr; N_channels=411, T_total=t_tr_dur)
model_c = initialize_scratch_model(data_tr_joint, Dict{Int, PurePacketMotif}(); R_init=1)
fit!(model_c, data_tr_joint; sweeps=25, allow_motif_birth=true)
println("Model C Trained: $(length(model_c.motifs)) motifs, $(sum(length(m.packets) for m in values(model_c.motifs))) packets.")

# Test Model C on held-out joint test spikes
data_te_joint = SpikeDataset(ts_te, ch_te; N_channels=411, T_total=t_te_dur)
model_c_test = initialize_scratch_model(
    data_te_joint, deepcopy(model_c.motifs);
    bg_rates=model_c.bg_rates,
    ard_state=model_c.ard_state,
    R_max=length(model_c.ard_state.active_mask)
)
fit!(model_c_test, data_te_joint; sweeps=15, allow_motif_birth=false, frozen_motif_ids=Set(keys(model_c_test.motifs)))
all_channels = union(mop_channels, cp_channels)
dll_joint, spk_joint, comp_joint = evaluate_model_ll(
    model_c_test.occurrences, model_c.motifs, all_channels,
    ts_te_eval, ch_te_eval, t_te_start, t_eval_start, t_eval_end, ou_full, bg_rates_full
)
println("Model C Held-out Test Delta LL (Joint): +$(round(dll_joint, digits=2)) nats (+$(round(dll_joint / N_eval_total, digits=4)) nats/spk)")

# =========================================================================
# 4. FINAL COMPARISON & INFORMATION GAIN
# =========================================================================
delta_ll_1 = dll_joint - dll_indep
gain_per_spike_nats = delta_ll_1 / N_eval_total
gain_per_spike_bits = gain_per_spike_nats / log(2.0)

println("\n" * "="^85)
println("EXPERIMENT 1 FINAL RESULTS: JOINT VS. INDEPENDENT PPSEQ")
println("="^85)
@printf("  Held-out Test Period:            t in [%.1f, %.1f s] (Duration: %.1f s)\n", t_te_start, t_te_end, t_te_dur)
@printf("  Total Held-out Spikes Evaluated: %d (MOp: %d, CP: %d)\n", N_eval_total, N_eval_mop, N_eval_cp)
@printf("  Model A (MOp Independent):       +%.2f nats (+%.4f nats/spk)\n", dll_mop, dll_mop / N_eval_mop)
@printf("  Model B (CP Independent):        +%.2f nats (+%.4f nats/spk)\n", dll_cp, dll_cp / N_eval_cp)
@printf("  Sum of Independent Models:       +%.2f nats (+%.4f nats/spk)\n", dll_indep, dll_indep / N_eval_total)
@printf("  Model C (Joint MOp + CP):        +%.2f nats (+%.4f nats/spk)\n", dll_joint, dll_joint / N_eval_total)
@printf("  -------------------------------------------------------------------\n")
@printf("  Delta LL_1 = LL_joint - (LL_MOp + LL_CP): +%.2f nats\n", delta_ll_1)
@printf("  Predictive Improvement per Spike:         +%.4f nats / spike\n", gain_per_spike_nats)
@printf("  Predictive Information Gain per Spike:    +%.4f bits / spike\n", gain_per_spike_bits)
println("="^85)

exp1_summary = Dict(
    :N_eval_total => N_eval_total,
    :N_eval_mop   => N_eval_mop,
    :N_eval_cp    => N_eval_cp,
    :dll_mop      => dll_mop,
    :dll_cp       => dll_cp,
    :dll_indep    => dll_indep,
    :dll_joint    => dll_joint,
    :delta_ll_1   => delta_ll_1,
    :gain_nats    => gain_per_spike_nats,
    :gain_bits    => gain_per_spike_bits
)

serialize("/home/foaleng/Documents/PurePacketPPSeq/scripts/exp1_joint_vs_indep_results.jls", exp1_summary)
println("Saved Experiment 1 summary to /home/foaleng/Documents/PurePacketPPSeq/scripts/exp1_joint_vs_indep_results.jls")
