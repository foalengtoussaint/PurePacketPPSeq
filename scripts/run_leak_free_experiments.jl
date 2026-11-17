using PurePacketPPSeq
using Serialization
using Statistics
using NPZ
using DelimitedFiles
using LinearAlgebra
using SpecialFunctions
using Printf

println("="^85)
println("STRICT LEAK-FREE RE-EVALUATION: EXPERIMENT 1 & EXPERIMENT 2")
println("="^85)

# Load units and channel regions
data_dir = "/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/results/top4_task_20261008/data/lgcy0288__20230519_114115__ephys"
units_raw, _ = readdlm(joinpath(data_dir, "units.csv"), Char(44), header=true)
raw_regions = String.(units_raw[:, 3])
mop_channels = Set(findall(r -> startswith(r, "MOp"), raw_regions))
cp_channels  = Set(findall(r -> !startswith(r, "MOp"), raw_regions))
all_channels = union(mop_channels, cp_channels)

raw_npz = npzread(joinpath(data_dir, "joint.npz"))
ts_all = Float64.(raw_npz["timestamp"])
ch_all = Int.(raw_npz["channel"])

# Checkpoint for OU background dynamics
ckpt_session = deserialize("/home/foaleng/Documents/PurePacketPPSeq/scripts/checkpoint_pure_packet_cold_start_100sweeps.jls")
ou_full = ckpt_session[:ou_ctx]
bg_rates_full = ckpt_session[:bg_rates]

# Forward temporal split
t_tr_start = 400.0
t_tr_end   = 1200.0
t_tr_dur   = t_tr_end - t_tr_start

t_te_start = 1200.0
t_te_end   = 1700.0
t_te_dur   = t_te_end - t_te_start

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

tr_mop_idx = findall(c -> c in mop_channels, ch_tr)
tr_cp_idx  = findall(c -> c in cp_channels,  ch_tr)

te_mop_idx = findall(c -> c in mop_channels, ch_te)
te_cp_idx  = findall(c -> c in cp_channels,  ch_te)

# Test spikes in inner evaluation window
te_eval_mask = findall(t -> t_eval_start <= t <= t_eval_end, ts_te)
ts_te_eval = ts_te[te_eval_mask]
ch_te_eval = ch_te[te_eval_mask]

N_eval_total = length(ts_te_eval)
N_eval_mop   = count(c -> c in mop_channels, ch_te_eval)
N_eval_cp    = count(c -> c in cp_channels,  ch_te_eval)

println("Training: t in [$t_tr_start, $t_tr_end s] ($t_tr_dur s) | MOp: $(length(tr_mop_idx)), CP: $(length(tr_cp_idx)), Joint: $(length(ch_tr))")
println("Testing:  t in [$t_te_start, $t_te_end s] ($t_te_dur s) | MOp: $N_eval_mop, CP: $N_eval_cp, Total: $N_eval_total")

# Helper function to evaluate point-process intensity & compensator
function evaluate_channel_ll(
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

# =========================================================================
# STEP 1: TRAIN 50-SWEEP CONVERGED LEAK-FREE MODELS
# =========================================================================
N_SWEEPS = 50

# 1A. Model A (MOp only)
println("\n" * "="^75)
println("Training Model A: MOp-only ($N_SWEEPS sweeps, cold start)...")
println("="^75)
data_tr_mop = SpikeDataset(ts_tr[tr_mop_idx], ch_tr[tr_mop_idx]; N_channels=411, T_total=t_tr_dur)
model_a = initialize_scratch_model(data_tr_mop, Dict{Int, PurePacketMotif}(); R_init=1)
fit!(model_a, data_tr_mop; sweeps=N_SWEEPS, allow_motif_birth=true)
M_a = length(model_a.motifs)
P_a = sum(length(m.packets) for m in values(model_a.motifs))
println("Model A Trained: $M_a motifs, $P_a packets.")

# 1B. Model B (CP only)
println("\n" * "="^75)
println("Training Model B: CP-only ($N_SWEEPS sweeps, cold start)...")
println("="^75)
data_tr_cp = SpikeDataset(ts_tr[tr_cp_idx], ch_tr[tr_cp_idx]; N_channels=411, T_total=t_tr_dur)
model_b = initialize_scratch_model(data_tr_cp, Dict{Int, PurePacketMotif}(); R_init=1)
fit!(model_b, data_tr_cp; sweeps=N_SWEEPS, allow_motif_birth=true)
M_b = length(model_b.motifs)
P_b = sum(length(m.packets) for m in values(model_b.motifs))
println("Model B Trained: $M_b motifs, $P_b packets.")

# 1C. Model C (Joint MOp + CP)
println("\n" * "="^75)
println("Training Model C: Joint MOp + CP ($N_SWEEPS sweeps, cold start)...")
println("="^75)
data_tr_joint = SpikeDataset(ts_tr, ch_tr; N_channels=411, T_total=t_tr_dur)
model_c = initialize_scratch_model(data_tr_joint, Dict{Int, PurePacketMotif}(); R_init=1)
fit!(model_c, data_tr_joint; sweeps=N_SWEEPS, allow_motif_birth=true)
M_c = length(model_c.motifs)
P_c = sum(length(m.packets) for m in values(model_c.motifs))
println("Model C Trained: $M_c motifs, $P_c packets.")

# =========================================================================
# STEP 2: EXPERIMENT 1 HELD-OUT EVALUATION & PER-REGION DECOMPOSITION
# =========================================================================
println("\n" * "="^85)
println("EVALUATING EXPERIMENT 1 ON HELD-OUT TEST BLOCK (1200 - 1700 s)")
println("="^85)

# Test Model A on MOp spikes
data_te_mop = SpikeDataset(ts_te[te_mop_idx], ch_te[te_mop_idx]; N_channels=411, T_total=t_te_dur)
model_a_test = initialize_scratch_model(
    data_te_mop, deepcopy(model_a.motifs);
    bg_rates=model_a.bg_rates, ard_state=model_a.ard_state, R_max=length(model_a.ard_state.active_mask)
)
fit!(model_a_test, data_te_mop; sweeps=15, allow_motif_birth=false, frozen_motif_ids=Set(keys(model_a_test.motifs)))
dll_model_a, _, _ = evaluate_channel_ll(
    model_a_test.occurrences, model_a.motifs, mop_channels,
    ts_te_eval, ch_te_eval, t_te_start, t_eval_start, t_eval_end, ou_full, bg_rates_full
)

# Test Model B on CP spikes
data_te_cp = SpikeDataset(ts_te[te_cp_idx], ch_te[te_cp_idx]; N_channels=411, T_total=t_te_dur)
model_b_test = initialize_scratch_model(
    data_te_cp, deepcopy(model_b.motifs);
    bg_rates=model_b.bg_rates, ard_state=model_b.ard_state, R_max=length(model_b.ard_state.active_mask)
)
fit!(model_b_test, data_te_cp; sweeps=15, allow_motif_birth=false, frozen_motif_ids=Set(keys(model_b_test.motifs)))
dll_model_b, _, _ = evaluate_channel_ll(
    model_b_test.occurrences, model_b.motifs, cp_channels,
    ts_te_eval, ch_te_eval, t_te_start, t_eval_start, t_eval_end, ou_full, bg_rates_full
)

dll_indep_total = dll_model_a + dll_model_b

# Test Model C on joint spikes
data_te_joint = SpikeDataset(ts_te, ch_te; N_channels=411, T_total=t_te_dur)
model_c_test = initialize_scratch_model(
    data_te_joint, deepcopy(model_c.motifs);
    bg_rates=model_c.bg_rates, ard_state=model_c.ard_state, R_max=length(model_c.ard_state.active_mask)
)
fit!(model_c_test, data_te_joint; sweeps=15, allow_motif_birth=false, frozen_motif_ids=Set(keys(model_c_test.motifs)))

# Decompose Model C into MOp and CP components
dll_joint_mop, _, _ = evaluate_channel_ll(
    model_c_test.occurrences, model_c.motifs, mop_channels,
    ts_te_eval, ch_te_eval, t_te_start, t_eval_start, t_eval_end, ou_full, bg_rates_full
)
dll_joint_cp, _, _ = evaluate_channel_ll(
    model_c_test.occurrences, model_c.motifs, cp_channels,
    ts_te_eval, ch_te_eval, t_te_start, t_eval_start, t_eval_end, ou_full, bg_rates_full
)
dll_joint_total = dll_joint_mop + dll_joint_cp

delta_ll_1_total = dll_joint_total - dll_indep_total
delta_ll_1_mop   = dll_joint_mop - dll_model_a
delta_ll_1_cp    = dll_joint_cp - dll_model_b

println("EXPERIMENT 1 DECOMPOSITION:")
@printf("  Model A (MOp Indep, M=%d):         +%.2f nats (+%.4f nats/spk)\n", M_a, dll_model_a, dll_model_a / N_eval_mop)
@printf("  Model C (Joint MOp component):     +%.2f nats (+%.4f nats/spk)\n", dll_joint_mop, dll_joint_mop / N_eval_mop)
@printf("  --> MOp Advantage from Joint:      %+.2f nats (%+.4f nats/spk)\n\n", delta_ll_1_mop, delta_ll_1_mop / N_eval_mop)

@printf("  Model B (CP Indep, M=%d):          +%.2f nats (+%.4f nats/spk)\n", M_b, dll_model_b, dll_model_b / N_eval_cp)
@printf("  Model C (Joint CP component):      +%.2f nats (+%.4f nats/spk)\n", dll_joint_cp, dll_joint_cp / N_eval_cp)
@printf("  --> CP Advantage from Joint:       %+.2f nats (%+.4f nats/spk)\n\n", delta_ll_1_cp, delta_ll_1_cp / N_eval_cp)

@printf("  Total Independent Sum (M_tot=%d):  +%.2f nats (+%.4f nats/spk)\n", M_a + M_b, dll_indep_total, dll_indep_total / N_eval_total)
@printf("  Total Joint Model C (M=%d):        +%.2f nats (+%.4f nats/spk)\n", M_c, dll_joint_total, dll_joint_total / N_eval_total)
@printf("  --> Net Joint Advantage (Delta LL_1): +%.2f nats (+%.4f nats/spk, +%.4f bits/spk)\n",
        delta_ll_1_total, delta_ll_1_total / N_eval_total, (delta_ll_1_total / N_eval_total) / log(2.0))

# =========================================================================
# STEP 3: EXPERIMENT 2 LEAK-FREE CROSS-REGION PREDICTION (ON HELD-OUT DATA)
# =========================================================================
println("\n" * "="^85)
println("EXPERIMENT 2: LEAK-FREE CROSS-REGION PREDICTION ON HELD-OUT BLOCK (1200 - 1700 s)")
println("="^85)

# 2A. MOp -> CP (Infer occurrences strictly from held-out MOp, evaluate on held-out CP)
model_cross_mop = initialize_scratch_model(
    data_te_mop, deepcopy(model_c.motifs);
    bg_rates=model_c.bg_rates, ard_state=model_c.ard_state, R_max=length(model_c.ard_state.active_mask)
)
fit!(model_cross_mop, data_te_mop; sweeps=15, allow_motif_birth=false, frozen_motif_ids=Set(keys(model_cross_mop.motifs)))
dll_cross_m2c, spk_m2c, comp_m2c = evaluate_channel_ll(
    model_cross_mop.occurrences, model_c.motifs, cp_channels,
    ts_te_eval, ch_te_eval, t_te_start, t_eval_start, t_eval_end, ou_full, bg_rates_full
)

gain_nat_m2c = dll_cross_m2c / N_eval_cp
gain_bit_m2c = gain_nat_m2c / log(2.0)

# 2B. CP -> MOp (Infer occurrences strictly from held-out CP, evaluate on held-out MOp)
model_cross_cp = initialize_scratch_model(
    data_te_cp, deepcopy(model_c.motifs);
    bg_rates=model_c.bg_rates, ard_state=model_c.ard_state, R_max=length(model_c.ard_state.active_mask)
)
fit!(model_cross_cp, data_te_cp; sweeps=15, allow_motif_birth=false, frozen_motif_ids=Set(keys(model_cross_cp.motifs)))
dll_cross_c2m, spk_c2m, comp_c2m = evaluate_channel_ll(
    model_cross_cp.occurrences, model_c.motifs, mop_channels,
    ts_te_eval, ch_te_eval, t_te_start, t_eval_start, t_eval_end, ou_full, bg_rates_full
)

gain_nat_c2m = dll_cross_c2m / N_eval_mop
gain_bit_c2m = gain_nat_c2m / log(2.0)

println("LEAK-FREE FULL-OCCURRENCE CROSS PREDICTION:")
@printf("  MOp -> CP: +%.2f nats across %d CP spikes (+%.4f nats/spk, +%.4f bits/spk)\n",
        dll_cross_m2c, N_eval_cp, gain_nat_m2c, gain_bit_m2c)
@printf("  CP -> MOp: +%.2f nats across %d MOp spikes (+%.4f nats/spk, +%.4f bits/spk)\n",
        dll_cross_c2m, N_eval_mop, gain_nat_c2m, gain_bit_c2m)

# =========================================================================
# STEP 4: EXPERIMENT 2c ASYMMETRIC CAUSAL LOOKAHEAD FORECASTING
# =========================================================================
# Prospectively forecast spikes in [tau_e + 200ms, tau_e + 450ms] (250 ms window)
# In MOp -> CP: MOp spikes (t < tau_e + 200ms) infer tau_e, forecasting CP in [tau_e + 200ms, tau_e + 450ms].
# In CP -> MOp: CP spikes infer tau_e, forecasting MOp in [tau_e + 200ms, tau_e + 450ms].

println("\n" * "="^85)
println("EXPERIMENT 2c: PROSPECTIVE CAUSAL LOOKAHEAD FORECASTING (Delta = 250 ms)")
println("="^85)

function evaluate_causal_lookahead(
    inferred_occurrences,
    motifs,
    target_channels::Set{Int},
    ts_eval::Vector{Float64},
    ch_eval::Vector{Int},
    t_start_abs::Float64,
    ou_full,
    bg_rates_full;
    tau_offset=0.200,
    lookahead_win=0.250
)
    # Define prospective forecasting windows [t_k, t_k + lookahead_win] for each occurrence
    # where t_k = tau_e + tau_offset
    forecast_windows = Tuple{Float64, Float64}[]
    for occ in values(inferred_occurrences)
        t_k = occ.tau + tau_offset
        push!(forecast_windows, (t_k, t_k + lookahead_win))
    end

    # Sum log-likelihood specifically across spikes and compensator in these prospective windows
    active_instances = Dict{Int, Vector{Tuple{Float64, Float64, Float64}}}()
    compensator_penalty = 0.0

    for occ in values(inferred_occurrences)
        m = occ.motif
        tau_e = occ.tau
        c_e   = occ.amplitude
        u_e   = occ.u
        t_w1  = tau_e + tau_offset
        t_w2  = t_w1 + lookahead_win
        
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
                
                # Compensator in [t_w1, t_w2]
                z2 = (t_w2 - mu_tilde) / (sqrt(2.0) * sigma_p)
                z1 = (t_w1 - mu_tilde) / (sqrt(2.0) * sigma_p)
                p_mass = 0.5 * (erf(z2) - erf(z1))
                compensator_penalty += c_e * a_p * p_mass

                if mu_tilde + 4.0 * sigma_p >= t_w1 && mu_tilde - 4.0 * sigma_p <= t_w2
                    if !haskey(active_instances, chan)
                        active_instances[chan] = Tuple{Float64, Float64, Float64}[]
                    end
                    push!(active_instances[chan], (c_e * a_p, mu_tilde, var_p))
                end
            end
        end
    end

    # Filter spikes that fall strictly into any lookahead window
    target_spk_in_windows = 0
    spike_gain = 0.0

    for (spk_idx, t_spk) in enumerate(ts_eval)
        c_spk = ch_eval[spk_idx]
        if !(c_spk in target_channels)
            continue
        end

        # Check if t_spk falls inside any prospective window
        in_win = false
        for (w1, w2) in forecast_windows
            if w1 <= t_spk <= w2
                in_win = true
                break
            end
        end
        if !in_win
            continue
        end

        target_spk_in_windows += 1
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
    return delta_ll, target_spk_in_windows
end

lookahead_dll_m2c, N_lookahead_cp = evaluate_causal_lookahead(
    model_cross_mop.occurrences, model_c.motifs, cp_channels,
    ts_te_eval, ch_te_eval, t_te_start, ou_full, bg_rates_full
)

lookahead_dll_c2m, N_lookahead_mop = evaluate_causal_lookahead(
    model_cross_cp.occurrences, model_c.motifs, mop_channels,
    ts_te_eval, ch_te_eval, t_te_start, ou_full, bg_rates_full
)

lookahead_gain_m2c = lookahead_dll_m2c / N_lookahead_cp
lookahead_gain_c2m = lookahead_dll_c2m / N_lookahead_mop
causal_diff = lookahead_gain_m2c - lookahead_gain_c2m
causal_ratio = lookahead_gain_m2c / max(lookahead_gain_c2m, 1e-12)

@printf("  MOp -> CP Causal Lookahead: +%.2f nats across %d CP window spikes (+%.4f nats/spk)\n",
        lookahead_dll_m2c, N_lookahead_cp, lookahead_gain_m2c)
@printf("  CP -> MOp Causal Lookahead: %+.2f nats across %d MOp window spikes (%+.4f nats/spk)\n",
        lookahead_dll_c2m, N_lookahead_mop, lookahead_gain_c2m)
@printf("  Net Causal Advantage (MOp -> CP minus CP -> MOp): +%.4f nats / spike\n", causal_diff)
@printf("  Causal Directionality Factor: %.2fx stronger feedforward forecasting\n", causal_ratio)
println("="^85)

results = Dict(
    :M_a => M_a, :P_a => P_a, :dll_model_a => dll_model_a,
    :M_b => M_b, :P_b => P_b, :dll_model_b => dll_model_b,
    :M_c => M_c, :P_c => P_c,
    :dll_joint_mop => dll_joint_mop, :dll_joint_cp => dll_joint_cp, :dll_joint_total => dll_joint_total,
    :delta_ll_1_total => delta_ll_1_total, :delta_ll_1_mop => delta_ll_1_mop, :delta_ll_1_cp => delta_ll_1_cp,
    :dll_cross_m2c => dll_cross_m2c, :gain_nat_m2c => gain_nat_m2c,
    :dll_cross_c2m => dll_cross_c2m, :gain_nat_c2m => gain_nat_c2m,
    :lookahead_dll_m2c => lookahead_dll_m2c, :lookahead_gain_m2c => lookahead_gain_m2c,
    :lookahead_dll_c2m => lookahead_dll_c2m, :lookahead_gain_c2m => lookahead_gain_c2m,
    :causal_diff => causal_diff, :causal_ratio => causal_ratio
)

serialize("/home/foaleng/Documents/PurePacketPPSeq/scripts/leak_free_validation_results.jls", results)
println("All results saved to /home/foaleng/Documents/PurePacketPPSeq/scripts/leak_free_validation_results.jls")
