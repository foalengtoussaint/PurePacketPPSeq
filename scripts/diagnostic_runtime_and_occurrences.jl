# scripts/diagnostic_runtime_and_occurrences.jl
# Read-only diagnostic script to profile Sweep 60 runtime breakdown,
# motif occurrence inflation (Sweep 52 vs Sweep 60), and trace LLR on newborn motifs.

include("/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/regime_triad_sampler/run_regime_sampler.jl")
using PurePacketPPSeq
using Printf
using Statistics
using Random

println("=== STEP 1: LOADING DATASET AND CHECKPOINTS ===")
data = load_awake_spikes()
ck_path_52 = joinpath(@__DIR__, "checkpoint_pure_packet_cold_start_52sweeps.jls")
ck_path_60 = joinpath(@__DIR__, "checkpoint_pure_packet_cold_start_60sweeps.jls")

println("Loading Checkpoint 52: ", ck_path_52)
model_52 = resume_model_from_checkpoint(ck_path_52, data; R_max=4)

println("Loading Checkpoint 60: ", ck_path_60)
model_60 = resume_model_from_checkpoint(ck_path_60, data; R_max=4)

println("\n" * "="^80)
println("DIAGNOSTIC 2: PER-MOTIF OCCURRENCE & PACKET BREAKDOWN (SWEEP 52 vs SWEEP 60)")
println("="^80)

all_mids = sort(unique(vcat(collect(keys(model_52.motifs)), collect(keys(model_60.motifs)))))

println(@sprintf("%-5s | %-12s | %-12s | %-12s | %-12s | %-10s | %-10s",
        "Motif", "E_m (Sw52)", "E_m (Sw60)", "P_m (Sw52)", "P_m (Sw60)", "A_m (Sw52)", "A_m (Sw60)"))
println("-"^80)

for mid in all_mids
    e52 = count(occ -> occ.motif == mid, values(model_52.occurrences))
    e60 = count(occ -> occ.motif == mid, values(model_60.occurrences))
    
    p52 = haskey(model_52.motifs, mid) ? length(model_52.motifs[mid].packets) : 0
    p60 = haskey(model_60.motifs, mid) ? length(model_60.motifs[mid].packets) : 0

    a52 = haskey(model_52.motifs, mid) ? sum(p.mass for p in model_52.motifs[mid].packets) : 0.0
    a60 = haskey(model_60.motifs, mid) ? sum(p.mass for p in model_60.motifs[mid].packets) : 0.0

    # Active occurrences (at least 1 spike)
    act52 = count(occ -> occ.motif == mid && sum(occ.counts) > 0, values(model_52.occurrences))
    act60 = count(occ -> occ.motif == mid && sum(occ.counts) > 0, values(model_60.occurrences))

    # Clean occurrences (>= 2 active packets)
    clean52 = count(occ -> occ.motif == mid && count(c -> c > 0, occ.counts) >= 2, values(model_52.occurrences))
    clean60 = count(occ -> occ.motif == mid && count(c -> c > 0, occ.counts) >= 2, values(model_60.occurrences))

    println(@sprintf("%-5d | %-5d (%-4d) | %-5d (%-4d) | %-12d | %-12d | %-10.2f | %-10.2f",
            mid, e52, clean52, e60, clean60, p52, p60, a52, a60))
end
println("-"^80)
total_e52 = length(model_52.occurrences)
total_e60 = length(model_60.occurrences)
total_p52 = sum(length(m.packets) for m in values(model_52.motifs))
total_p60 = sum(length(m.packets) for m in values(model_60.motifs))
println(@sprintf("TOTAL | %-12d | %-12d | %-12d | %-12d |", total_e52, total_e60, total_p52, total_p60))

println("\n" * "="^80)
println("DIAGNOSTIC 3: CHANNEL BASELINE RATES & AUDIT OF NEWBORN MOTIFS (MOTIFS 8..12)")
println("="^80)

bg60 = model_60.bg_rates
println("Channel Background Baseline Rates (Sweep 60, N=411):")
println(@sprintf("  Min:    %.6f Hz", minimum(bg60)))
println(@sprintf("  1%%:     %.6f Hz", quantile(bg60, 0.01)))
println(@sprintf("  5%%:     %.6f Hz", quantile(bg60, 0.05)))
println(@sprintf("  10%%:    %.6f Hz", quantile(bg60, 0.10)))
println(@sprintf("  Median: %.6f Hz", median(bg60)))
println(@sprintf("  Mean:   %.6f Hz", mean(bg60)))
println(@sprintf("  Max:    %.6f Hz", maximum(bg60)))

low_rate_chans = findall(c -> bg60[c] < 0.10, 1:length(bg60))
very_low_rate_chans = findall(c -> bg60[c] < 0.01, 1:length(bg60))
println(@sprintf("Channels with bg_rate < 0.10 Hz: %d / %d (%.1f%%)",
        length(low_rate_chans), length(bg60), 100.0 * length(low_rate_chans) / length(bg60)))
println(@sprintf("Channels with bg_rate < 0.01 Hz: %d / %d (%.1f%%)",
        length(very_low_rate_chans), length(bg60), 100.0 * length(very_low_rate_chans) / length(bg60)))

for mid in 8:12
    if haskey(model_60.motifs, mid)
        m = model_60.motifs[mid]
        P = length(m.packets)
        chans = [p.channel for p in m.packets]
        bgs = [bg60[c] for c in chans]
        masses = [p.mass for p in m.packets]
        mus = [p.mu for p in m.packets]
        e_count = count(occ -> occ.motif == mid, values(model_60.occurrences))
        spk_count = sum(sum(occ.counts) for occ in values(model_60.occurrences) if occ.motif == mid; init=0)

        println(@sprintf("\n--- Motif %d (Packets: %d, Occurrences: %d, Spikes: %d) ---", mid, P, e_count, spk_count))
        for p in 1:P
            println(@sprintf("  Packet %d: Chan=%d | lambda_0=%.6f Hz | mu=%.4fs | mass=%.4f | var=%.6f",
                    p, chans[p], bgs[p], mus[p], masses[p], m.packets[p].var))
        end
        # Theoretical single spike LLR at dt=0 for each packet
        sig_prior = 0.015
        println("  Single-spike bump / lambda_0:")
        for p in 1:P
            c = chans[p]
            lam = bgs[p]
            # If 1 spike: a_post = (1+1)/(1+1) = 1.0
            bump = 1.0 / (sqrt(2π) * sig_prior)
            ratio = bump / lam
            llr_1spk = log(max(ratio, 1.0))
            println(@sprintf("    Packet %d (Chan %d): bump=%.2f Hz, lam0=%.6f Hz => bump/lam0 = %.1f => ln(bump/lam0) = %.2f nats",
                    p, c, bump, lam, ratio, llr_1spk))
        end
    end
end

println("\n" * "="^80)
println("DIAGNOSTIC 1: PER-STEP WALL-CLOCK BREAKDOWN ON SWEEP 60")
println("="^80)

# We benchmark a single sweep step by step on model_60
rng = Random.default_rng()
thread_bufs = PurePacketPPSeq.allocate_thread_buffers(Threads.maxthreadid(); rng=rng)
birth_bufs = PurePacketPPSeq.allocate_birth_buffers(data.chan_to_spikes)

T_total = data.T_total
R_max = model_60.R_max
motifs = deepcopy(model_60.motifs)
occurrences = deepcopy(model_60.occurrences)
assign = deepcopy(model_60.assign)
bg_rates = copy(model_60.bg_rates)
counts_bg = copy(model_60.counts_bg)
ard_state = deepcopy(model_60.ard_state)
ou_ctx = deepcopy(model_60.ou_ctx)
spike_t = data.t
spike_neuron = data.neuron
chan_to_spikes = data.chan_to_spikes
N_channels = length(bg_rates)
gamma_m = 100.0
alpha_c = 10.0
beta_c = 10.0
alpha_a = 1.0
beta_a = 1.0
alpha_m = 0.05
beta_m = 1.0
delta_margin = 3.0
frozen_motif_ids = Set{Int}()
allow_motif_birth = true
sweep = 61

step_times = Dict{String, Float64}()
step_allocs = Dict{String, Float64}()

# Step 0: Motif Archetype Birth
res = @timed PurePacketPPSeq.detect_and_birth_new_motifs!(
    motifs, occurrences, assign, spike_t, spike_neuron,
    bg_rates, T_total, length(ard_state.active_mask);
    N_channels=N_channels, occam_motif=nothing, gamma_motif=gamma_m,
    alpha_a=alpha_a, beta_a=beta_a, alpha_c=alpha_c, beta_c=beta_c,
    alpha_m=alpha_m, beta_m=beta_m,
    counts_bg=counts_bg, ou_ctx=ou_ctx, rng=rng
)
step_times["Step 0: m=3 Motif Birth"] = res.time
step_allocs["Step 0: m=3 Motif Birth"] = res.bytes / 1e6

# Step 1: Occurrence Candidate Birth
res = @timed PurePacketPPSeq.detect_and_birth_occurrences!(
    occurrences, motifs, spike_t, spike_neuron, chan_to_spikes,
    assign, bg_rates, T_total, R_max;
    active_mask=ard_state.active_mask,
    alpha_m=alpha_m, beta_m=beta_m,
    gamma_m=gamma_m, alpha_c=alpha_c, beta_c=beta_c, delta_margin=delta_margin,
    bg_times=birth_bufs.bg_times, bg_indices=birth_bufs.bg_indices,
    ou_ctx=ou_ctx, rng=rng
)
step_times["Step 1: m=2 Occurrence Birth"] = res.time
step_allocs["Step 1: m=2 Occurrence Birth"] = res.bytes / 1e6

# Step 2: Marginal Negative Binomial Occurrence Pruning
res = @timed PurePacketPPSeq.prune_dead_occurrences!(
    occurrences, motifs, spike_t, assign, bg_rates, T_total, ard_state.active_mask;
    chan_to_spikes=chan_to_spikes, counts_bg=counts_bg,
    alpha_c=alpha_c, beta_c=beta_c, alpha_m=alpha_m, beta_m=beta_m, gamma_m=gamma_m,
    ou_ctx=ou_ctx, rng=rng
)
step_times["Step 2: m=2 Occurrence Prune"] = res.time
step_allocs["Step 2: m=2 Occurrence Prune"] = res.bytes / 1e6

# Step 3: Dual-Index Spike Resampling (Multi-threaded)
res = @timed begin
    chan_to_motifs = PurePacketPPSeq.update_chan_to_motifs(motifs, N_channels)
    motif_occs = Dict{Int, Vector{PurePacketPPSeq.PureOccurrence}}()
    motif_taus = Dict{Int, Vector{Float64}}()
    for mid in keys(motifs)
        occs_m = [occ for occ in values(occurrences) if occ.motif == mid]
        sort!(occs_m, by=occ -> occ.tau)
        motif_occs[mid] = occs_m
        motif_taus[mid] = [occ.tau for occ in occs_m]
    end

    PurePacketPPSeq.resample_all_spikes_pure_packet_threaded!(
        assign, spike_t, spike_neuron,
        chan_to_motifs, motif_occs, motif_taus,
        occurrences, motifs, bg_rates, ard_state.active_mask,
        thread_bufs.tl_occ, thread_bufs.tl_pkt, thread_bufs.tl_logw, thread_bufs.tl_rng;
        ou_ctx=ou_ctx
    )
end
step_times["Step 3: Spike Resampling"] = res.time
step_allocs["Step 3: Spike Resampling"] = res.bytes / 1e6

# Step 3b: Prune dead occurrences after resampling
res = @timed PurePacketPPSeq.prune_dead_occurrences!(
    occurrences, motifs, spike_t, assign, bg_rates, T_total, ard_state.active_mask;
    chan_to_spikes=chan_to_spikes, counts_bg=counts_bg,
    alpha_c=alpha_c, beta_c=beta_c, alpha_m=alpha_m, beta_m=beta_m,
    gamma_m=gamma_m, ou_ctx=ou_ctx, rng=rng
)
step_times["Step 3b: Occ Prune (Post-Resample)"] = res.time
step_allocs["Step 3b: Occ Prune (Post-Resample)"] = res.bytes / 1e6

# Step 4: Occurrence Coordinates u_e
res = @timed begin
    for (eid, occ) in occurrences
        pm = motifs[occ.motif]
        P = length(pm.packets)
        res_v = zeros(Float64, P)
        prec_v = zeros(Float64, P)
        for p in 1:P
            c = occ.counts[p]
            if c > 0
                mean_t = occ.sum_t[p] / c
                res_v[p] = mean_t - (occ.tau + pm.packets[p].mu)
                prec_v[p] = c / pm.packets[p].var
            end
        end
        tau_prec = get(ou_ctx.tau_precisions, eid, 1.0)
        PurePacketPPSeq.sample_occurrence_coordinate!(occ.u, pm, ard_state.active_mask, res_v, prec_v; tau_precision=tau_prec, rng=rng)
    end
end
step_times["Step 4: Kinematics u_e"] = res.time
step_allocs["Step 4: Kinematics u_e"] = res.bytes / 1e6

# Step 5: Loadings W
res = @timed begin
    for (mid, pm) in motifs
        occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
        isempty(occ_list) && continue
        P = length(pm.packets)
        E_m = length(occ_list)
        u_list = [occ.u for occ in occ_list]
        res_list = Vector{Vector{Float64}}(undef, E_m)
        prec_list = Vector{Vector{Float64}}(undef, E_m)
        for (idx, occ) in enumerate(occ_list)
            res_v = zeros(Float64, P)
            prec_v = zeros(Float64, P)
            for p in 1:P
                c = occ.counts[p]
                if c > 0
                    mean_t = occ.sum_t[p] / c
                    res_v[p] = mean_t - (occ.tau + pm.packets[p].mu)
                    prec_v[p] = c / pm.packets[p].var
                end
            end
            res_list[idx] = res_v
            prec_list[idx] = prec_v
        end
        PurePacketPPSeq.sample_packet_loadings!(pm, ard_state.alpha, ard_state.active_mask, u_list, res_list, prec_list; rng=rng)
    end
end
step_times["Step 5: Loadings W"] = res.time
step_allocs["Step 5: Loadings W"] = res.bytes / 1e6

# Step 6: ARD Precisions
res = @timed begin
    all_pm = collect(values(motifs))
    PurePacketPPSeq.sample_global_ard_precisions!(ard_state, all_pm; rng=rng)
    PurePacketPPSeq.prune_inactive_axes!(ard_state, all_pm)
    PurePacketPPSeq.check_and_birth_axis!(ard_state, motifs, collect(values(occurrences)); rng=rng)
end
step_times["Step 6: ARD Precisions"] = res.time
step_allocs["Step 6: ARD Precisions"] = res.bytes / 1e6

# Step 7: Occurrence Onsets tau_e
res = @timed begin
    for occ in values(occurrences)
        pm = motifs[occ.motif]
        PurePacketPPSeq.resample_occurrence_onset!(occ, pm, ard_state.active_mask; T_max=T_total, rng=rng)
    end
end
step_times["Step 7: Onsets tau_e"] = res.time
step_allocs["Step 7: Onsets tau_e"] = res.bytes / 1e6

# Step 8: Corridors mu_p & var_p
res = @timed begin
    for (mid, pm) in motifs
        occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
        isempty(occ_list) && continue
        PurePacketPPSeq.update_packet_corridors!(pm, occ_list, ard_state.active_mask; anchor_origin=true, damp=0.2)
        PurePacketPPSeq.project_translational_gauge!(pm, ard_state.active_mask)
    end
end
step_times["Step 8: Corridors mu & var"] = res.time
step_allocs["Step 8: Corridors mu & var"] = res.bytes / 1e6

# Step 9: Packet Masses a_{m,p}
res = @timed begin
    for (mid, pm) in motifs
        occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
        isempty(occ_list) && continue
        PurePacketPPSeq.sample_packet_masses!(pm, occ_list; alpha_a=1.0, beta_a=1.0, bg_rates=bg_rates, ou_ctx=ou_ctx, rng=rng)
    end
end
step_times["Step 9: Masses a_{m,p}"] = res.time
step_allocs["Step 9: Masses a_{m,p}"] = res.bytes / 1e6

# Step 10: Packet Pruning
res = @timed begin
    for (mid, pm) in motifs
        occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
        isempty(occ_list) && continue
        PurePacketPPSeq.prune_dead_packets!(
            pm, occ_list, bg_rates, N_channels;
            spike_t=spike_t, spike_chan=spike_neuron, assign=assign,
            ou_ctx=ou_ctx, active_mask=ard_state.active_mask, ard_alphas=ard_state.alpha,
            sigma_prior=0.015, alpha_a=alpha_a, beta_a=beta_a,
            counts_bg=counts_bg, rng=rng
        )
        PurePacketPPSeq.project_translational_gauge!(pm, ard_state.active_mask)
    end
end
step_times["Step 10: m=1 Packet Prune"] = res.time
step_allocs["Step 10: m=1 Packet Prune"] = res.bytes / 1e6

# Step 11: Packet Birth & Split
res = @timed begin
    for (mid, pm) in motifs
        occ_list = [occ for occ in values(occurrences) if occ.motif == mid]
        isempty(occ_list) && continue
        PurePacketPPSeq.detect_and_birth_or_split_packets!(
            pm, occ_list, spike_t, spike_neuron, assign, bg_rates, N_channels;
            chan_spikes=chan_to_spikes, ou_ctx=ou_ctx, sigma_prior=0.015,
            alpha_a=alpha_a, beta_a=beta_a,
            active_mask=ard_state.active_mask, ard_alphas=ard_state.alpha
        )
        PurePacketPPSeq.project_translational_gauge!(pm, ard_state.active_mask)
    end
end
step_times["Step 11: m=1 Packet Birth/Split"] = res.time
step_allocs["Step 11: m=1 Packet Birth/Split"] = res.bytes / 1e6

# Step 11b: Motif Prune
res = @timed PurePacketPPSeq.prune_dead_motifs!(
    motifs, occurrences, frozen_motif_ids;
    spike_t=spike_t, spike_neuron=spike_neuron, assign=assign,
    bg_rates=bg_rates, counts_bg=counts_bg, active_mask=ard_state.active_mask,
    ou_ctx=ou_ctx, T_total=T_total,
    gamma_motif=gamma_m, alpha_c=alpha_c, beta_c=beta_c, beta_a=beta_a,
    alpha_m=alpha_m, beta_m=beta_m, sigma_prior=0.015, rng=rng
)
step_times["Step 11b: m=3 Motif Prune"] = res.time
step_allocs["Step 11b: m=3 Motif Prune"] = res.bytes / 1e6

# Step 12: Continuous OU & Conjugate Triad
res = @timed begin
    PurePacketPPSeq.update_regional_ou!(ou_ctx, assign, spike_t, spike_neuron, bg_rates; max_iter=2)
    PurePacketPPSeq.update_cell_ou!(ou_ctx, assign, spike_t, spike_neuron, bg_rates; max_iter=2)
    PurePacketPPSeq.update_motif_ou!(ou_ctx, occurrences, motifs; max_iter=2)
    PurePacketPPSeq.update_continuous_background_rates!(
        bg_rates, ou_ctx, counts_bg, assign, spike_neuron, T_total;
        alpha_bg=1.0, beta_bg=1.0, rng=rng
    )
    PurePacketPPSeq.update_vigor_shape!(ou_ctx, occurrences; max_iter=10, tol=1e-6)
    PurePacketPPSeq.sample_continuous_occurrence_amplitudes!(occurrences, motifs, ou_ctx.alpha_c; rng=rng)
    PurePacketPPSeq.sample_student_t_precisions!(ou_ctx, occurrences, count(ard_state.active_mask); rng=rng)
    PurePacketPPSeq.update_all_motif_bounds!(motifs)
end
step_times["Step 12: OU & Triad Adaptation"] = res.time
step_allocs["Step 12: OU & Triad Adaptation"] = res.bytes / 1e6

total_time = sum(values(step_times))

println(@sprintf("%-35s | %-10s | %-8s | %-10s", "Step", "Time (s)", "Share", "Allocs (MB)"))
println("-"^75)
for (k, v) in sort(collect(step_times), by=x -> x[2], rev=true)
    pct = (v / total_time) * 100.0
    alloc = step_allocs[k]
    println(@sprintf("%-35s | %-10.3f | %6.1f%% | %-10.1f", k, v, pct, alloc))
end
println("-"^75)
println(@sprintf("%-35s | %-10.3f | 100.0%% | %-10.1f", "TOTAL SWEEP TIME", total_time, sum(values(step_allocs))))
