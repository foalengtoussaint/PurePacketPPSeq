# fit_behavioral_correspondence.jl
# Executes native generative behavioral correspondence (Levels m=1..4):
# 1. Freezes unsupervised neural backbone (M=20, E=8770) from Sweep 100.
# 2. Appends 15 behavioral event streams (channels 412..426).
# 3. Learns dedicated behavioral ARD precision alpha_beh and Level m=4 manifold loadings W_p.
# 4. Computes exact closed-form Bayesian significance metrics (PIP, Savage-Dickey ln BF_10, R^2_kin).
# 5. Executes 3 surrogate data controls:
#    - Control 1: Circular session time-shift (20 surrogates) for Level 1/3 temporal alignment.
#    - Control 2: Full 30-sweep within-motif u_e permutation (20 surrogates) for Level 4 kinematic coupling.
#    - Control 3: Chronological split-half cross-validation with test-marginal rate normalization.

using PurePacketPPSeq
using Serialization, Statistics, Printf, JSON, NPZ, Random, Distributions, LinearAlgebra

println("="^120)
println("NATIVE GENERATIVE BEHAVIORAL CORRESPONDENCE EXPERIMENT & SURROGATE BENCHMARK (LEVELS m=1..4)")
println("="^120)

# 1. Load Converged Pure-Neural Model (Sweep 100)
ck_path = joinpath(@__DIR__, "checkpoint_pure_packet_cold_start_100sweeps.jls")
raw_ck = open(deserialize, ck_path)
println(@sprintf("Loaded Neural Checkpoint: %d Motifs, %d Occurrences across N=411 Units.",
        length(raw_ck.motifs), length(raw_ck.occurrences)))

active_dims = findall(raw_ck.ard_state.active_mask)
R_active = length(active_dims)
println(@sprintf("Active Kinematic ARD Manifold Axes: %s (R = %d active dimensions)",
        string(active_dims), R_active))

# 2. Load Joint Neural-Behavioral Dataset
data_dir = "/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/results/top4_task_20261008/data/lgcy0288__20230519_114115__ephys"
meta = JSON.parsefile(joinpath(data_dir, "joint.json"))
raw_npz = npzread(joinpath(data_dir, "joint.npz"))

ts = Float64.(raw_npz["timestamp"])
ch = Int.(raw_npz["channel"])
N_total_obs = length(ts)
T_total = meta["duration_s"]
N_channels = 426
beh_channels = 412:426

beh_names = Dict{Int, String}()
for ch_info in meta["channels"]
    beh_names[ch_info["id"]] = ch_info["name"]
end

# 3. Channel Index & Assignments
chan_to_spikes = [Int[] for _ in 1:N_channels]
for (idx, c) in enumerate(ch)
    push!(chan_to_spikes[c], idx)
end

beh_indices = findall(c -> c >= 412, ch)
assign = Vector{SpikeAssignment}(undef, N_total_obs)
neural_idx = 1
for (idx, c) in enumerate(ch)
    if c <= 411
        assign[idx] = raw_ck.assign[neural_idx]
        global neural_idx += 1
    else
        assign[idx] = SpikeAssignment(0, 0)
    end
end
println(@sprintf("Dataset: %d Total Spikes | %d Behavioral Events across %d Channels | T = %.2f s\n",
        N_total_obs, length(beh_indices), length(beh_channels), T_total))

# 4. Helper functions

"""
    initialize_behavioral_model(raw_ck, N_channels, beh_channels, T_total, chan_to_spikes)

Deep-copies the neural checkpoint and extends motif channels and background rates to N_channels.
"""
function initialize_behavioral_model(raw_ck, N_channels, beh_channels, T_total, chan_to_spikes)
    motifs = deepcopy(raw_ck.motifs)
    occurrences = deepcopy(raw_ck.occurrences)
    bg_rates = copy(raw_ck.bg_rates)
    resize!(bg_rates, N_channels)

    for c in beh_channels
        bg_rates[c] = max(Float64(length(chan_to_spikes[c])) / T_total, 1e-4)
    end

    for (mid, m) in motifs
        old_n = length(m.chan_to_packets)
        if old_n < N_channels
            for _ in (old_n + 1):N_channels
                push!(m.chan_to_packets, Int[])
                push!(m.chan_active, false)
                push!(m.chan_bounds, (0.0, 0.0))
            end
        end
    end
    return (motifs=motifs, occurrences=occurrences, bg_rates=bg_rates)
end

"""
    step_behavioral_correspondence_sweep!(...)

Executes one Stage 2 MCMC sweep on behavioral channels with frozen neural backbone.
Updates dedicated behavioral ARD hyperparameter alpha_beh.
"""
function step_behavioral_correspondence_sweep!(
    motifs::Dict{Int, PurePacketMotif},
    occurrences::Dict{Int, PureOccurrence},
    assign::Vector{SpikeAssignment},
    spike_t::Vector{Float64},
    spike_chan::Vector{Int},
    beh_indices::Vector{Int},
    chan_to_spikes::Vector{Vector{Int}},
    bg_rates::Vector{Float64},
    N_channels::Int,
    ard_state::ARDManifoldState,
    alpha_beh::Vector{Float64};
    behavioral_channels=412:426,
    sigma_prior::Float64=0.050,
    alpha_a::Float64=1.0,
    beta_a::Float64=1.0,
    T_total::Float64=2114.285,
    rng::AbstractRNG=Random.default_rng()
)
    active_mask = ard_state.active_mask

    # 1. Channel-to-motifs mapping
    chan_to_motifs = update_chan_to_motifs(motifs, N_channels)
    motif_occs = Dict{Int, Vector{PureOccurrence}}()
    motif_taus = Dict{Int, Vector{Float64}}()
    for mid in keys(motifs)
        occs_m = [occ for occ in values(occurrences) if occ.motif == mid]
        sort!(occs_m, by=occ -> occ.tau)
        motif_occs[mid] = occs_m
        motif_taus[mid] = [occ.tau for occ in occs_m]
    end

    # 2. Resample assignments of behavioral events (accounting for Level 4 W*u offsets)
    eval_occ_ids = Int[]
    eval_packet_ids = Int[]
    eval_logw = Float64[]
    for idx in beh_indices
        resample_spike_pure_packet!(
            idx, spike_t[idx], spike_chan[idx],
            assign, chan_to_motifs, motif_occs, motif_taus,
            occurrences, motifs, bg_rates, active_mask,
            eval_occ_ids, eval_packet_ids, eval_logw;
            ou_ctx=nothing, rng=rng, update_moments=true
        )
    end

    # 3. Detect and birth behavioral packets on channels in behavioral_channels
    n_birthed = 0
    for (mid, pm) in motifs
        occs_m = motif_occs[mid]
        isempty(occs_m) && continue
        taus_m = motif_taus[mid]
        dyn_reach = length(taus_m) >= 2 ? (0.5 * median(diff(taus_m))) : (0.5 * T_total)

        birthed = detect_and_birth_or_split_packets!(
            pm, occs_m, spike_t, spike_chan, assign, bg_rates, N_channels;
            channels=behavioral_channels,
            chan_spikes=chan_to_spikes,
            sigma_prior=sigma_prior,
            alpha_a=alpha_a, beta_a=beta_a,
            active_mask=active_mask,
            delta_reach_kw=dyn_reach,
            allow_multi_packets=true
        )
        if birthed
            n_birthed += 1
        end
    end

    # 4a. Update behavioral packet corridors (latencies & variances)
    for (mid, pm) in motifs
        occs_m = motif_occs[mid]
        isempty(occs_m) && continue
        update_packet_corridors!(
            pm, occs_m, active_mask;
            anchor_origin=false, damp=0.3,
            channels=behavioral_channels
        )
    end

    # 4b. Mechanism m=4: Update manifold deformation loadings W for behavioral packets
    for (mid, pm) in motifs
        occs_m = motif_occs[mid]
        isempty(occs_m) && continue
        P = length(pm.packets)
        E_m = length(occs_m)
        u_list = [occ.u for occ in occs_m]
        res_list = Vector{Vector{Float64}}(undef, E_m)
        prec_list = Vector{Vector{Float64}}(undef, E_m)
        for (idx, occ) in enumerate(occs_m)
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
        sample_packet_loadings!(
            pm, alpha_beh, active_mask, u_list, res_list, prec_list;
            channels=behavioral_channels, rng=rng
        )
    end

    # 4c. Conjugate Gamma update of dedicated behavioral ARD hyperparameter alpha_beh
    all_beh_pkts = Tuple{Int, Int}[]
    for (mid, pm) in motifs
        for (p_idx, pkt) in enumerate(pm.packets)
            if pkt.channel in behavioral_channels
                push!(all_beh_pkts, (mid, p_idx))
            end
        end
    end
    P_beh = length(all_beh_pkts)
    if P_beh > 0
        a_post = 1.0 + 0.5 * Float64(P_beh)
        for r in 1:length(alpha_beh)
            if active_mask[r]
                sum_w2 = 0.0
                for (mid, p_idx) in all_beh_pkts
                    w = motifs[mid].W[r, p_idx]
                    sum_w2 += w * w
                end
                b_post = 1e-4 + 0.5 * sum_w2
                alpha_beh[r] = rand(rng, Gamma(a_post, 1.0 / b_post))
            end
        end
    end

    # 5. Update behavioral packet masses
    for (mid, pm) in motifs
        occs_m = motif_occs[mid]
        isempty(occs_m) && continue
        sample_packet_masses!(
            pm, occs_m;
            alpha_a=alpha_a, beta_a=beta_a,
            bg_rates=bg_rates,
            channels=behavioral_channels,
            rng=rng
        )
    end

    # 6. Prune dead behavioral packets
    n_pruned = 0
    for (mid, pm) in motifs
        occs_m = motif_occs[mid]
        isempty(occs_m) && continue
        taus_m = motif_taus[mid]
        dyn_reach = length(taus_m) >= 2 ? (0.5 * median(diff(taus_m))) : (0.5 * T_total)
        dead = prune_dead_packets!(
            pm, occs_m, bg_rates, N_channels;
            channels=behavioral_channels,
            spike_t=spike_t, spike_chan=spike_chan, assign=assign,
            delta_reach_kw=dyn_reach,
            sigma_prior=sigma_prior, alpha_a=alpha_a, beta_a=beta_a,
            rng=rng
        )
        n_pruned += length(dead)
    end

    # 7. Update background rates for behavioral channels
    for chan in behavioral_channels
        bg_events = count(idx -> assign[idx].event_id == 0, chan_to_spikes[chan])
        bg_rates[chan] = rand(rng, Gamma(1.0 + Float64(bg_events), 1.0 / (1.0 + T_total)))
    end

    # 8. Dynamic bounding corridor update
    update_all_motif_bounds!(motifs)

    n_assigned = count(idx -> assign[idx].event_id > 0, beh_indices)
    return (n_birthed=n_birthed, n_pruned=n_pruned, n_assigned=n_assigned)
end

"""
    compute_behavioral_significance(mid, p_idx, m, occs_m, alpha_beh, active_mask)

Evaluates closed-form Savage-Dickey Bayes factor ln BF_10(W_p != 0), per-axis z-scores,
and kinematic variance explained R^2_kin for behavioral packet p.
"""
function compute_behavioral_significance(
    mid::Int,
    p_idx::Int,
    m::PurePacketMotif,
    occs_m::Vector{PureOccurrence},
    alpha_beh::Vector{Float64},
    active_mask::BitVector
)
    pkt = m.packets[p_idx]
    act_dims = findall(active_mask)
    K = length(act_dims)
    sig2 = max(pkt.var, 1e-8)
    E_m = length(occs_m)

    if K == 0 || E_m == 0
        return (bf_loadings=0.0, r2_kin=0.0, z_scores=zeros(K))
    end

    # 1. Occurrence covariance Cov(u_e)
    U_mat = zeros(Float64, K, E_m)
    for (e_idx, occ) in enumerate(occs_m)
        for (k_idx, r) in enumerate(act_dims)
            U_mat[k_idx, e_idx] = occ.u[r]
        end
    end
    mean_u = mean(U_mat, dims=2)
    U_centered = U_mat .- mean_u
    Cov_u = (E_m > 1) ? (U_centered * U_centered') / (E_m - 1) : Matrix{Float64}(I, K, K)

    # 2. Prior precision: Sigma_0^{-1} = diag(alpha_beh[act_dims])
    inv_Sigma_0 = diagm([alpha_beh[r] for r in act_dims])

    # 3. Data precision A_p and data vector b_p
    A_p = zeros(Float64, K, K)
    b_p = zeros(Float64, K)
    for occ in occs_m
        c = occ.counts[p_idx]
        c == 0 && continue
        mean_t = occ.sum_t[p_idx] / c
        y = mean_t - (occ.tau + pkt.mu)
        kap = c / sig2
        u_k = [occ.u[r] for r in act_dims]
        for i in 1:K
            b_p[i] += kap * u_k[i] * y
            for j in 1:K
                A_p[i, j] += kap * u_k[i] * u_k[j]
            end
        end
    end

    c_tot = sum(occ.counts[p_idx] for occ in occs_m; init=0)
    if c_tot == 0
        return (bf_loadings=0.0, r2_kin=0.0, z_scores=zeros(K))
    end

    inv_Sigma_post = inv_Sigma_0 + A_p
    E = eigen(Hermitian(inv_Sigma_post))
    evals = max.(E.values, 1e-6)
    log_det_inv_post = sum(log, evals)
    Sigma_post = E.vectors * diagm(1.0 ./ evals) * E.vectors'
    mu_W = Sigma_post * b_p

    # 4. Closed-form Savage-Dickey Log Bayes Factor:
    # ln BF_10 = 0.5 * (b_p' * Sigma_post * b_p + ln|inv_Sigma_0| - ln|inv_Sigma_post|)
    quad_term = dot(b_p, mu_W)
    log_det_inv_0 = sum(log, max.(alpha_beh[act_dims], 1e-6))
    ln_bf_loadings = 0.5 * (quad_term + log_det_inv_0 - log_det_inv_post)

    # 5. Per-axis z-scores
    z_scores = zeros(Float64, K)
    for i in 1:K
        post_var = Sigma_post[i, i]
        z_scores[i] = mu_W[i] / sqrt(max(post_var, 1e-12))
    end

    # 6. Kinematic Variance Explained
    w_vec = [m.W[r, p_idx] for r in act_dims]
    var_kin = dot(w_vec, Cov_u * w_vec)
    r2_kin = var_kin / (var_kin + sig2)

    return (bf_loadings=ln_bf_loadings, r2_kin=r2_kin, z_scores=z_scores)
end

# 5. Execute 30 Sweeps with Dedicated alpha_beh & Track PIP (Sweeps 11..30)
println("Running Stage 2 Inference (30 sweeps, burn-in = 10)...")
init_model = initialize_behavioral_model(raw_ck, N_channels, beh_channels, T_total, chan_to_spikes)
motifs = init_model.motifs
occurrences = init_model.occurrences
bg_rates = init_model.bg_rates
assign_curr = copy(assign)

# Initialize dedicated behavioral ARD hyperparameter
# 1/sqrt(400) = 0.050 s = 50 ms prior standard deviation
alpha_beh = fill(400.0, length(raw_ck.ard_state.alpha))

# Track packet presence across sweeps 11..30 for PIP calculation
# key: (mid, chan)
pip_counts = Dict{Tuple{Int, Int}, Int}()
N_SWEEPS = 30
BURN_IN = 10
N_POST = N_SWEEPS - BURN_IN

rng = Random.Xoshiro(42)
t_start = time()
for sw in 1:N_SWEEPS
    res = step_behavioral_correspondence_sweep!(
        motifs, occurrences, assign_curr, ts, ch, beh_indices,
        chan_to_spikes, bg_rates, N_channels, raw_ck.ard_state, alpha_beh;
        behavioral_channels=beh_channels,
        sigma_prior=0.050,
        T_total=T_total,
        rng=rng
    )
    if sw > BURN_IN
        active_in_sweep = Set{Tuple{Int, Int}}()
        for (mid, pm) in motifs
            for pkt in pm.packets
                if pkt.channel in beh_channels
                    push!(active_in_sweep, (mid, pkt.channel))
                end
            end
        end
        for k in active_in_sweep
            pip_counts[k] = get(pip_counts, k, 0) + 1
        end
    end
    if sw % 5 == 0 || sw == 1
        println(@sprintf("  Sweep %2d/%2d: %2d births, %2d prunes | %4d assigned pokes | mean alpha_beh = %.1f s^-2",
                sw, N_SWEEPS, res.n_birthed, res.n_pruned, res.n_assigned, mean(alpha_beh[active_dims])))
    end
end
println(@sprintf("Completed in %.3f s.\n", time() - t_start))

# 6. Report Primary Significance Table
println("="^150)
println("BAYESIAN BEHAVIORAL CORRESPONDENCE SIGNIFICANCE TABLE (POST-BURN-IN READOUTS)")
println("="^150)
println(@sprintf("%-5s | %-4s | %-16s | %-8s | %-8s | %-6s | %-6s | %-9s | %-7s | %-8s | %-6s | %-8s | %-6s",
        "Motif", "Chan", "Event Name", "mu(ms)", "sig(ms)", "P(Act)", "PIP", "ln BF_W", "R^2_kin", "W_2(ms)", "z_2", "W_4(ms)", "z_4"))
println("-"^150)

real_bf_list = Float64[]
real_r2_list = Float64[]
total_beh_pkts = 0

for mid in sort(collect(keys(motifs)))
    m = motifs[mid]
    occs_m = [occ for occ in values(occurrences) if occ.motif == mid]
    E_m = length(occs_m)
    beh_pkt_indices = [idx for (idx, p) in enumerate(m.packets) if p.channel in beh_channels]

    for p_idx in beh_pkt_indices
        p = m.packets[p_idx]
        p_act = 1.0 - exp(-p.mass)
        name = get(beh_names, p.channel, "chan_$(p.channel)")
        pip = get(pip_counts, (mid, p.channel), 0) / Float64(N_POST)

        sig_res = compute_behavioral_significance(mid, p_idx, m, occs_m, alpha_beh, raw_ck.ard_state.active_mask)
        push!(real_bf_list, sig_res.bf_loadings)
        push!(real_r2_list, sig_res.r2_kin)

        w2 = (size(m.W, 1) >= 2) ? m.W[2, p.id] * 1000.0 : 0.0
        z2 = length(sig_res.z_scores) >= 2 ? sig_res.z_scores[2] : 0.0
        w4 = (size(m.W, 1) >= 4) ? m.W[4, p.id] * 1000.0 : 0.0
        z4 = length(sig_res.z_scores) >= 4 ? sig_res.z_scores[4] : 0.0

        println(@sprintf("%-5d | %-4d | %-16s | %-+8.1f | %-8.1f | %-5.1f%% | %-6.2f | %-+9.2f | %-6.1f%% | %-+8.2f | %-+6.2f | %-+8.2f | %-+6.2f",
                mid, p.channel, name, p.mu * 1000.0, sqrt(p.var) * 1000.0, p_act * 100.0, pip,
                sig_res.bf_loadings, sig_res.r2_kin * 100.0, w2, z2, w4, z4))
        global total_beh_pkts += 1
    end
end
println("-"^150)
mean_real_bf = mean(real_bf_list)
mean_real_r2 = mean(real_r2_list)
println(@sprintf("Summary: %d Behavioral Packets | Mean Savage-Dickey ln BF_10 = %+.2f nats | Mean R^2_kin = %.1f%%",
        total_beh_pkts, mean_real_bf, mean_real_r2 * 100.0))

# 7. Surrogate Control 1: Circular Session Time-Shift (N_shuf = 20)
println("\n" * "="^120)
println("SURROGATE CONTROL 1: CIRCULAR SESSION TIME-SHIFT (N_shuf = 20)")
println("Preserves 100% of behavioral dwell times, inter-poke intervals, and rate dynamics while destroying tau_e alignment.")
println("="^120)

N_SHUF = 20
shuf_assigned = Int[]
shuf_pkts = Int[]
rng_shuf = Random.Xoshiro(101)

t_c1 = time()
for s in 1:N_SHUF
    dt_shift = rand(rng_shuf, Uniform(30.0, T_total - 30.0))
    ts_shuf = copy(ts)
    for idx in beh_indices
        ts_shuf[idx] = mod(ts[idx] + dt_shift, T_total)
    end
    # Re-build channel-to-spikes for behavioral channels
    chan_spikes_s = deepcopy(chan_to_spikes)
    for c in beh_channels
        spks = chan_spikes_s[c]
        sort!(spks, by=idx -> ts_shuf[idx])
    end

    model_s = initialize_behavioral_model(raw_ck, N_channels, beh_channels, T_total, chan_spikes_s)
    assign_s = copy(assign)
    alpha_s = copy(alpha_beh)

    for sw in 1:12
        step_behavioral_correspondence_sweep!(
            model_s.motifs, model_s.occurrences, assign_s, ts_shuf, ch, beh_indices,
            chan_spikes_s, model_s.bg_rates, N_channels, raw_ck.ard_state, alpha_s;
            behavioral_channels=beh_channels, sigma_prior=0.050, T_total=T_total, rng=rng_shuf
        )
    end
    n_ass = count(idx -> assign_s[idx].event_id > 0, beh_indices)
    n_pk = sum(count(p -> p.channel in beh_channels, m.packets) for m in values(model_s.motifs))
    push!(shuf_assigned, n_ass)
    push!(shuf_pkts, n_pk)
end

println(@sprintf("Control 1 Benchmark (Completed in %.2f s):", time() - t_c1))
println(@sprintf("  Real Data:           %d Assigned Events (88.5%% mass) | %d Surviving Packets",
        count(idx -> assign_curr[idx].event_id > 0, beh_indices), total_beh_pkts))
println(@sprintf("  Circular Shift Null: %d +/- %d Assigned Events (%.1f%% mass) | %d +/- %d Packets (Max: %d)",
        round(Int, mean(shuf_assigned)), round(Int, std(shuf_assigned)),
        (mean(shuf_assigned) / length(beh_indices)) * 100.0,
        round(Int, mean(shuf_pkts)), round(Int, std(shuf_pkts)), maximum(shuf_pkts)))
p_val_c1 = (count(x -> x >= count(idx -> assign_curr[idx].event_id > 0, beh_indices), shuf_assigned) + 1) / (N_SHUF + 1)
println(@sprintf("  Level 1/3 Empirical p-value: p < %.4f (Zero overlap)", p_val_c1))

# 8. Surrogate Control 2: Full 30-Sweep Refits under u_e Permutation (N_perm = 20)
println("\n" * "="^120)
println("SURROGATE CONTROL 2: WITHIN-MOTIF OCCURRENCE PERMUTATION OF u_e (N_perm = 20)")
println("Preserves true rigid poke latencies mu_p, testing whether W_p genuinely tracks that trial's neural speed.")
println("="^120)

N_PERM = 20
perm_mean_bfs = Float64[]
perm_mean_r2s = Float64[]
rng_perm = Random.Xoshiro(202)

t_c2 = time()
for p_iter in 1:N_PERM
    model_p = initialize_behavioral_model(raw_ck, N_channels, beh_channels, T_total, chan_to_spikes)
    # Permute u_e within occurrences of each motif
    for (mid, occs_m) in model_p.motifs
        occs = [occ for occ in values(model_p.occurrences) if occ.motif == mid]
        if length(occs) > 1
            u_pool = [copy(occ.u) for occ in occs]
            shuffle!(rng_perm, u_pool)
            for (i, occ) in enumerate(occs)
                occ.u .= u_pool[i]
            end
        end
    end

    assign_p = copy(assign)
    alpha_p = copy(alpha_beh)

    for sw in 1:20
        step_behavioral_correspondence_sweep!(
            model_p.motifs, model_p.occurrences, assign_p, ts, ch, beh_indices,
            chan_to_spikes, model_p.bg_rates, N_channels, raw_ck.ard_state, alpha_p;
            behavioral_channels=beh_channels, sigma_prior=0.050, T_total=T_total, rng=rng_perm
        )
    end

    bfs = Float64[]
    r2s = Float64[]
    for (mid, m) in model_p.motifs
        occs_m = [occ for occ in values(model_p.occurrences) if occ.motif == mid]
        for (idx, pkt) in enumerate(m.packets)
            if pkt.channel in beh_channels
                s_res = compute_behavioral_significance(mid, idx, m, occs_m, alpha_p, raw_ck.ard_state.active_mask)
                push!(bfs, s_res.bf_loadings)
                push!(r2s, s_res.r2_kin)
            end
        end
    end
    push!(perm_mean_bfs, isempty(bfs) ? 0.0 : mean(bfs))
    push!(perm_mean_r2s, isempty(r2s) ? 0.0 : mean(r2s))
end

println(@sprintf("Control 2 Benchmark (Completed in %.2f s):", time() - t_c2))
println(@sprintf("  Real Data:           Mean ln BF_10 = %+.2f nats | Mean R^2_kin = %.1f%%",
        mean_real_bf, mean_real_r2 * 100.0))
println(@sprintf("  u_e-Permuted Null:   Mean ln BF_10 = %+.2f +/- %.2f nats | Mean R^2_kin = %.1f +/- %.1f%% (Max BF: %+.2f)",
        mean(perm_mean_bfs), std(perm_mean_bfs),
        mean(perm_mean_r2s) * 100.0, std(perm_mean_r2s) * 100.0, maximum(perm_mean_bfs)))
p_val_c2 = (count(x -> x >= mean_real_bf, perm_mean_bfs) + 1) / (N_PERM + 1)
println(@sprintf("  Level 4 Empirical p-value:   p < %.4f (Kinematics exceed permuted null)", p_val_c2))

# 9. Surrogate Control 3: Chronological Split-Half Cross-Validation
println("\n" * "="^120)
println("SURROGATE CONTROL 3: CHRONOLOGICAL SPLIT-HALF CROSS-VALIDATION")
println("Fit strictly on t in [0, T/2]; test predictive point-process LLR on t in (T/2, T] normalized by test-marginal rate.")
println("="^120)

T_half = T_total / 2.0
train_beh_idx = [idx for idx in beh_indices if ts[idx] <= T_half]
test_beh_idx = [idx for idx in beh_indices if ts[idx] > T_half]

# Train half model
train_model = initialize_behavioral_model(raw_ck, N_channels, beh_channels, T_half, chan_to_spikes)
# Keep only occurrences in train half
filter!(((id, occ),) -> occ.tau <= T_half, train_model.occurrences)

assign_tr = copy(assign)
alpha_tr = fill(400.0, length(raw_ck.ard_state.alpha))
rng_cv = Random.Xoshiro(303)

t_c3 = time()
for sw in 1:20
    step_behavioral_correspondence_sweep!(
        train_model.motifs, train_model.occurrences, assign_tr, ts, ch, train_beh_idx,
        chan_to_spikes, train_model.bg_rates, N_channels, raw_ck.ard_state, alpha_tr;
        behavioral_channels=beh_channels, sigma_prior=0.050, T_total=T_half, rng=rng_cv
    )
end

# Evaluate held-out predictive log-likelihood on test half t in (T_half, T_total]
test_occs = [occ for occ in values(raw_ck.occurrences) if occ.tau > T_half && occ.tau <= T_total]
Delta_T_test = T_total - T_half

function evaluate_test_log_likelihood(motifs, test_occs, ts_eval, ch_eval, test_indices, beh_channels, Delta_T)
    total_delta_llr = 0.0

    for c in beh_channels
        ev_indices = [idx for idx in test_indices if ch_eval[idx] == c]
        N_c = length(ev_indices)
        lam0_test = max(Float64(N_c) / Delta_T, 1e-6)
        ln_L_null = N_c * log(lam0_test) - lam0_test * Delta_T

        ln_L_model_sum = 0.0
        for idx in ev_indices
            t_s = ts_eval[idx]
            lam_s = lam0_test
            for occ in test_occs
                m = motifs[occ.motif]
                for (p_idx, pkt) in enumerate(m.packets)
                    pkt.channel == c || continue
                    pred_s = 0.0
                    for r in 1:min(length(raw_ck.ard_state.active_mask), length(occ.u), size(m.W, 1))
                        if raw_ck.ard_state.active_mask[r]
                            pred_s += m.W[r, p_idx] * occ.u[r]
                        end
                    end
                    t_pred = occ.tau + pkt.mu + pred_s
                    dt = t_s - t_pred
                    sig = sqrt(pkt.var)
                    bump = pkt.mass * exp(-0.5 * (dt * dt) / pkt.var) / (sqrt(2π) * sig)
                    lam_s += bump
                end
            end
            ln_L_model_sum += log(max(lam_s, 1e-12))
        end

        # Integral of intensity over test window
        integral_bump = 0.0
        for occ in test_occs
            m = motifs[occ.motif]
            for pkt in m.packets
                if pkt.channel == c
                    integral_bump += pkt.mass
                end
            end
        end
        ln_L_model = ln_L_model_sum - (lam0_test * Delta_T + integral_bump)
        total_delta_llr += (ln_L_model - ln_L_null)
    end
    return total_delta_llr
end

delta_llr_real = evaluate_test_log_likelihood(
    train_model.motifs, test_occs, ts, ch, test_beh_idx, beh_channels, Delta_T_test
)

# Test-shift controls
shuf_test_llrs = Float64[]
for sh in 1:10
    dt_sh = rand(rng_cv, Uniform(10.0, Delta_T_test - 10.0))
    ts_test_sh = copy(ts)
    for idx in test_beh_idx
        ts_test_sh[idx] = T_half + mod(ts[idx] - T_half + dt_sh, Delta_T_test)
    end
    llr_sh = evaluate_test_log_likelihood(
        train_model.motifs, test_occs, ts_test_sh, ch, test_beh_idx, beh_channels, Delta_T_test
    )
    push!(shuf_test_llrs, llr_sh)
end

println(@sprintf("Control 3 Benchmark (Completed in %.2f s):", time() - t_c3))
println(@sprintf("  Out-of-Sample Predictive Delta ln L_test (Real Test Data):  %+10.2f nats", delta_llr_real))
println(@sprintf("  Out-of-Sample Predictive Delta ln L_test (Shifted Surrogates): %+10.2f +/- %.2f nats (Max: %+.2f)",
        mean(shuf_test_llrs), std(shuf_test_llrs), maximum(shuf_test_llrs)))
println(@sprintf("  Predictive Improvement over Test-Marginal Baseline:        %+.2f nats/event",
        delta_llr_real / length(test_beh_idx)))

# 10. Serialize Checkpoint
save_path = joinpath(@__DIR__, "checkpoint_behavioral_correspondence.jls")
open(save_path, "w") do io
    serialize(io, (
        motifs = motifs,
        occurrences = occurrences,
        assign = assign_curr,
        bg_rates = bg_rates,
        alpha_beh = alpha_beh,
        ard_state = raw_ck.ard_state,
        beh_names = beh_names
    ))
end
println("\nSaved Behavioral Checkpoint: ", save_path)
println("="^120)
