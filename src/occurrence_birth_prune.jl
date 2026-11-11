# occurrence_birth_prune.jl - Step 0 Rao Score Occurrence Birth & Pruning

function calc_marginal_log_evidence(
    m::PurePacketMotif,
    counts::Vector{Int},
    members::Vector{Int},
    tau::Float64,
    u::Vector{Float64},
    spike_t::Vector{Float64},
    assign::Vector{SpikeAssignment},
    bg_rates::Vector{Float64},
    active_mask::BitVector;
    alpha_c::Float64=10.0,
    beta_c::Float64=10.0
)
    P = length(m.packets)
    P == 0 && return -Inf
    A_m = sum(p.mass for p in m.packets; init=0.0)
    N_e = sum(counts; init=0)

    # Base prior deficit when N_e == 0: alpha_c * ln(beta_c / (beta_c + A_m))
    prior_deficit = alpha_c * log(beta_c / (beta_c + A_m))
    N_e == 0 && return prior_deficit

    # Unpack active deformation coordinates
    k1 = 0; k2 = 0; k3 = 0; k4 = 0
    K = 0
    for r in 1:length(active_mask)
        if active_mask[r]
            K += 1
            if K == 1; k1 = r
            elseif K == 2; k2 = r
            elseif K == 3; k3 = r
            elseif K == 4; k4 = r
            end
        end
    end

    # Kernel likelihood ratio sum: sum_{p} sum_{i in p} ln( (a_{m,p} * N_i) / lambda_0 )
    kernel_sum = 0.0
    for spk_idx in members
        p = assign[spk_idx].packet_idx
        (p <= 0 || p > P) && continue
        pkt = m.packets[p]
        sig2 = pkt.var
        sig = sqrt(sig2)
        lam0 = max(bg_rates[pkt.channel], 1e-12)

        pred_s = 0.0
        if K == 1; pred_s = m.W[k1, p] * u[k1]
        elseif K == 2; pred_s = m.W[k1, p] * u[k1] + m.W[k2, p] * u[k2]
        elseif K == 3; pred_s = m.W[k1, p] * u[k1] + m.W[k2, p] * u[k2] + m.W[k3, p] * u[k3]
        elseif K == 4; pred_s = m.W[k1, p] * u[k1] + m.W[k2, p] * u[k2] + m.W[k3, p] * u[k3] + m.W[k4, p] * u[k4]
        end

        dt = spike_t[spk_idx] - (tau + pkt.mu + pred_s)
        N_i = exp(-0.5 * (dt * dt) / sig2) / (sqrt(2π) * sig)
        ratio = (pkt.mass * N_i) / lam0
        kernel_sum += log(max(ratio, 1e-12))
    end

    # Negative Binomial count terms
    gamma_ratio = loggamma(alpha_c + Float64(N_e)) - loggamma(alpha_c)
    scale_term = - Float64(N_e) * log(beta_c + A_m)

    return kernel_sum + gamma_ratio + scale_term + prior_deficit
end

"""
    prune_dead_occurrences!(
        occurrences::Dict{Int, PureOccurrence},
        motifs::Dict{Int, PurePacketMotif},
        spike_t::Vector{Float64},
        assign::Vector{SpikeAssignment},
        bg_rates::Vector{Float64},
        T_max::Float64,
        active_mask::BitVector;
        alpha_c::Float64=10.0,
        beta_c::Float64=10.0,
        gamma_m::Float64=100.0
    )

Prunes occurrences that fail the exact Marginal Negative Binomial Bayes Factor test:
    Delta ln M(e) <= Tax_Occam
where Tax_Occam = ln((T - W) / W) + ln((E_m + 1) / gamma_m).
Releases member spikes to background (assign[i] = SpikeAssignment(0, 0)) and deletes the occurrence.
"""
function prune_dead_occurrences!(
    occurrences::Dict{Int, PureOccurrence},
    motifs::Dict{Int, PurePacketMotif},
    spike_t::Vector{Float64},
    assign::Vector{SpikeAssignment},
    bg_rates::Vector{Float64},
    T_max::Float64,
    active_mask::BitVector;
    alpha_c::Float64=10.0,
    beta_c::Float64=10.0,
    gamma_m::Float64=100.0
)
    pruned_eids = Int[]
    motif_occ_counts = Dict{Int, Int}()
    for occ in values(occurrences)
        motif_occ_counts[occ.motif] = get(motif_occ_counts, occ.motif, 0) + 1
    end

    for (eid, occ) in occurrences
        m = get(motifs, occ.motif, nothing)
        if m === nothing
            push!(pruned_eids, eid)
            continue
        end

        W_eff = max(m.delta_right - m.delta_left, 0.05)
        E_m = get(motif_occ_counts, occ.motif, 1)
        tax_occam = log(max((T_max - W_eff) / W_eff, 1.0)) + log(max((E_m + 1.0) / gamma_m, 1e-3))

        N_e = sum(occ.counts; init=0)
        if N_e == 0
            push!(pruned_eids, eid)
            continue
        end

        # Exact marginal Negative Binomial log-evidence
        delta_ln_M = calc_marginal_log_evidence(
            m, occ.counts, occ.members, occ.tau, occ.u,
            spike_t, assign, bg_rates, active_mask;
            alpha_c=alpha_c, beta_c=beta_c
        )

        # Death condition: exactly symmetric with birth condition
        if delta_ln_M <= tax_occam
            push!(pruned_eids, eid)
        end
    end

    # Release spikes of dead occurrences back to background
    for eid in pruned_eids
        occ = occurrences[eid]
        for spk_idx in occ.members
            assign[spk_idx] = SpikeAssignment(0, 0)
        end
        delete!(occurrences, eid)
    end

    return pruned_eids
end

# Fallback for simple tests
function prune_dead_occurrences!(
    occurrences::Dict{Int, PureOccurrence},
    motifs::Dict{Int, PurePacketMotif},
    assign::Vector{SpikeAssignment},
    bg_rates::Vector{Float64};
    min_spikes::Int=2,
    min_amplitude::Float64=0.25,
    min_evidence::Float64=0.0
)
    pruned_eids = Int[]
    for (eid, occ) in occurrences
        n_spk = sum(occ.counts; init=0)
        if n_spk < min_spikes || occ.amplitude < min_amplitude
            push!(pruned_eids, eid)
        end
    end
    for eid in pruned_eids
        occ = occurrences[eid]
        for spk_idx in occ.members
            assign[spk_idx] = SpikeAssignment(0, 0)
        end
        delete!(occurrences, eid)
    end
    return pruned_eids
end


# =====================================================================
# 8. NONPARAMETRIC OCCURRENCE BIRTH & COMPETITIVE OVERWRITE
# =====================================================================

"""
Continuous Data-Driven Reversible Jump MCMC Occurrence Birth Sampler:
Derived from the continuous log-evidence functional (Rao score field) of the marked point process:
  S_m(tau) = sum_{p in P_m} sum_{i in D_bg(k_p)} w_{i,p} N(tau | t_i - mu_p, sigma_p^2) - Tax_Occam
where w_{i,p} = a_{m,p} / lambda_{0, k_p}.

1. Normalized component weights:
   W_{m,p} = (a_{m,p} / lambda_{0, k_p}) * N_bg(k_p),  W_m = sum_p W_{m,p}
2. O(1) Candidate Onset Proposal:
   Draw packet p* ~ Categorical(W_{m,:}), background spike i* ~ Uniform(D_bg(k_{p*})),
   continuous candidate onset seed tau_seed ~ N(t_{i*} - mu_{p*}, sigma_{p*}^2).
   Local Newton mode centering: tau* = tau_seed - score / hessian.
3. Exact Denominator Evaluation q(tau* | D_bg):
   Evaluated via local binary search over sorted background spikes on motif channels within 4.0 * sigma.
4. Metropolis-Hastings Detailed Balance Acceptance:
   ln R = Delta ln M*(tau*) - Tax_Occam - ln(W_window * q(tau* | D_bg))
   Accept if log(rand(rng)) < min(0.0, ln_R).
5. Competitive Overwrite & Refractory Exclusion.
"""
function detect_and_birth_occurrences!(
    occurrences::Dict{Int, PureOccurrence},
    motifs::Dict{Int, PurePacketMotif},
    spike_t::Vector{Float64},
    spike_chan::Vector{Int},
    chan_to_spikes::Vector{Vector{Int}},
    assign::Vector{SpikeAssignment},
    bg_rates::Vector{Float64},
    T_max::Float64,
    R_max::Int;
    alpha_c::Float64=10.0,
    beta_c::Float64=10.0,
    gamma_m::Float64=100.0,
    delta_margin::Float64=3.0,
    refractory_fraction::Float64=0.5,
    delta_win::Float64=0.030,
    alpha_stopping::Float64=0.025,
    K_prop::Int=50,
    bg_times::Union{Nothing, Vector{Vector{Float64}}}=nothing,
    bg_indices::Union{Nothing, Vector{Vector{Int}}}=nothing,
    active_mask::Union{Nothing, BitVector}=nothing,
    rng::AbstractRNG=Random.default_rng()
)
    next_eid = isempty(occurrences) ? 1 : (maximum(keys(occurrences)) + 1)
    intra_sweep_exclusion = Tuple{Int, Float64}[]
    n_birthed = 0
    n_overwritten = 0
    act_mask = active_mask !== nothing ? active_mask : trues(R_max)

    N_channels = length(chan_to_spikes)

    # Pre-gather background spikes per channel
    times_bg = bg_times !== nothing ? bg_times : [Float64[] for _ in 1:N_channels]
    indices_bg = bg_indices !== nothing ? bg_indices : [Int[] for _ in 1:N_channels]

    for c in 1:N_channels
        empty!(times_bg[c])
        empty!(indices_bg[c])
        spks = chan_to_spikes[c]
        sizehint!(times_bg[c], length(spks))
        sizehint!(indices_bg[c], length(spks))
        @inbounds for s in spks
            if assign[s].event_id == 0
                push!(times_bg[c], spike_t[s])
                push!(indices_bg[c], s)
            end
        end
    end

    # Preallocate candidate scratch buffers
    cand_spk_indices = Int[]
    cand_pkt_indices = Int[]
    cand_lr = Float64[]
    sizehint!(cand_spk_indices, 256)
    sizehint!(cand_pkt_indices, 256)
    sizehint!(cand_lr, 256)

    for (mid, m) in motifs
        P = length(m.packets)
        P == 0 && continue
        W_eff = max(m.delta_right - m.delta_left, 0.05)
        total_mass = sum(p.mass for p in m.packets; init=0.0)
        E_m = count(occ -> occ.motif == mid, values(occurrences))
        sig_min = minimum(sqrt(p.var) for p in m.packets)

        # Prior Deficit and Occam Volume Tax evaluated with dynamic physical span W_eff
        prior_deficit = alpha_c * log(beta_c / (beta_c + total_mass))
        tax_occam = log(max((T_max - W_eff) / W_eff, 1.0)) + log(max((E_m + 1.0) / gamma_m, 1e-3))
        N_min = max(ceil(Int, (abs(prior_deficit) + tax_occam) / 3.5), 2)

        # 1. Component Weights for Continuous Mixture Proposal
        W_mp = zeros(Float64, P)
        sum_snr = 0.0
        for p in 1:P
            pkt = m.packets[p]
            k = pkt.channel
            lam0 = max(bg_rates[k], 1e-12)
            N_bg_k = length(times_bg[k])
            W_mp[p] = (pkt.mass / lam0) * N_bg_k
            sum_snr += (pkt.mass * pkt.mass) / lam0
        end
        W_m = sum(W_mp)
        (W_m <= 0.0 || sum_snr <= 0.0) && continue

        cum_W = cumsum(W_mp)

        # Dynamic Physical Stopping Bound L_m* based on single-occurrence sensitivity (alpha = 0.025 for >=95% confidence)
        p_hit_1 = sum_snr / W_m
        L_m_star = max(ceil(Int, log(1.0 / alpha_stopping) / p_hit_1), 100)
        refract_dist = refractory_fraction * W_eff
        tau_min = max(0.0, -m.delta_left)
        tau_max = max(tau_min, T_max - m.delta_right)

        c_reject = 0

        while c_reject < L_m_star
            # 2. O(1) Candidate Onset Proposal
            u_draw = rand(rng) * W_m
            p_star = searchsortedfirst(cum_W, u_draw)
            p_star = clamp(p_star, 1, P)
            pkt_star = m.packets[p_star]
            k_star = pkt_star.channel
            N_bg_star = length(times_bg[k_star])
            if N_bg_star == 0
                c_reject += 1
                continue
            end

            i_star = rand(rng, 1:N_bg_star)
            t_spk = times_bg[k_star][i_star]
            sig_p_star = sqrt(pkt_star.var)
            tau_seed = t_spk - pkt_star.mu + sig_p_star * randn(rng)

            if (tau_seed < tau_min || tau_seed > tau_max)
                c_reject += 1
                continue
            end

            # Local Newton mode centering step
            score = 0.0
            hessian = 0.0
            for p in 1:P
                pkt = m.packets[p]
                c = pkt.channel
                spk_times_c = times_bg[c]
                isempty(spk_times_c) && continue
                sig2 = pkt.var
                sig = sqrt(sig2)
                lam0 = max(bg_rates[c], 1e-12)
                w_p = pkt.mass / lam0
                t_pred = tau_seed + pkt.mu
                j_lo = searchsortedfirst(spk_times_c, t_pred - 3.5 * sig)
                j_hi = searchsortedlast(spk_times_c, t_pred + 3.5 * sig)
                j_lo > j_hi && continue
                inv_2sig2 = 0.5 / sig2
                norm_const = 1.0 / (sqrt(2pi) * sig)
                for j in j_lo:j_hi
                    dt = spk_times_c[j] - t_pred
                    N_j = exp(-dt * dt * inv_2sig2) * norm_const
                    ratio = w_p * N_j
                    p_j = ratio / (1.0 + ratio)
                    score += p_j * (dt / sig2)
                    hessian -= p_j / sig2
                end
            end

            tau_star = tau_seed
            if abs(hessian) > 1e-9
                delta_tau = clamp(-score / hessian, -2.0 * sig_min, 2.0 * sig_min)
                tau_star = clamp(tau_seed + delta_tau, tau_min, tau_max)
            end

            # Intra-sweep exclusion check
            is_excluded = false
            for (ex_m, ex_tau) in intra_sweep_exclusion
                if ex_m == mid && abs(tau_star - ex_tau) < refract_dist
                    is_excluded = true
                    break
                end
            end
            if is_excluded
                c_reject += 1
                continue
            end

            # 3. Denominator Evaluation q(tau* | D_bg) & Candidate Spikes
            q_sum = 0.0
            empty!(cand_spk_indices)
            empty!(cand_pkt_indices)
            empty!(cand_lr)

            for p in 1:P
                pkt = m.packets[p]
                c = pkt.channel
                spk_times_c = times_bg[c]
                isempty(spk_times_c) && continue

                sig2 = pkt.var
                sig = sqrt(sig2)
                lam0 = max(bg_rates[c], 1e-12)
                w_p = pkt.mass / lam0

                t_pred = tau_star + pkt.mu
                t_lo = t_pred - 4.0 * sig
                t_hi = t_pred + 4.0 * sig

                j_lo = searchsortedfirst(spk_times_c, t_lo)
                j_hi = searchsortedlast(spk_times_c, t_hi)
                j_lo > j_hi && continue

                inv_2sig2 = 0.5 / sig2
                norm_const = 1.0 / (sqrt(2pi) * sig)

                for j in j_lo:j_hi
                    t_j = spk_times_c[j]
                    dt = t_j - t_pred
                    N_j = exp(-dt * dt * inv_2sig2) * norm_const
                    q_sum += w_p * N_j

                    if abs(dt) <= 3.0 * sig
                        ratio = w_p * N_j
                        if ratio > 1.0
                            s_orig = indices_bg[c][j]
                            lr = log(ratio)
                            pos = findfirst(==(s_orig), cand_spk_indices)
                            if pos === nothing
                                push!(cand_spk_indices, s_orig)
                                push!(cand_pkt_indices, p)
                                push!(cand_lr, lr)
                            elseif lr > cand_lr[pos]
                                cand_pkt_indices[pos] = p
                                cand_lr[pos] = lr
                            end
                        end
                    end
                end
            end

            N_cand = length(cand_spk_indices)
            if N_cand < N_min
                c_reject += 1
                continue
            end

            # 4. Latent deformation solve u* = A^{-1} b
            u_star_vec = zeros(Float64, R_max)
            k1 = 0; k2 = 0; k3 = 0; k4 = 0
            K = 0
            for r in 1:min(R_max, length(act_mask))
                if act_mask[r]
                    K += 1
                    if K == 1; k1 = r
                    elseif K == 2; k2 = r
                    elseif K == 3; k3 = r
                    elseif K == 4; k4 = r
                    end
                end
            end

            kernel_sum = sum(cand_lr)
            if K > 0 && N_cand >= 2
                A_mat = zeros(MMatrix{4, 4, Float64})
                b_vec = zeros(MVector{4, Float64})
                for r in 1:K
                    A_mat[r, r] = 1.0
                end
                for k_idx in 1:N_cand
                    s = cand_spk_indices[k_idx]
                    p = cand_pkt_indices[k_idx]
                    pkt = m.packets[p]
                    kap = 1.0 / pkt.var
                    y = spike_t[s] - (tau_star + pkt.mu)
                    for r1 in 1:K
                        r1_idx = (r1 == 1) ? k1 : ((r1 == 2) ? k2 : ((r1 == 3) ? k3 : k4))
                        wr1 = m.W[r1_idx, p]
                        b_vec[r1] += kap * wr1 * y
                        for r2 in 1:K
                            r2_idx = (r2 == 1) ? k1 : ((r2 == 2) ? k2 : ((r2 == 3) ? k3 : k4))
                            A_mat[r1, r2] += kap * wr1 * m.W[r2_idx, p]
                        end
                    end
                end
                A_sub = SMatrix{K, K, Float64}(view(A_mat, 1:K, 1:K))
                b_sub = SVector{K, Float64}(view(b_vec, 1:K))
                u_sub = inv(A_sub) * b_sub

                # Recompute kernel sum with u*
                kernel_sum_warped = 0.0
                for k_idx in 1:N_cand
                    s = cand_spk_indices[k_idx]
                    p = cand_pkt_indices[k_idx]
                    c = spike_chan[s]
                    pkt = m.packets[p]
                    sig2 = pkt.var
                    sig = sqrt(sig2)
                    lam0 = max(bg_rates[c], 1e-12)
                    pred_s = 0.0
                    for r in 1:K
                        r_idx = (r == 1) ? k1 : ((r == 2) ? k2 : ((r == 3) ? k3 : k4))
                        pred_s += m.W[r_idx, p] * u_sub[r]
                    end
                    dt_warp = spike_t[s] - (tau_star + pkt.mu + pred_s)
                    N_j = exp(-0.5 * (dt_warp * dt_warp) / sig2) / (sqrt(2pi) * sig)
                    ratio = (pkt.mass * N_j) / lam0
                    kernel_sum_warped += log(max(ratio, 1e-12))
                end
                if kernel_sum_warped > kernel_sum
                    kernel_sum = kernel_sum_warped
                    for r in 1:K
                        r_idx = (r == 1) ? k1 : ((r == 2) ? k2 : ((r == 3) ? k3 : k4))
                        u_star_vec[r_idx] = u_sub[r]
                    end
                end
            end

            # 5. Marginal Negative Binomial evidence & Hastings acceptance
            gamma_ratio = loggamma(alpha_c + Float64(N_cand)) - loggamma(alpha_c)
            scale_term = - Float64(N_cand) * log(beta_c + total_mass)
            delta_ln_M_star = kernel_sum + gamma_ratio + scale_term + prior_deficit

            q_tau = max(q_sum / W_m, 1e-15)
            ln_R = delta_ln_M_star - tax_occam - log(W_eff * q_tau)

            # 6. Competitive Overwrite or Fresh Birth
            incumbent_eid = 0
            min_dist = Inf
            for (eid, occ) in occurrences
                if occ.motif == mid
                    dist = abs(occ.tau - tau_star)
                    if dist < refract_dist && dist < min_dist
                        min_dist = dist
                        incumbent_eid = eid
                    end
                end
            end

            if incumbent_eid == 0
                if log(rand(rng)) < min(0.0, ln_R)
                    new_occ = PureOccurrence(next_eid, mid, tau_star, R_max, P; amplitude=1.0)
                    next_eid += 1
                    for r in 1:R_max
                        new_occ.u[r] = u_star_vec[r]
                    end
                    for k_idx in 1:N_cand
                        j = cand_spk_indices[k_idx]
                        p = cand_pkt_indices[k_idx]
                        assign[j] = SpikeAssignment(new_occ.id, p)
                        push!(new_occ.members, j)
                        new_occ.counts[p] += 1
                        new_occ.sum_t[p] += spike_t[j]
                        new_occ.sum_t2[p] += spike_t[j] * spike_t[j]
                    end
                    new_occ.amplitude = rand(rng, Gamma(alpha_c + Float64(N_cand), 1.0 / (beta_c + total_mass)))
                    occurrences[new_occ.id] = new_occ
                    push!(intra_sweep_exclusion, (mid, tau_star))
                    n_birthed += 1
                    E_m += 1
                    tax_occam = log(max((T_max - W_eff) / W_eff, 1.0)) + log(max((E_m + 1.0) / gamma_m, 1e-3))

                    # ACCEPTED: Reset rejection counter
                    c_reject = 0

                    # Sequential depletion: excise claimed spikes from background pool
                    for k_idx in 1:N_cand
                        j_spk = cand_spk_indices[k_idx]
                        c_ch = spike_chan[j_spk]
                        pos = findfirst(==(j_spk), indices_bg[c_ch])
                        if pos !== nothing
                            deleteat!(times_bg[c_ch], pos)
                            deleteat!(indices_bg[c_ch], pos)
                        end
                    end
                    # In-place update of W_mp and cum_W
                    for p in 1:P
                        pkt = m.packets[p]
                        k = pkt.channel
                        lam0 = max(bg_rates[k], 1e-12)
                        N_bg_k = length(times_bg[k])
                        W_mp[p] = (pkt.mass / lam0) * N_bg_k
                    end
                    W_m = sum(W_mp)
                    W_m <= 0.0 && break
                    cum_W = cumsum(W_mp)
                else
                    c_reject += 1
                end
            else
                incumbent = occurrences[incumbent_eid]
                ev_incumbent = calc_marginal_log_evidence(
                    m, incumbent.counts, incumbent.members, incumbent.tau, incumbent.u,
                    spike_t, assign, bg_rates, act_mask;
                    alpha_c=alpha_c, beta_c=beta_c
                )
                if delta_ln_M_star > ev_incumbent + delta_margin
                    for j in incumbent.members
                        assign[j] = SpikeAssignment(0, 0)
                    end
                    empty!(incumbent.members)
                    fill!(incumbent.counts, 0)
                    fill!(incumbent.sum_t, 0.0)
                    fill!(incumbent.sum_t2, 0.0)
                    for r in 1:R_max
                        incumbent.u[r] = u_star_vec[r]
                    end
                    incumbent.tau = tau_star
                    for k_idx in 1:N_cand
                        j = cand_spk_indices[k_idx]
                        p = cand_pkt_indices[k_idx]
                        assign[j] = SpikeAssignment(incumbent.id, p)
                        push!(incumbent.members, j)
                        incumbent.counts[p] += 1
                        incumbent.sum_t[p] += spike_t[j]
                        incumbent.sum_t2[p] += spike_t[j] * spike_t[j]
                    end
                    incumbent.amplitude = rand(rng, Gamma(alpha_c + Float64(N_cand), 1.0 / (beta_c + total_mass)))
                    push!(intra_sweep_exclusion, (mid, tau_star))
                    n_overwritten += 1

                    # ACCEPTED: Reset rejection counter
                    c_reject = 0

                    # Sequential depletion: excise newly claimed spikes
                    for k_idx in 1:N_cand
                        j_spk = cand_spk_indices[k_idx]
                        c_ch = spike_chan[j_spk]
                        pos = findfirst(==(j_spk), indices_bg[c_ch])
                        if pos !== nothing
                            deleteat!(times_bg[c_ch], pos)
                            deleteat!(indices_bg[c_ch], pos)
                        end
                    end
                    for p in 1:P
                        pkt = m.packets[p]
                        k = pkt.channel
                        lam0 = max(bg_rates[k], 1e-12)
                        N_bg_k = length(times_bg[k])
                        W_mp[p] = (pkt.mass / lam0) * N_bg_k
                    end
                    W_m = sum(W_mp)
                    W_m <= 0.0 && break
                    cum_W = cumsum(W_mp)
                else
                    c_reject += 1
                end
            end
        end
    end

    return (n_birthed, n_overwritten)
end

# =====================================================================
# 8. GLOBAL ARD PRECISIONS, PRUNING & DYNAMIC BIRTH (Level m=4)
# =====================================================================

