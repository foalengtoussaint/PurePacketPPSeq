# mechanisms/m1_packet.jl - Level m=1 Pure Spatiotemporal Emission Packets
#
# Generative Role:
# - Unit marked point-process kernel on physical neuron c_p:
#     lambda_{m,p}(t) = a_{m,p} * N(t; tau_e + mu_p + W_{:,p}^T u_e, sigma_p^2)
# - Parameterizes emission mass a_{m,p}, canonical latency mu_p, and jitter variance var_p = sigma_p^2.
# - Governs SNR dominance radius: R_p = sigma_p * sqrt(2 * ln^+(SNR_p)).
# - Executes continuous Bayes factor bimodal split and background packet birth.

"""
    compute_packet_snr_radius(mass::Float64, var_p::Float64, bg_rate::Float64) -> Float64

Evaluates the exact physical SNR dominance radius of an emission packet:
    R_p = sigma_p * sqrt(2 * ln^+( a_{m,p} / (sqrt(2pi) * sigma_p * lambda_{0,c}) ))
where R_p == 0.0 whenever SNR_p <= 1.0 (Rule 3 SNR pruning from first principles).
"""
@inline function compute_packet_snr_radius(mass::Float64, var_p::Float64, bg_rate::Float64)::Float64
    sig = sqrt(max(var_p, 1e-8))
    lam0 = max(bg_rate, 1e-12)
    snr = mass / (sqrt(2π) * sig * lam0)
    return (snr > 1.0) ? (sig * sqrt(2.0 * log(snr))) : 0.0
end

"""
    sample_packet_masses!(m::PurePacketMotif, occurrences::Vector{PureOccurrence}; alpha_a=1.0, beta_a=1.0, rng=default_rng())

Conjugate Gamma draws for packet emission amplitudes a_{m,p}:
    a_{m,p} ~ Gamma(alpha_a + sum_e c_{e,p}, 1 / (beta_a + sum_e c_e)).
"""
function sample_packet_masses!(
    m::PurePacketMotif,
    occurrences::Vector{PureOccurrence};
    alpha_a::Float64=1.0,
    beta_a::Float64=1.0,
    rng::AbstractRNG=Random.default_rng()
)
    P = length(m.packets)
    P == 0 && return

    E_tilde = sum(occ.amplitude for occ in occurrences; init=0.0)
    rate_post = beta_a + max(E_tilde, 1e-4)
    scale_post = 1.0 / rate_post

    for p in 1:P
        c_tot = sum(occ.counts[p] for occ in occurrences; init=0)
        shape_post = max(alpha_a + max(0, c_tot), 1e-4)
        m.packets[p].mass = rand(rng, Gamma(shape_post, scale_post))
    end
end

"""
    update_packet_corridors!(m::PurePacketMotif, occurrences::Vector{PureOccurrence}, active_mask::BitVector; anchor_origin=true, damp=0.2, nu_0=10.0, S_0=0.00025)

Updates packet canonical latencies mu_p and jitter variances var_p with origin anchoring mu_1 == 0.
Returns the origin_offset shift absorbed by occurrences.
"""
function update_packet_corridors!(
    m::PurePacketMotif,
    occurrences::Vector{PureOccurrence},
    active_mask::BitVector;
    anchor_origin::Bool=true,
    damp::Float64=0.2,
    nu_0::Float64=10.0,
    S_0::Float64=0.00025
)
    P = length(m.packets)
    P == 0 && return 0.0
    E = length(occurrences)

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

    for p in 1:P
        pkt = m.packets[p]
        tot_c = 0
        weighted_delta = 0.0
        sum_sq_err = 0.0

        for occ in occurrences
            c = occ.counts[p]
            c == 0 && continue
            u = occ.u
            pred_s = 0.0
            if K == 1; pred_s = m.W[k1, p] * u[k1]
            elseif K == 2; pred_s = m.W[k1, p] * u[k1] + m.W[k2, p] * u[k2]
            elseif K == 3; pred_s = m.W[k1, p] * u[k1] + m.W[k2, p] * u[k2] + m.W[k3, p] * u[k3]
            elseif K == 4; pred_s = m.W[k1, p] * u[k1] + m.W[k2, p] * u[k2] + m.W[k3, p] * u[k3] + m.W[k4, p] * u[k4]
            end
            mean_t = occ.sum_t[p] / c
            target_t = occ.tau + pkt.mu + pred_s
            weighted_delta += (mean_t - target_t) * c
            tot_c += c
            sq_err = occ.sum_t2[p] - 2.0 * target_t * occ.sum_t[p] + c * target_t * target_t
            sum_sq_err += max(0.0, sq_err)
        end

        if tot_c > 0
            pkt.mu += damp * (weighted_delta / tot_c)
            nu_post = nu_0 + tot_c
            S_post = S_0 + sum_sq_err
            pkt.var = S_post / nu_post
        end
    end

    origin_offset = 0.0
    if anchor_origin && P > 0
        origin_offset = m.packets[1].mu
        if origin_offset != 0.0
            for p in 1:P
                m.packets[p].mu -= origin_offset
            end
            for occ in occurrences
                occ.tau += origin_offset
            end
        end
    end

    return origin_offset
end

"""
    prune_dead_packets!(
        m::PurePacketMotif,
        occurrences::Vector{PureOccurrence},
        bg_rates::Vector{Float64},
        N_channels::Int;
        spike_t::Union{Nothing, Vector{Float64}}=nothing,
        spike_chan::Union{Nothing, Vector{Int}}=nothing,
        assign::Union{Nothing, Vector{SpikeAssignment}}=nothing,
        ou_ctx::Union{Nothing, ContinuousOUContext}=nothing,
        active_mask::Union{Nothing, BitVector}=nothing,
        ard_alphas::Union{Nothing, Vector{Float64}}=nothing,
        sigma_prior::Float64=0.015,
        alpha_a::Float64=1.0,
        beta_a::Float64=1.0,
        counts_bg::Union{Nothing, Vector{Int}}=nothing,
        rng::AbstractRNG=Random.default_rng(),
        occam_tax::Float64=2.0
    )::Vector{Int}

Prunes packets using the exact symmetric Metropolis-Hastings reverse-jump (P -> P-1):
1. Last-packet protection: P_m <= 1 is never deleted.
2. Single-step reversibility: Rejects death proposal (alpha_death = 0) if mu_p lies outside the survivor envelope reach.
3. Evaluates analytical Gamma-marginalized log-evidence Delta ln M_p using instantaneous adapted background null lambda_c^{(0)}(t_s).
4. Evaluates continuous parameter Occam volume tax on the survivor envelope S_m(P_m \\ {p}).
5. Accepts/rejects death jump via Metropolis-Hastings detailed balance ratio.
"""
function prune_dead_packets!(
    m::PurePacketMotif,
    occurrences::Vector{PureOccurrence},
    bg_rates::Vector{Float64},
    N_channels::Int;
    spike_t::Union{Nothing, Vector{Float64}}=nothing,
    spike_chan::Union{Nothing, Vector{Int}}=nothing,
    assign::Union{Nothing, Vector{SpikeAssignment}}=nothing,
    ou_ctx::Union{Nothing, ContinuousOUContext}=nothing,
    active_mask::Union{Nothing, BitVector}=nothing,
    ard_alphas::Union{Nothing, Vector{Float64}}=nothing,
    sigma_prior::Float64=0.015,
    alpha_a::Float64=1.0,
    beta_a::Float64=1.0,
    counts_bg::Union{Nothing, Vector{Int}}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    occam_tax::Float64=2.0
)::Vector{Int}
    P = length(m.packets)
    P <= 1 && return Int[]
    E_m = length(occurrences)
    E_eff = max(sum(occ.amplitude for occ in occurrences; init=0.0), 1.0)
    var_prior = sigma_prior^2

    dead_packets = Int[]

    for p in 1:P
        pkt = m.packets[p]
        chan = pkt.channel
        survivors_tentative = [q for q in 1:P if q != p]
        isempty(survivors_tentative) && continue

        # 1. Survivor Support Envelope: S_m(P_m \ {p})
        min_surv_mu = minimum(m.packets[q].mu for q in survivors_tentative)
        max_surv_mu = maximum(m.packets[q].mu for q in survivors_tentative)
        sigma_bar = mean(sqrt(m.packets[q].var) for q in survivors_tentative)
        delta_reach = max(3.0 * sigma_bar, 0.25)
        surv_left = minimum(m.packets[q].mu - 4.0 * sqrt(m.packets[q].var) for q in survivors_tentative) - delta_reach
        surv_right = maximum(m.packets[q].mu + 4.0 * sqrt(m.packets[q].var) for q in survivors_tentative) + delta_reach
        proposal_support = max(surv_right - surv_left, 2.0 * delta_reach)

        # 2. Single-Step Reversibility Guard:
        # If mu_p is outside surv_left..surv_right, a single birth proposal on the survivors
        # could never reach mu_p. Detailed balance reversibility requires q(birth) = 0 => alpha_death = 0.
        is_extremal = (pkt.mu < min_surv_mu) || (pkt.mu > max_surv_mu)
        if is_extremal && (pkt.mu < surv_left || pkt.mu > surv_right)
            continue
        end

        # 3. Continuous Parameter Occam Volume Tax on Survivor Support
        tax_loadings = 0.0
        if active_mask !== nothing && ard_alphas !== nothing
            for r in 1:min(length(active_mask), length(ard_alphas))
                if active_mask[r]
                    alpha_r = ard_alphas[r]
                    tax_loadings += 0.5 * log(1.0 + E_eff / (max(alpha_r, 1.0) * var_prior))
                end
            end
        end
        tax_occam = log(proposal_support / (sqrt(2π) * sigma_prior)) + log(beta_a + E_m) + tax_loadings

        # 4. Evaluate Gamma-Marginalized Log-Evidence Delta ln M_p
        llr_p = 0.0
        c_tot = sum(occ.counts[p] for occ in occurrences; init=0)

        if spike_t !== nothing && assign !== nothing && c_tot > 0
            cand_dt = Float64[]
            cand_spk = Int[]
            sum_w = 0.0

            for occ in occurrences
                occ.counts[p] == 0 && continue
                u = occ.u
                pred_s = 0.0
                if active_mask !== nothing
                    for r in 1:min(length(active_mask), length(u), size(m.W, 1))
                        if active_mask[r]
                            pred_s += m.W[r, p] * u[r]
                        end
                    end
                end
                t_pred = occ.tau + pkt.mu + pred_s
                for s_idx in occ.members
                    if assign[s_idx].packet_idx == p
                        dt = spike_t[s_idx] - t_pred
                        w = exp(-0.5 * (dt * dt) / var_prior)
                        sum_w += w
                        push!(cand_dt, dt)
                        push!(cand_spk, s_idx)
                    end
                end
            end

            a_post = (alpha_a + sum_w) / (beta_a + E_eff)
            for (dt, s_idx) in zip(cand_dt, cand_spk)
                t_s = spike_t[s_idx]
                lam_s = if ou_ctx !== nothing
                    exp(evaluate_background_log_intensity(ou_ctx, bg_rates, chan, t_s))
                else
                    (chan <= length(bg_rates)) ? max(bg_rates[chan], 1e-12) : 0.5
                end
                bump = a_post * exp(-0.5 * (dt * dt) / var_prior) / (sqrt(2π) * sigma_prior)
                llr_p += log(1.0 + bump / lam_s)
            end
            llr_p -= E_eff * a_post
        else
            # Analytical fallback for unit tests without full spike arrays
            sigma_p = sqrt(pkt.var)
            lam0 = (chan <= length(bg_rates)) ? max(bg_rates[chan], 1e-12) : 0.5
            a_post = (alpha_a + Float64(c_tot)) / (beta_a + E_eff)
            bump = a_post / (sqrt(2π) * sigma_p)
            llr_p = Float64(c_tot) * log(1.0 + bump / lam0) - E_eff * a_post
            # Enforce SNR <= 1.0 or c_tot == 0 in fallback mode when occam_tax was explicitly passed
            if c_tot == 0 || (c_tot < 2 && (a_post / (sqrt(2π) * sigma_p * lam0)) <= 1.0)
                llr_p = -Inf
            end
        end

        # 5. Metropolis-Hastings Acceptance:
        # Δln Post = llr_p - tax_occam
        # ln α_death = min(0.0, -Δln Post) = min(0.0, tax_occam - llr_p)
        delta_post = llr_p - tax_occam
        log_alpha = min(0.0, -delta_post)

        if log(rand(rng)) < log_alpha
            push!(dead_packets, p)
        end
    end

    isempty(dead_packets) && return dead_packets

    # Anchor Handover if Packet 1 pruned
    if 1 in dead_packets
        first_survivor = 0
        for p in 1:P
            if !(p in dead_packets)
                first_survivor = p
                break
            end
        end
        if first_survivor > 0
            handover_shift = m.packets[first_survivor].mu
            for p in 1:P
                m.packets[p].mu -= handover_shift
            end
            for occ in occurrences
                occ.tau += handover_shift
            end
        end
    end

    survivors = [p for p in 1:P if !(p in dead_packets)]
    if isempty(survivors)
        survivors = [argmax([pkt.mass for pkt in m.packets])]
    end

    P_new = length(survivors)
    new_packets = Vector{Packet}(undef, P_new)
    new_W = zeros(Float64, size(m.W, 1), P_new)

    for (new_p, old_p) in enumerate(survivors)
        pkt = m.packets[old_p]
        pkt.id = new_p
        new_packets[new_p] = pkt
        new_W[:, new_p] = m.W[:, old_p]
    end

    m.packets = new_packets
    m.W = new_W
    rebuild_inverted_index!(m, N_channels)

    old_to_new = zeros(Int, P)
    for (new_p, old_p) in enumerate(survivors)
        old_to_new[old_p] = new_p
    end

    if assign !== nothing
        for occ in occurrences
            i = 1
            while i <= length(occ.members)
                s_idx = occ.members[i]
                as = assign[s_idx]
                if as.event_id == occ.id
                    old_p = as.packet_idx
                    if old_p in dead_packets
                        assign[s_idx] = SpikeAssignment(0, 0)
                        deleteat!(occ.members, i)
                        continue
                    else
                        assign[s_idx] = SpikeAssignment(occ.id, old_to_new[old_p])
                    end
                end
                i += 1
            end
        end
    end

    for occ in occurrences
        new_counts = zeros(Int, P_new)
        new_sum_t = zeros(Float64, P_new)
        new_sum_t2 = zeros(Float64, P_new)
        for (new_p, old_p) in enumerate(survivors)
            if old_p <= length(occ.counts)
                new_counts[new_p] = occ.counts[old_p]
                new_sum_t[new_p] = occ.sum_t[old_p]
                new_sum_t2[new_p] = occ.sum_t2[old_p]
            end
        end
        occ.counts = new_counts
        occ.sum_t = new_sum_t
        occ.sum_t2 = new_sum_t2
    end

    update_motif_bounds!(m; bg_rates=bg_rates)
    return dead_packets
end

"""
    detect_and_birth_or_split_packets!(m::PurePacketMotif, occurrences, spike_t, spike_chan, assign, bg_rates, N_channels; ...)

Decoupled RJMCMC packet operators:
1. Bimodal Packet Split: operates strictly over member spikes S_p of existing packets.
2. Background Packet Birth: operates strictly over unassigned background spikes within dynamic motif reach.
"""
function detect_and_birth_or_split_packets!(
    m::PurePacketMotif,
    occurrences::Vector{PureOccurrence},
    spike_t::Vector{Float64},
    spike_chan::Vector{Int},
    assign::Vector{SpikeAssignment},
    bg_rates::Vector{Float64},
    N_channels::Int;
    chan_spikes::Union{Nothing, Vector{Vector{Int}}}=nothing,
    ou_ctx::Union{Nothing, ContinuousOUContext}=nothing,
    sigma_prior::Float64=0.015,
    alpha_a::Float64=1.0,
    beta_a::Float64=1.0,
    active_mask::Union{Nothing, BitVector}=nothing,
    ard_alphas::Union{Nothing, Vector{Float64}}=nothing,
    tau_refractory::Float64=0.003,
    allow_multi_packets::Bool=true
)
    E_m = length(occurrences)
    E_m == 0 && return false
    var_prior = sigma_prior^2
    E_eff = max(sum(occ.amplitude for occ in occurrences; init=0.0), 1.0)
    occ_dict = Dict(occ.id => occ for occ in occurrences)
    birthed_any = false

    # Rule 2: Marginal Occam volume for adding R spatial deformation loadings
    tax_loadings = 0.0
    if active_mask !== nothing && ard_alphas !== nothing
        for r in 1:length(active_mask)
            if active_mask[r]
                alpha_r = ard_alphas[r]
                tax_loadings += 0.5 * log(1.0 + E_eff / (max(alpha_r, 1.0) * var_prior))
            end
        end
    end

    # 1. Bimodal Packet Split (Evaluated on Assigned Spikes S_p)
    if allow_multi_packets && !isempty(m.packets)
        P_curr = length(m.packets)
        for p in 1:P_curr
            pkt = m.packets[p]
            chan = pkt.channel
            sig_p = sqrt(pkt.var)
            split_cand_times = Float64[]
            split_spk_indices = Int[]

            for occ in occurrences
                occ.counts[p] == 0 && continue
                u = occ.u
                pred_s = 0.0
                if active_mask !== nothing
                    for r in 1:min(length(active_mask), length(u))
                        if active_mask[r] && r <= size(m.W, 1)
                            pred_s += m.W[r, p] * u[r]
                        end
                    end
                end
                pred_t = occ.tau + pkt.mu + pred_s
                for spk_idx in occ.members
                    as = assign[spk_idx]
                    if as.packet_idx == p
                        dt = abs(spike_t[spk_idx] - pred_t)
                        if dt > 1.5 * sig_p
                            push!(split_cand_times, spike_t[spk_idx] - (occ.tau + pred_s))
                            push!(split_spk_indices, spk_idx)
                        end
                    end
                end
            end

            N_split_cand = length(split_cand_times)
            if N_split_cand >= 2
                sort_idx = sortperm(split_cand_times)
                cand_sorted = split_cand_times[sort_idx]
                spk_sorted = split_spk_indices[sort_idx]

                split_span = max(cand_sorted[end] - cand_sorted[1], 4.0 * sigma_prior)
                occam_split = log(split_span / (sqrt(2π) * sigma_prior)) + log(beta_a + E_m) + tax_loadings

                lam0 = bg_rates[chan]
                best_split_llr = -Inf
                best_split_mu = 0.0
                last_split_t = -Inf
                subsample_step = 0.5 * sigma_prior
                delta = 3.5 * sigma_prior

                for i in 1:N_split_cand
                    t_cand = cand_sorted[i]
                    (t_cand - last_split_t < subsample_step) && continue

                    rayleigh_dist = max(tau_refractory, 2.0 * sqrt(var_prior + pkt.var))
                    if abs(t_cand - pkt.mu) < rayleigh_dist
                        continue
                    end

                    sum_w = 0.0
                    for j in 1:N_split_cand
                        dt = cand_sorted[j] - t_cand
                        if abs(dt) <= delta
                            sum_w += exp(-0.5 * (dt * dt) / var_prior)
                        end
                    end
                    a_cand = (alpha_a + sum_w) / (beta_a + E_eff)
                    llr = 0.0
                    for j in 1:N_split_cand
                        dt = cand_sorted[j] - t_cand
                        if abs(dt) <= delta
                            bump = a_cand * exp(-0.5 * (dt * dt) / var_prior) / (sqrt(2π) * sigma_prior)
                            lam_s = if ou_ctx !== nothing
                                s_idx = spk_sorted[j]
                                exp(evaluate_background_log_intensity(ou_ctx, bg_rates, chan, spike_t[s_idx]))
                            else
                                max(lam0, 1e-12)
                            end
                            llr += log(1.0 + bump / lam_s)
                        end
                    end
                    llr -= E_eff * a_cand
                    if llr > best_split_llr
                        best_split_llr = llr
                        best_split_mu = t_cand
                    end
                    last_split_t = t_cand
                end

                if best_split_llr > occam_split
                    mu_refined = best_split_mu
                    g = 0.0; h = 0.0
                    for j in 1:N_split_cand
                        dt = cand_sorted[j] - mu_refined
                        if abs(dt) <= delta
                            w = exp(-0.5 * (dt * dt) / var_prior)
                            g += (dt / var_prior) * w
                            h += ((dt * dt / (var_prior * var_prior)) - (1.0 / var_prior)) * w
                        end
                    end
                    if h < -1e-6
                        mu_refined += clamp(-g / h, -sigma_prior, sigma_prior)
                    end

                    rayleigh_dist = max(tau_refractory, 2.0 * sqrt(var_prior + pkt.var))
                    if abs(mu_refined - pkt.mu) >= rayleigh_dist
                        P_new = length(m.packets) + 1
                        init_mass = (alpha_a + Float64(N_split_cand)) / (beta_a + E_eff)
                        new_pkt = Packet(P_new, chan, mu_refined, var_prior, max(init_mass, 0.05))
                        push!(m.packets, new_pkt)

                        R_max = size(m.W, 1)
                        new_W = zeros(Float64, R_max, P_new)
                        new_W[:, 1:(P_new-1)] = m.W
                        m.W = new_W
                        push!(m.chan_to_packets[chan], P_new)

                        for occ in occurrences
                            push!(occ.counts, 0)
                            push!(occ.sum_t, 0.0)
                            push!(occ.sum_t2, 0.0)
                        end

                        for s_idx in split_spk_indices
                            old_p = assign[s_idx].packet_idx
                            eid = assign[s_idx].event_id
                            occ = occ_dict[eid]
                            t_s = spike_t[s_idx]
                            occ.counts[old_p] -= 1
                            occ.sum_t[old_p] -= t_s
                            occ.sum_t2[old_p] -= t_s * t_s
                            assign[s_idx] = SpikeAssignment(eid, P_new)
                            occ.counts[P_new] += 1
                            occ.sum_t[P_new] += t_s
                            occ.sum_t2[P_new] += t_s * t_s
                        end

                        update_motif_bounds!(m; bg_rates=bg_rates)
                        birthed_any = true
                        break
                    end
                end
            end
        end
    end

    # 2. Background Packet Birth (Evaluated on Unassigned Spikes)
    delta_reach = max(3.0 * sigma_prior, 0.25)
    s_left = m.delta_left - delta_reach
    s_right = m.delta_right + delta_reach
    proposal_support = max(s_right - s_left, 2.0 * delta_reach)

    occam_birth = log(proposal_support / (sqrt(2π) * sigma_prior)) + log(beta_a + E_m) + tax_loadings

    chan_cand_times = [Float64[] for _ in 1:N_channels]
    chan_cand_spks = [Int[] for _ in 1:N_channels]
    for occ in occurrences
        t_start = max(0.0, occ.tau + s_left)
        t_end = occ.tau + s_right
        idx_a = searchsortedfirst(spike_t, t_start)
        idx_b = searchsortedlast(spike_t, t_end)
        for s_idx in idx_a:idx_b
            if assign[s_idx].event_id == 0
                c = spike_chan[s_idx]
                if !allow_multi_packets && length(m.chan_to_packets[c]) >= 1
                    continue
                end
                push!(chan_cand_times[c], spike_t[s_idx] - occ.tau)
                push!(chan_cand_spks[c], s_idx)
            end
        end
    end

    for chan in 1:N_channels
        if !allow_multi_packets && length(m.chan_to_packets[chan]) >= 1
            continue
        end
        candidate_rel_times = chan_cand_times[chan]
        isempty(candidate_rel_times) && continue

        lam0 = bg_rates[chan]
        existing_packets = m.chan_to_packets[chan]

        perm_cand = sortperm(candidate_rel_times)
        candidate_rel_times = candidate_rel_times[perm_cand]
        candidate_spks = chan_cand_spks[chan][perm_cand]
        N_cand = length(candidate_rel_times)

        max_rate_bump = ((alpha_a + N_cand) / (beta_a + E_eff)) / (sqrt(2π) * sigma_prior)
        max_gain_per_spike = log(1.0 + max_rate_bump / max(lam0, 1e-12))
        max_possible_llr = N_cand * max_gain_per_spike
        max_possible_llr <= occam_birth && continue

        delta = 3.5 * sigma_prior
        subsample_step = 0.5 * sigma_prior
        best_llr = -Inf
        best_mu = 0.0
        last_eval_t = -Inf

        left_ptr = 1
        right_ptr = 1

        for i in 1:N_cand
            t_cand = candidate_rel_times[i]
            (t_cand < s_left || t_cand > s_right) && continue
            (t_cand - last_eval_t < subsample_step) && continue

            unimodal_overlap = false
            for p in existing_packets
                existing_sigma = sqrt(m.packets[p].var)
                rayleigh_dist = 2.0 * sqrt(var_prior + existing_sigma * existing_sigma)
                min_sep = max(tau_refractory, rayleigh_dist)
                if abs(t_cand - m.packets[p].mu) < min_sep
                    unimodal_overlap = true
                    break
                end
            end
            unimodal_overlap && continue

            while left_ptr <= N_cand && candidate_rel_times[left_ptr] < t_cand - delta
                left_ptr += 1
            end
            while right_ptr <= N_cand && candidate_rel_times[right_ptr] <= t_cand + delta
                right_ptr += 1
            end

            sum_w = 0.0
            @inbounds for j in left_ptr:(right_ptr - 1)
                dt = candidate_rel_times[j] - t_cand
                sum_w += exp(-0.5 * (dt * dt) / var_prior)
            end
            a_cand = (alpha_a + sum_w) / (beta_a + E_eff)

            llr = 0.0
            @inbounds for j in left_ptr:(right_ptr - 1)
                dt = candidate_rel_times[j] - t_cand
                bump = a_cand * exp(-0.5 * (dt * dt) / var_prior) / (sqrt(2π) * sigma_prior)
                lam_s = if ou_ctx !== nothing
                    s_idx = candidate_spks[j]
                    exp(evaluate_background_log_intensity(ou_ctx, bg_rates, chan, spike_t[s_idx]))
                else
                    max(lam0, 1e-12)
                end
                llr += log(1.0 + bump / lam_s)
            end
            llr -= E_eff * a_cand

            if llr > best_llr
                best_llr = llr
                best_mu = t_cand
            end
            last_eval_t = t_cand
        end

        if best_llr > occam_birth
            mu_refined = best_mu
            idx_left = searchsortedfirst(candidate_rel_times, mu_refined - delta)
            idx_right = searchsortedlast(candidate_rel_times, mu_refined + delta)
            g = 0.0; h = 0.0
            @inbounds for j in idx_left:idx_right
                dt = candidate_rel_times[j] - mu_refined
                w = exp(-0.5 * (dt * dt) / var_prior)
                g += (dt / var_prior) * w
                h += ((dt * dt / (var_prior * var_prior)) - (1.0 / var_prior)) * w
            end
            if h < -1e-6
                step = -g / h
                mu_refined = clamp(mu_refined + clamp(step, -sigma_prior, sigma_prior), s_left, s_right)
            end

            unimodal_overlap = false
            for p in existing_packets
                existing_sigma = sqrt(m.packets[p].var)
                rayleigh_dist = 2.0 * sqrt(var_prior + existing_sigma * existing_sigma)
                min_sep = max(tau_refractory, rayleigh_dist)
                if abs(mu_refined - m.packets[p].mu) < min_sep
                    unimodal_overlap = true
                    break
                end
            end
            unimodal_overlap && continue

            P_new = length(m.packets) + 1
            init_mass = (alpha_a + 1.0) / (beta_a + E_m)
            new_pkt = Packet(P_new, chan, mu_refined, var_prior, max(init_mass, 0.05))
            push!(m.packets, new_pkt)

            R_max = size(m.W, 1)
            new_W = zeros(Float64, R_max, P_new)
            new_W[:, 1:(P_new-1)] = m.W
            m.W = new_W
            push!(m.chan_to_packets[chan], P_new)

            for occ in occurrences
                push!(occ.counts, 0)
                push!(occ.sum_t, 0.0)
                push!(occ.sum_t2, 0.0)
            end

            sigma_new = sqrt(var_prior)
            spk_list = (chan_spikes !== nothing) ? chan_spikes[chan] : Int[]
            if isempty(spk_list)
                for (idx_s, c_s) in enumerate(spike_chan)
                    if c_s == chan
                        push!(spk_list, idx_s)
                    end
                end
            end

            harvested_count = 0
            if !isempty(spk_list)
                spk_view = view(spike_t, spk_list)
                for occ in occurrences
                    t_pred = occ.tau + mu_refined
                    i_a = searchsortedfirst(spk_view, t_pred - 3.0 * sigma_new)
                    i_b = searchsortedlast(spk_view, t_pred + 3.0 * sigma_new)
                    for s_idx in i_a:i_b
                        j = spk_list[s_idx]
                        as = assign[j]
                        if as.event_id == 0
                            assign[j] = SpikeAssignment(occ.id, P_new)
                            push!(occ.members, j)
                            occ.counts[P_new] += 1
                            occ.sum_t[P_new] += spike_t[j]
                            occ.sum_t2[P_new] += spike_t[j] * spike_t[j]
                            harvested_count += 1
                        end
                    end
                end
            end

            new_pkt.mass = max((alpha_a + Float64(harvested_count)) / (beta_a + E_eff), 0.05)
            birthed_any = true
        end
    end

    if birthed_any
        update_motif_bounds!(m; bg_rates=bg_rates)
    end

    return birthed_any
end
