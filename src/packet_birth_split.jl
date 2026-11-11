# packet_birth_split.jl - Nonparametric Packet Birth, Split & Pruning

function sample_packet_masses!(
    m::PurePacketMotif,
    occurrences::Vector{PureOccurrence};
    alpha_a::Float64=1.0,
    beta_a::Float64=1.0,
    rng::AbstractRNG=Random.default_rng()
)
    P = length(m.packets)
    P == 0 && return

    # Effective trial exposure is sum_{e in E_m} c_e
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
    sample_occurrence_amplitudes!(occurrences, motifs; alpha_c=25.0, rng)

Conjugate Gamma draw for occurrence amplitudes c_e:
    c_e ~ Gamma(alpha_c + N_e, 1.0 / (alpha_c + sum_{p} a_{m_e, p}))
where N_e = sum_p counts[p] is the total assigned spike count, and sum_p a_{m_e, p} is the total active packet mass.
Alpha_c >= 20 clamps the ensemble mean strictly to 1.00 +- 0.02, preventing hyperbolic scale drift c_e * a_{m,p} = const.
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
            # Sufficient variance residual: sum (t_i - target_t)^2 = sum_t2 - 2 target sum_t + c target^2
            sq_err = occ.sum_t2[p] - 2.0 * target_t * occ.sum_t[p] + c * target_t * target_t
            sum_sq_err += max(0.0, sq_err)
        end

        if tot_c > 0
            pkt.mu += damp * (weighted_delta / tot_c)
            # Conjugate Inverse-Gamma update for packet variance
            nu_post = nu_0 + tot_c
            S_post = S_0 + sum_sq_err
            pkt.var = S_post / nu_post
        end
    end

    # Anchor origin: Packet 1 is locked to identically 0.0000 s
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

    update_motif_bounds!(m)
    return origin_offset
end

# =====================================================================
# 4. NONPARAMETRIC DEATH: MASS COLLAPSE & ANCHOR HANDOVER
# =====================================================================

"""
    prune_dead_packets!(
        m::PurePacketMotif,
        occurrences::Vector{PureOccurrence},
        bg_rates::Vector{Float64},
        N_channels::Int;
        assign::Union{Nothing, Vector{SpikeAssignment}}=nothing,
        occam_tax::Float64=2.0
    )

Prunes packets whose peak instantaneous firing rate drops below the Poisson background floor (SNR_p <= 1.0):
    SNR_p = lambda_peak / lambda_0(k_p) = (a_{m,p} / (sqrt(2pi) * sigma_p)) / lambda_0(k_p) <= 1.0
Or whose empirical count is zero and cannot clear the background rate floor.
Executes the ANCHOR HANDOVER rule if Packet 1 dies:
    tau_e += mu_{p_next}, mu_p -= mu_{p_next}
Releases assigned spikes of dead packets back to background: assign[i] = (0, 0).
Rebuilds inverted index, atomically shrinks motif total mass A_m, and updates dense occurrence moment vectors.
"""
function prune_dead_packets!(
    m::PurePacketMotif,
    occurrences::Vector{PureOccurrence},
    bg_rates::Vector{Float64},
    N_channels::Int;
    assign::Union{Nothing, Vector{SpikeAssignment}}=nothing,
    occam_tax::Float64=2.0
)
    P = length(m.packets)
    P <= 1 && return Int[]
    E_eff = max(sum(occ.amplitude for occ in occurrences; init=0.0), 1.0)

    dead_packets = Int[]
    for p in 1:P
        pkt = m.packets[p]
        sigma_p = sqrt(pkt.var)
        c_tot = sum(occ.counts[p] for occ in occurrences; init=0)
        lam0 = max(bg_rates[pkt.channel], 1e-12)
        lambda_peak = pkt.mass / (sqrt(2π) * sigma_p)
        snr_p = lambda_peak / lam0

        # Exact SNR Death Condition (Rule 3):
        # A packet whose peak firing rate cannot beat Poisson background (SNR <= 1.0)
        # provides negative log-likelihood to spikes while adding useless compensator mass.
        # Also prune if zero assigned spikes and empirical rate cannot clear background floor.
        rate_empirical = c_tot / E_eff
        bg_floor = lam0 * sqrt(2π) * sigma_p

        if snr_p <= 1.0 || (c_tot == 0 && rate_empirical < bg_floor)
            push!(dead_packets, p)
        end
    end

    isempty(dead_packets) && return dead_packets

    # RJMCMC Boundary Reversibility Gate:
    # Reject death moves if removing an extremal packet leaves it outside the reverse
    # birth proposal reach of the surviving packets: [delta_left^surv - delta_reach, delta_right^surv + delta_reach]
    if !isempty(dead_packets) && length(dead_packets) < P
        survivors_tentative = [p for p in 1:P if !(p in dead_packets)]
        if !isempty(survivors_tentative)
            min_surv_mu = minimum(m.packets[p].mu for p in survivors_tentative)
            max_surv_mu = maximum(m.packets[p].mu for p in survivors_tentative)

            sigma_bar = mean(sqrt(m.packets[p].var) for p in survivors_tentative)
            delta_reach = max(3.0 * sigma_bar, 0.25)
            surv_left = minimum(m.packets[p].mu - 4.0 * sqrt(m.packets[p].var) for p in survivors_tentative) - delta_reach
            surv_right = maximum(m.packets[p].mu + 4.0 * sqrt(m.packets[p].var) for p in survivors_tentative) + delta_reach

            filtered_dead = Int[]
            for p in dead_packets
                mu_p = m.packets[p].mu
                is_extremal = (mu_p < min_surv_mu) || (mu_p > max_surv_mu)
                if is_extremal
                    if mu_p < surv_left || mu_p > surv_right
                        # Packet is strictly outside the reverse birth proposal reach of survivors:
                        # reject prune to preserve MH detailed balance
                        continue
                    end
                end
                push!(filtered_dead, p)
            end
            dead_packets = filtered_dead
        end
    end

    isempty(dead_packets) && return dead_packets

    # Check Anchor Handover: is Packet 1 being pruned?
    if 1 in dead_packets
        # Find earliest surviving packet
        first_survivor = 0
        for p in 1:P
            if !(p in dead_packets)
                first_survivor = p
                break
            end
        end
        if first_survivor > 0
            handover_shift = m.packets[first_survivor].mu
            # Anchor Handover: all surviving packets shift by -handover_shift
            for p in 1:P
                m.packets[p].mu -= handover_shift
            end
            # All occurrences absorb this exact shift: tau_e += handover_shift
            for occ in occurrences
                occ.tau += handover_shift
            end
        end
    end

    # Build survivors
    survivors = [p for p in 1:P if !(p in dead_packets)]
    if isempty(survivors)
        # Never prune all packets; retain the one with highest mass
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

    # Remap spike assignments and update occurrence moment vectors
    old_to_new = zeros(Int, P)
    for (new_p, old_p) in enumerate(survivors)
        old_to_new[old_p] = new_p
    end

    if assign !== nothing
        for occ in occurrences
            surviving_members = Int[]
            for spk_idx in occ.members
                old_p = assign[spk_idx].packet_idx
                if old_p > 0 && old_p <= P
                    new_p = old_to_new[old_p]
                    if new_p > 0
                        assign[spk_idx] = SpikeAssignment(occ.id, new_p)
                        push!(surviving_members, spk_idx)
                    else
                        assign[spk_idx] = SpikeAssignment(0, 0)
                    end
                end
            end
            occ.members = surviving_members
        end
    end

    for occ in occurrences
        new_counts = [occ.counts[old_p] for old_p in survivors]
        new_sum_t = [occ.sum_t[old_p] for old_p in survivors]
        new_sum_t2 = [occ.sum_t2[old_p] for old_p in survivors]
        occ.counts = new_counts
        occ.sum_t = new_sum_t
        occ.sum_t2 = new_sum_t2
    end

    update_motif_bounds!(m)
    return dead_packets
end

"""
    calc_marginal_log_evidence(
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

Calculates the exact Marginal Negative Binomial log-evidence Delta ln M(e)
integrating out occurrence gain c_e ~ Gamma(alpha_c, beta_c):
    Delta ln M(e) = sum_{p} sum_{i in p} ln( (a_{m,p} * N_i) / lambda_0(k_p) )
                  + ln(Gamma(alpha_c + N_e) / Gamma(alpha_c))
                  - N_e * ln(beta_c + A_m)
                  + alpha_c * ln(beta_c / (beta_c + A_m))
where A_m = sum_p a_{m,p} and N_e = sum_p counts[p].
"""

# =====================================================================
# 7. NONPARAMETRIC BIRTH & SPLIT: CONTINUOUS BAYES FACTOR
# =====================================================================

"""
    detect_and_birth_or_split_packets!(
        m::PurePacketMotif,
        occurrences::Vector{PureOccurrence},
        spike_t::Vector{Float64},
        spike_chan::Vector{Int},
        assign::Vector{SpikeAssignment},
        bg_rates::Vector{Float64},
        N_channels::Int;
        sigma_prior::Float64=0.015,
        alpha_a::Float64=1.0,
        beta_a::Float64=1.0
    )

Continuous Bayes Factor birth & bimodal split operator:
1. Gathers candidate spikes: background spikes inside occurrences + spikes on existing packets with |residual| > 1.5 sigma_p.
2. Evaluates Delta LLR directly at candidate spike timestamps (0 grid binning).
3. Refines with 1 local Newton-Raphson step.
4. Accepts if max Delta LLR > continuous Occam volume penalty log(W / sqrt(2pi*sigma^2)).
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

    # Rule 2: Exact Marginal Occam Volume for adding R spatial deformation loadings W_{r,p}
    tax_loadings = 0.0
    if active_mask !== nothing && ard_alphas !== nothing
        for r in 1:length(active_mask)
            if active_mask[r]
                alpha_r = ard_alphas[r]
                tax_loadings += 0.5 * log(1.0 + E_eff / (max(alpha_r, 1.0) * var_prior))
            end
        end
    end

    # =============================================================
    # 1. BIMODAL PACKET SPLIT OPERATOR (Evaluated on Assigned Spikes)
    # =============================================================
    if allow_multi_packets && !isempty(m.packets)
        P_curr = length(m.packets)
        for p in 1:P_curr
            pkt = m.packets[p]
            chan = pkt.channel
            sig_p = sqrt(pkt.var)
            split_cand_times = Float64[]
            split_spk_indices = Int[]

            # Gather spikes assigned to packet p with residual |dt| > 1.5 * sig_p
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

                # Support of split proposal is strictly the empirical span of candidate split spikes
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

                    # Rayleigh bimodality check against existing packet p
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
                            llr += log(1.0 + bump / lam0)
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
                    # Continuous Newton refinement
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

                    # Verify Rayleigh separation on refined center
                    rayleigh_dist = max(tau_refractory, 2.0 * sqrt(var_prior + pkt.var))
                    if abs(mu_refined - pkt.mu) >= rayleigh_dist
                        # Birth split packet P_new
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

                        # Reassign split candidate spikes to new packet
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

    # =============================================================
    # 2. BACKGROUND PACKET BIRTH OPERATOR (Evaluated on Unassigned Spikes)
    # =============================================================
    delta_reach = max(3.0 * sigma_prior, 0.25)
    s_left = m.delta_left - delta_reach
    s_right = m.delta_right + delta_reach
    proposal_support = max(s_right - s_left, 2.0 * delta_reach)

    occam_birth = log(proposal_support / (sqrt(2π) * sigma_prior)) + log(beta_a + E_m) + tax_loadings

    chan_cand_times = [Float64[] for _ in 1:N_channels]
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
            end
        end
    end

    # Evaluate candidate channels for background birth
    for chan in 1:N_channels
        if !allow_multi_packets && length(m.chan_to_packets[chan]) >= 1
            continue
        end
        candidate_rel_times = chan_cand_times[chan]
        isempty(candidate_rel_times) && continue

        lam0 = bg_rates[chan]
        existing_packets = m.chan_to_packets[chan]

        sort!(candidate_rel_times)
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
                llr += log(1.0 + bump / lam0)
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
            g = 0.0
            h = 0.0
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

            # Birth Packet P_new from background
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

