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
    compute_effective_bg_rate(
        chan::Int,
        mu_p::Float64,
        occurrences::Vector{PureOccurrence},
        bg_rates::Union{Nothing, Vector{Float64}},
        ou_ctx::Union{Nothing, ContinuousOUContext}
    )::Float64

Calculates the occurrence-averaged instantaneous background rate:
    lambda_bar_{0,c}^{eff}(p) = (1 / E_m) * sum_{e=1}^{E_m} lambda_c^{(0)}(tau_e + mu_p)
Ensures birth, death, and Gibbs mass sampling share the identical scalar lower truncation bound:
    a_min(p) = sqrt(2*pi) * sigma_p * lambda_bar_{0,c}^{eff}(p)
"""
function compute_effective_bg_rate(
    chan::Int,
    mu_p::Float64,
    occurrences::Vector{PureOccurrence},
    bg_rates::Union{Nothing, Vector{Float64}},
    ou_ctx::Union{Nothing, ContinuousOUContext}
)::Float64
    T_total_ou = (ou_ctx !== nothing && !isempty(ou_ctx.t_net)) ? ou_ctx.t_net[end] : 2114.29
    min_prior = 1.0 / (1.0 + T_total_ou)
    lam_static = (bg_rates !== nothing && chan <= length(bg_rates)) ? max(bg_rates[chan], min_prior) : 0.5
    if ou_ctx === nothing || isempty(occurrences)
        return lam_static
    end
    sum_lam = 0.0
    for occ in occurrences
        t_eval = occ.tau + mu_p
        sum_lam += exp(evaluate_background_log_intensity(ou_ctx, bg_rates, chan, t_eval))
    end
    return max(sum_lam / Float64(length(occurrences)), min_prior)
end

"""
    compute_channel_tax_birth(
        chan::Int,
        chan_to_packets::Vector{Vector{Int}},
        P_curr::Int,
        N_channels::Int,
        proposal_support::Float64;
        sigma_prior::Float64=0.015,
        eta::Float64=log(2.0)
    )::Float64

Computes the Two-Part MDL + 1D Tonks gas channel Occam tax for adding a packet on `chan` (P_curr -> P_curr + 1):
- New channel (k_c == 0 -> 1):
    tax = ln((N_channels - |C_m|) / (|C_m| + 1)) + ln(P_curr + 1)
- Existing channel (k_c >= 1 -> k_c + 1):
    tax = eta + ln((P_curr + 1) / (k_c + 1)) - Delta_ln_p_Tonks
where Delta_ln_p_Tonks = (k_c + 1) * ln(1 - k_c * Delta_sep / |S_m|) - k_c * ln(1 - (k_c - 1) * Delta_sep / |S_m|).
If |S_m| <= k_c * Delta_sep, returns Inf (Rayleigh packing limit).
"""
function compute_channel_tax_birth(
    chan::Int,
    chan_to_packets::Vector{Vector{Int}},
    P_curr::Int,
    N_channels::Int,
    proposal_support::Float64;
    sigma_prior::Float64=0.015,
    eta::Float64=log(2.0)
)::Float64
    k_c = (chan <= length(chan_to_packets)) ? length(chan_to_packets[chan]) : 0
    active_chans = count(!isempty, chan_to_packets)
    delta_sep = 2.0 * sqrt(2.0) * sigma_prior

    if k_c == 0
        num = max(1, N_channels - active_chans)
        den = active_chans + 1
        return log(Float64(num) / Float64(den)) + log(Float64(P_curr + 1))
    else
        if proposal_support <= Float64(k_c) * delta_sep
            return Inf
        end
        tonks_curr = (k_c > 1) ? Float64(k_c) * log(max(1e-12, 1.0 - (Float64(k_c - 1) * delta_sep) / proposal_support)) : 0.0
        tonks_next = Float64(k_c + 1) * log(max(1e-12, 1.0 - (Float64(k_c) * delta_sep) / proposal_support))
        delta_tonks = tonks_next - tonks_curr
        return eta + log(Float64(P_curr + 1) / Float64(k_c + 1)) - delta_tonks
    end
end

"""
    compute_channel_tax_death(
        chan::Int,
        chan_to_packets::Vector{Vector{Int}},
        P_curr::Int,
        N_channels::Int,
        surv_support::Float64;
        sigma_prior::Float64=0.015,
        eta::Float64=log(2.0)
    )::Float64

Computes the exact symmetric Two-Part MDL + 1D Tonks gas channel Occam credit for removing a packet on `chan` (P_curr -> P_curr - 1):
- Removing the only packet on channel (k_c == 1 -> 0):
    tax = ln((N_channels - |C_m| + 1) / |C_m|) + ln(P_curr)
- Removing one of multiple packets on channel (k_c >= 2 -> k_c - 1):
    tax = eta + ln(P_curr / k_c) - Delta_ln_p_Tonks
where Delta_ln_p_Tonks = k_c * ln(1 - (k_c - 1) * Delta_sep / |S_surv|) - (k_c - 1) * ln(1 - (k_c - 2) * Delta_sep / |S_surv|).
"""
function compute_channel_tax_death(
    chan::Int,
    chan_to_packets::Vector{Vector{Int}},
    P_curr::Int,
    N_channels::Int,
    surv_support::Float64;
    sigma_prior::Float64=0.015,
    eta::Float64=log(2.0)
)::Float64
    k_c = (chan <= length(chan_to_packets)) ? length(chan_to_packets[chan]) : 1
    active_chans = count(!isempty, chan_to_packets)
    delta_sep = 2.0 * sqrt(2.0) * sigma_prior

    if k_c <= 1
        num = max(1, N_channels - active_chans + 1)
        den = max(1, active_chans)
        return log(Float64(num) / Float64(den)) + log(Float64(max(1, P_curr)))
    else
        tonks_curr = Float64(k_c) * log(max(1e-12, 1.0 - (Float64(k_c - 1) * delta_sep) / surv_support))
        tonks_prev = (k_c > 2) ? Float64(k_c - 1) * log(max(1e-12, 1.0 - (Float64(k_c - 2) * delta_sep) / surv_support)) : 0.0
        delta_tonks = tonks_curr - tonks_prev
        return eta + log(Float64(P_curr) / Float64(k_c)) - delta_tonks
    end
end

"""
    compute_motif_channel_log_prior(
        chan_to_packets::Vector{Vector{Int}},
        P_m::Int,
        N_channels::Int,
        proposal_support::Float64;
        sigma_prior::Float64=0.015,
        eta::Float64=log(2.0)
    )::Float64

Computes the exact Two-Part MDL + 1D Tonks gas channel log-prior:
    ln p(c_{1:P_m}) = -ln(N_channels choose |C_m|) - sum_{c in C_m} eta * (k_c - 1)
                      - ln(P_m! / prod_{c in C_m} k_c!) + sum_{c in C_m} Delta_ln_p_Tonks(k_c)
"""
function compute_motif_channel_log_prior(
    chan_to_packets::Vector{Vector{Int}},
    P_m::Int,
    N_channels::Int,
    proposal_support::Float64;
    sigma_prior::Float64=0.015,
    eta::Float64=log(2.0)
)::Float64
    active_chans = count(!isempty, chan_to_packets)
    active_chans == 0 && return 0.0
    delta_sep = 2.0 * sqrt(2.0) * sigma_prior

    # Part 1: -ln(N choose |C_m|)
    ln_binom = loggamma(N_channels + 1) - loggamma(active_chans + 1) - loggamma(N_channels - active_chans + 1)
    ln_p = -ln_binom

    # Part 2: Waveform complexity -sum eta*(k_c - 1) and multinomial -ln(P! / prod k_c!)
    # and Tonks configurational entropy
    ln_multinomial = loggamma(P_m + 1)
    for c in 1:length(chan_to_packets)
        k_c = length(chan_to_packets[c])
        k_c == 0 && continue
        ln_multinomial -= loggamma(k_c + 1)
        if k_c > 1
            ln_p -= eta * Float64(k_c - 1)
            if proposal_support <= Float64(k_c - 1) * delta_sep
                return -Inf
            end
            ln_p += Float64(k_c) * log(max(1e-12, 1.0 - (Float64(k_c - 1) * delta_sep) / proposal_support))
        end
    end
    ln_p -= ln_multinomial
    return ln_p
end

"""
    sample_packet_masses!(m::PurePacketMotif, occurrences::Vector{PureOccurrence}; alpha_a=1.0, beta_a=1.0, bg_rates=nothing, ou_ctx=nothing, rng=default_rng())

Conjugate Gamma draws for packet emission amplitudes a_{m,p} under the biophysical truncated prior:
    a_{m,p} ~ Gamma(alpha_a + sum_e c_{e,p}, 1 / (beta_a + sum_e c_e)) * I(a_{m,p} > a_min(p))
where a_min(p) = sqrt(2pi) * sigma_p * lambda_bar_{0,c}^{eff}(p) guarantees SNR_p > 1.0 for all live packets.
"""
function sample_packet_masses!(
    m::PurePacketMotif,
    occurrences::Vector{PureOccurrence};
    alpha_a::Float64=1.0,
    beta_a::Float64=1.0,
    bg_rates::Union{Nothing, Vector{Float64}}=nothing,
    ou_ctx::Union{Nothing, ContinuousOUContext}=nothing,
    channels::Union{Nothing, AbstractVector{Int}}=nothing,
    rng::AbstractRNG=Random.default_rng()
)
    P = length(m.packets)
    P == 0 && return

    E_tilde = sum(occ.amplitude for occ in occurrences; init=0.0)
    rate_post = beta_a + max(E_tilde, 1e-4)
    scale_post = 1.0 / rate_post

    for p in 1:P
        pkt = m.packets[p]
        chan = pkt.channel
        if channels !== nothing && !(chan in channels)
            continue
        end
        lam0 = compute_effective_bg_rate(chan, pkt.mu, occurrences, bg_rates, ou_ctx)
        sig_p = sqrt(max(pkt.var, 1e-8))
        a_min = (lam0 > 0.0) ? (sqrt(2π) * sig_p * lam0) : 0.0

        c_tot = sum(occ.counts[p] for occ in occurrences; init=0)
        shape_post = max(alpha_a + max(0, c_tot), 1e-4)
        d = Gamma(shape_post, scale_post)
        if a_min > 0.0
            td = truncated(d; lower=a_min)
            pkt.mass = rand(rng, td)
        else
            pkt.mass = rand(rng, d)
        end
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
    S_0::Float64=0.00025,
    channels::Union{Nothing, AbstractVector{Int}}=nothing
)
    P = length(m.packets)
    P == 0 && return 0.0
    E = length(occurrences)

    for p in 1:P
        pkt = m.packets[p]
        chan = pkt.channel
        if channels !== nothing && !(chan in channels)
            continue
        end
        tot_c = 0
        weighted_delta = 0.0
        sum_sq_err = 0.0

        for occ in occurrences
            c = occ.counts[p]
            c == 0 && continue
            u = occ.u
            pred_s = 0.0
            for r in 1:min(length(active_mask), length(u), size(m.W, 1))
                if active_mask[r]
                    pred_s += m.W[r, p] * u[r]
                end
            end
            mean_t = occ.sum_t[p] / c
            target_t = occ.tau + pkt.mu + pred_s
            weighted_delta += (mean_t - target_t) * c
            tot_c += c
            sq_err = occ.sum_t2[p] - 2.0 * target_t * occ.sum_t[p] + c * target_t * target_t
            sum_sq_err += max(0.0, sq_err)
        end

        if tot_c > 0
            raw_shift = damp * (weighted_delta / tot_c)
            # Check 3: Metropolis-Hastings acceptance / regularized shrinkage against Delta Tax_span
            if P > 1
                curr_lats = [m.packets[q].mu for q in 1:P]
                curr_L = maximum(curr_lats) - minimum(curr_lats)
                cand_mu = pkt.mu + raw_shift
                cand_lats = copy(curr_lats)
                cand_lats[p] = cand_mu
                cand_L = maximum(cand_lats) - minimum(cand_lats)

                if cand_L > curr_L && curr_L > 0.0
                    sig_bar = mean(sqrt(max(m.packets[q].var, 1e-8)) for q in 1:P)
                    tax_old = Float64(P) * log(1.0 + curr_L / max(sig_bar, 1e-6))
                    tax_new = Float64(P) * log(1.0 + cand_L / max(sig_bar, 1e-6))
                    delta_tax = tax_new - tax_old

                    var_eff = max(pkt.var, 1e-6)
                    delta_ln_L = raw_shift * (weighted_delta / var_eff) - 0.5 * Float64(tot_c) * (raw_shift * raw_shift / var_eff)

                    if delta_ln_L >= delta_tax || (rand() < exp(delta_ln_L - delta_tax))
                        pkt.mu = cand_mu
                    else
                        span_grad = Float64(P) / (sig_bar + curr_L)
                        shrink = span_grad * var_eff / Float64(tot_c)
                        damped = (raw_shift > 0.0) ? max(0.0, raw_shift - shrink) : min(0.0, raw_shift + shrink)
                        pkt.mu += damped
                    end
                else
                    pkt.mu += raw_shift
                end
            else
                pkt.mu += raw_shift
            end

            nu_post = nu_0 + tot_c
            S_post = S_0 + sum_sq_err
            # Upper bound packet variance to max 200ms SD (var <= 0.04)
            pkt.var = clamp(S_post / nu_post, 1e-6, 0.04)
        end
    end

    origin_offset = 0.0
    if anchor_origin && P > 0 && channels === nothing
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
    channels::Union{Nothing, AbstractVector{Int}}=nothing,
    delta_reach_kw::Union{Nothing, Float64}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    occam_tax::Float64=2.0
)::Vector{Int}
    P = length(m.packets)
    (channels === nothing && P <= 2) && return Int[]
    E_m = length(occurrences)
    E_eff = max(sum(occ.amplitude for occ in occurrences; init=0.0), 1.0)
    var_prior = sigma_prior^2
    N_ch_eff = (channels === nothing) ? N_channels : length(channels)

    dead_packets = Int[]

    for p in 1:P
        if channels === nothing && P - length(dead_packets) <= 2
            break
        end
        pkt = m.packets[p]
        chan = pkt.channel
        if channels !== nothing && !(chan in channels)
            continue
        end
        survivors_tentative = [q for q in 1:P if q != p]
        isempty(survivors_tentative) && continue

        # 1. Physical Survivor Support Envelope: S_m(P_m \ {p})
        min_surv_mu = minimum(m.packets[q].mu for q in survivors_tentative)
        max_surv_mu = maximum(m.packets[q].mu for q in survivors_tentative)
        surv_left = minimum(m.packets[q].mu - 4.0 * sqrt(m.packets[q].var) for q in survivors_tentative)
        surv_right = maximum(m.packets[q].mu + 4.0 * sqrt(m.packets[q].var) for q in survivors_tentative)
        proposal_support = max(surv_right - surv_left, 6.0 * sigma_prior)

        curr_left = minimum(m.packets[q].mu - 4.0 * sqrt(m.packets[q].var) for q in 1:P)
        curr_right = maximum(m.packets[q].mu + 4.0 * sqrt(m.packets[q].var) for q in 1:P)
        proposal_support_curr = max(curr_right - curr_left, 6.0 * sigma_prior)

        # 2. Single-Step Reversibility Guard:
        # If mu_p is outside surv_left..surv_right enlarged by search reach delta_reach,
        # a single birth proposal on the survivors could never reach mu_p.
        delta_reach = (delta_reach_kw !== nothing) ? delta_reach_kw : max(3.0 * sigma_prior, 0.25)
        search_left = surv_left - delta_reach
        search_right = surv_right + delta_reach
        is_extremal = (pkt.mu < min_surv_mu) || (pkt.mu > max_surv_mu)
        if is_extremal && (pkt.mu < search_left || pkt.mu > search_right)
            continue
        end

        # Reversibility Guard: If E_m < 2, reverse packet birth is forbidden (q_birth = 0 => alpha_death = 0)
        if spike_t !== nothing && E_m < 2
            continue
        end

        # 3. Continuous Parameter Occam Volume Tax on Survivor Support with Contraction Bonus and Channel Prior
        tax_loadings = 0.0
        if active_mask !== nothing && ard_alphas !== nothing
            for r in 1:min(length(active_mask), length(ard_alphas))
                if active_mask[r]
                    alpha_r = ard_alphas[r]
                    tax_loadings += 0.5 * log(1.0 + E_eff / (max(alpha_r, 1.0) * var_prior))
                end
            end
        end

        tax_contraction = 0.0
        if proposal_support_curr > proposal_support && proposal_support > 0.0
            tax_contraction = Float64(P - 1) * log(proposal_support_curr / proposal_support)
        end
        tax_chan = compute_channel_tax_death(chan, m.chan_to_packets, P, N_ch_eff, proposal_support; sigma_prior=sigma_prior)
        tax_occam = log(proposal_support / (sqrt(2π) * sigma_prior)) + log(beta_a + E_m) + tax_loadings + tax_chan + tax_contraction

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
                    (s_idx < 1 || s_idx > length(assign)) && continue
                    as = assign[s_idx]
                    (as.packet_idx == p && as.event_id == occ.id) || continue
                    dt = spike_t[s_idx] - t_pred
                    w = exp(-0.5 * (dt * dt) / var_prior)
                    sum_w += w
                    push!(cand_dt, dt)
                    push!(cand_spk, s_idx)
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
                if bump > lam_s
                    llr_p += log(1.0 + bump / lam_s)
                end
            end
            llr_p -= E_eff * a_post

            # Biophysical Truncated Gamma marginal likelihood correction (ln Z_post - ln Z_0)
            lam0 = compute_effective_bg_rate(chan, pkt.mu, occurrences, bg_rates, ou_ctx)
            a_min = sqrt(2π) * sigma_prior * lam0
            if a_min > 0.0
                ln_Z0 = logccdf(Gamma(alpha_a, 1.0 / beta_a), a_min)
                ln_Zpost = logccdf(Gamma(max(alpha_a + sum_w, 1e-4), 1.0 / (beta_a + E_eff)), a_min)
                llr_p += (ln_Zpost - ln_Z0)
            end
        else
            # Analytical fallback for unit tests without full spike arrays
            sigma_p = sqrt(pkt.var)
            lam0 = compute_effective_bg_rate(chan, pkt.mu, occurrences, bg_rates, ou_ctx)
            a_min = sqrt(2π) * sigma_p * lam0
            a_post = (alpha_a + Float64(c_tot)) / (beta_a + E_eff)
            bump = a_post / (sqrt(2π) * sigma_p)
            llr_p = (bump > lam0) ? (Float64(c_tot) * log(1.0 + bump / lam0)) : 0.0
            llr_p -= E_eff * a_post
            if a_min > 0.0
                ln_Z0 = logccdf(Gamma(alpha_a, 1.0 / beta_a), a_min)
                ln_Zpost = logccdf(Gamma(max(alpha_a + Float64(c_tot), 1e-4), 1.0 / (beta_a + E_eff)), a_min)
                llr_p += (ln_Zpost - ln_Z0)
            end
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
    CandidateBirth

Pre-allocated carrier for deterministic candidate evaluation across threads.
"""
struct CandidateBirth
    valid::Bool
    chan::Int
    best_llr::Float64
    mu_refined::Float64
    occam_birth::Float64
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
    allow_multi_packets::Bool=true,
    channels::Union{Nothing, AbstractVector{Int}}=nothing,
    delta_reach_kw::Union{Nothing, Float64}=nothing
)
    E_m = length(occurrences)
    E_m < 2 && return false
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

    delta_reach = (delta_reach_kw !== nothing) ? delta_reach_kw : max(3.0 * sigma_prior, 0.25)
    scan_channels = (channels === nothing) ? (1:N_channels) : channels
    N_ch_eff = (channels === nothing) ? N_channels : length(channels)
    env_left = isempty(m.packets) ? m.delta_left : minimum(m.packets[q].mu - 4.0 * sqrt(m.packets[q].var) for q in 1:length(m.packets))
    env_right = isempty(m.packets) ? m.delta_right : maximum(m.packets[q].mu + 4.0 * sqrt(m.packets[q].var) for q in 1:length(m.packets))
    proposal_support = max(env_right - env_left, 6.0 * sigma_prior)
    s_left = env_left - delta_reach
    s_right = env_right + delta_reach
    tax_chan = log(Float64(max(1, N_ch_eff)))

    # 1. Bimodal Packet Split (Evaluated on Assigned Spikes S_p)
    if allow_multi_packets && !isempty(m.packets)
        P_curr = length(m.packets)
        for p in 1:P_curr
            pkt = m.packets[p]
            chan = pkt.channel
            if channels !== nothing && !(chan in channels)
                continue
            end
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
                    if as.packet_idx == p && as.event_id == occ.id
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
                tax_chan_split = compute_channel_tax_birth(chan, m.chan_to_packets, length(m.packets), N_ch_eff, proposal_support; sigma_prior=sigma_prior)
                isinf(tax_chan_split) && continue
                occam_split = log(split_span / (sqrt(2π) * sigma_prior)) + log(beta_a + E_m) + tax_loadings + tax_chan_split

                best_split_llr = -Inf
                best_split_mu = 0.0
                last_split_t = -Inf
                subsample_step = 0.5 * sigma_prior
                delta = 3.5 * sigma_prior

                for i in 1:N_split_cand
                    t_cand = cand_sorted[i]
                    (t_cand - last_split_t < subsample_step) && continue

                    # Rayleigh Resolvability Boundary (Delta mu >= 2*sqrt(2)*sigma_prior) against all same-channel packets
                    unimodal_overlap = false
                    for q in m.chan_to_packets[chan]
                        q_sigma = sqrt(m.packets[q].var)
                        rayleigh_dist = max(tau_refractory, 2.0 * sqrt(2.0) * sigma_prior, 2.0 * sqrt(var_prior + q_sigma * q_sigma))
                        if abs(t_cand - m.packets[q].mu) < rayleigh_dist
                            unimodal_overlap = true
                            break
                        end
                    end
                    unimodal_overlap && continue

                    sum_w = 0.0
                    for j in 1:N_split_cand
                        dt = cand_sorted[j] - t_cand
                        if abs(dt) <= delta
                            sum_w += exp(-0.5 * (dt * dt) / var_prior)
                        end
                    end
                    a_cand = (alpha_a + sum_w) / (beta_a + E_eff)
                    lam0 = compute_effective_bg_rate(chan, t_cand, occurrences, bg_rates, ou_ctx)
                    a_min = sqrt(2π) * sigma_prior * max(lam0, 1e-12)
                    if a_cand <= a_min
                        continue
                    end

                    ln_Z0 = logccdf(Gamma(alpha_a, 1.0 / beta_a), a_min)
                    ln_Zpost = logccdf(Gamma(alpha_a + sum_w, 1.0 / (beta_a + E_eff)), a_min)
                    delta_ln_Z = ln_Zpost - ln_Z0

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
                            if bump > lam_s
                                llr += log(1.0 + bump / lam_s)
                            end
                        end
                    end
                    llr -= E_eff * a_cand
                    llr += delta_ln_Z
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

                    # Enforce Rayleigh separation on refined latency against all same-channel packets
                    unimodal_overlap = false
                    for q in m.chan_to_packets[chan]
                        q_sigma = sqrt(m.packets[q].var)
                        rayleigh_dist = max(tau_refractory, 2.0 * sqrt(2.0) * sigma_prior, 2.0 * sqrt(var_prior + q_sigma * q_sigma))
                        if abs(mu_refined - m.packets[q].mu) < rayleigh_dist
                            unimodal_overlap = true
                            break
                        end
                    end
                    unimodal_overlap && continue

                    # Existing-Packet Envelope Dilution Tax on Split
                    new_env_left = min(env_left, mu_refined - 4.0 * sigma_prior)
                    new_env_right = max(env_right, mu_refined + 4.0 * sigma_prior)
                    new_proposal_support = max(new_env_right - new_env_left, 6.0 * sigma_prior)
                    tax_dilation = 0.0
                    if channels === nothing && new_proposal_support > proposal_support && proposal_support > 0.0
                        tax_dilation = Float64(P_curr) * log(new_proposal_support / proposal_support)
                    end
                    if best_split_llr <= occam_split + tax_dilation
                        continue
                    end

                    P_new = length(m.packets) + 1
                    init_mass = (alpha_a + Float64(N_split_cand)) / (beta_a + E_eff)
                    lam0_refined = compute_effective_bg_rate(chan, mu_refined, occurrences, bg_rates, ou_ctx)
                    a_min_split = sqrt(2π) * sigma_prior * max(lam0_refined, 1e-12)
                    new_pkt = Packet(P_new, chan, mu_refined, var_prior, max(init_mass, a_min_split + 1e-4))
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
                        as = assign[s_idx]
                        eid = as.event_id
                        !haskey(occ_dict, eid) && continue
                        old_p = as.packet_idx
                        occ = occ_dict[eid]
                        t_s = spike_t[s_idx]
                        if old_p > 0 && old_p <= length(occ.counts)
                            occ.counts[old_p] -= 1
                            occ.sum_t[old_p] -= t_s
                            occ.sum_t2[old_p] -= t_s * t_s
                        end
                        assign[s_idx] = SpikeAssignment(eid, P_new)
                        occ.counts[P_new] += 1
                        occ.sum_t[P_new] += t_s
                        occ.sum_t2[P_new] += t_s * t_s
                    end

                    update_motif_bounds!(m; bg_rates=bg_rates)
                    birthed_any = true
                    env_left = new_env_left
                    env_right = new_env_right
                    proposal_support = new_proposal_support
                    s_left = env_left - delta_reach
                    s_right = env_right + delta_reach
                    break
                end
            end
        end
    end

    # 2. Background Packet Birth (Evaluated on Unassigned Spikes)
    proposal_support = max(env_right - env_left, 6.0 * sigma_prior)
    s_left = env_left - delta_reach
    s_right = env_right + delta_reach

    occ_taus = sort([occ.tau for occ in occurrences])

    max_threads = Threads.maxthreadid()
    tl_cand_times = [Float64[] for _ in 1:max_threads]
    tl_cand_spks  = [Int[] for _ in 1:max_threads]
    for tid in 1:max_threads
        sizehint!(tl_cand_times[tid], 2048)
        sizehint!(tl_cand_spks[tid], 2048)
    end

    best_cands = Vector{CandidateBirth}(undef, N_channels)

    # Step 1: Thread-parallel candidate identification across all channels (Read-only)
    Threads.@threads for chan in scan_channels
        tid = Threads.threadid()
        cand_times_buf = tl_cand_times[tid]
        cand_spks_buf = tl_cand_spks[tid]
        empty!(cand_times_buf)
        empty!(cand_spks_buf)

        if !allow_multi_packets && length(m.chan_to_packets[chan]) >= 1
            best_cands[chan] = CandidateBirth(false, chan, -Inf, 0.0, Inf)
            continue
        end

        if chan_spikes !== nothing
            spks_c = chan_spikes[chan]
            for s_idx in spks_c
                if assign[s_idx].event_id == 0
                    t_s = spike_t[s_idx]
                    i_lo = searchsortedfirst(occ_taus, t_s - s_right)
                    i_hi = searchsortedlast(occ_taus, t_s - s_left)
                    if i_lo <= i_hi
                        for i in i_lo:i_hi
                            push!(cand_times_buf, t_s - occ_taus[i])
                            push!(cand_spks_buf, s_idx)
                        end
                    end
                end
            end
        else
            for occ in occurrences
                t_start = max(0.0, occ.tau + s_left)
                t_end = occ.tau + s_right
                idx_a = searchsortedfirst(spike_t, t_start)
                idx_b = searchsortedlast(spike_t, t_end)
                for s_idx in idx_a:idx_b
                    if assign[s_idx].event_id == 0 && spike_chan[s_idx] == chan
                        push!(cand_times_buf, spike_t[s_idx] - occ.tau)
                        push!(cand_spks_buf, s_idx)
                    end
                end
            end
        end

        if isempty(cand_times_buf)
            best_cands[chan] = CandidateBirth(false, chan, -Inf, 0.0, Inf)
            continue
        end
        existing_packets = m.chan_to_packets[chan]

        tax_chan = compute_channel_tax_birth(chan, m.chan_to_packets, length(m.packets), N_ch_eff, proposal_support; sigma_prior=sigma_prior)
        if isinf(tax_chan)
            best_cands[chan] = CandidateBirth(false, chan, -Inf, 0.0, Inf)
            continue
        end
        occam_birth = log(proposal_support / (sqrt(2π) * sigma_prior)) + log(beta_a + E_m) + tax_loadings + tax_chan

        N_cand = length(cand_times_buf)
        T_total_ou = (ou_ctx !== nothing && !isempty(ou_ctx.t_net)) ? ou_ctx.t_net[end] : 2114.29
        min_prior = 1.0 / (1.0 + T_total_ou)
        lam0_coarse = (bg_rates !== nothing && chan <= length(bg_rates)) ? max(bg_rates[chan], min_prior) : 0.5
        
        # Guard 1: Use true posterior mode a_hat_max and exact -ln(Z0) bound
        a_min_chan = sqrt(2π) * sigma_prior * max(lam0_coarse, 1e-12)
        a_hat_max = max(a_min_chan, (alpha_a + N_cand) / (beta_a + E_eff))
        max_rate_bump = a_hat_max / (sqrt(2π) * sigma_prior)

        ln_Z0 = logccdf(Gamma(alpha_a, 1.0 / beta_a), a_min_chan)
        bound_delta_ln_Z = -ln_Z0
        max_gain_per_spike = max_rate_bump > lam0_coarse ? log(1.0 + max_rate_bump / lam0_coarse) : 0.0
        max_possible_llr = N_cand * max_gain_per_spike + bound_delta_ln_Z
        if max_possible_llr <= occam_birth
            best_cands[chan] = CandidateBirth(false, chan, -Inf, 0.0, Inf)
            continue
        end

        perm_cand = sortperm(cand_times_buf)
        candidate_rel_times = cand_times_buf[perm_cand]
        candidate_spks = cand_spks_buf[perm_cand]

        delta = 3.5 * sigma_prior
        subsample_step = 0.5 * sigma_prior
        best_llr = -Inf
        best_mu = 0.0
        last_eval_t = -Inf

        left_ptr = 1
        right_ptr = 1

        lam0_chan = compute_effective_bg_rate(chan, 0.5 * (s_left + s_right), occurrences, bg_rates, ou_ctx)

        for i in 1:N_cand
            t_cand = candidate_rel_times[i]
            (t_cand < s_left || t_cand > s_right) && continue
            (t_cand - last_eval_t < subsample_step) && continue

            # Rayleigh Resolvability Boundary (Delta mu >= 2*sqrt(2)*sigma_prior) against all same-channel packets
            unimodal_overlap = false
            for p in existing_packets
                existing_sigma = sqrt(m.packets[p].var)
                rayleigh_dist = max(tau_refractory, 2.0 * sqrt(2.0) * sigma_prior, 2.0 * sqrt(var_prior + existing_sigma * existing_sigma))
                if abs(t_cand - m.packets[p].mu) < rayleigh_dist
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
            lam0 = lam0_chan
            a_min = sqrt(2π) * sigma_prior * max(lam0, 1e-12)
            if a_cand <= a_min
                continue
            end

            ln_Z0 = logccdf(Gamma(alpha_a, 1.0 / beta_a), a_min)
            ln_Zpost = logccdf(Gamma(alpha_a + sum_w, 1.0 / (beta_a + E_eff)), a_min)
            delta_ln_Z = ln_Zpost - ln_Z0

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
                if bump > lam_s
                    llr += log(1.0 + bump / lam_s)
                end
            end
            llr -= E_eff * a_cand
            llr += delta_ln_Z

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

            # Enforce Rayleigh separation on refined latency against all same-channel packets
            unimodal_overlap = false
            for p in existing_packets
                existing_sigma = sqrt(m.packets[p].var)
                rayleigh_dist = max(tau_refractory, 2.0 * sqrt(2.0) * sigma_prior, 2.0 * sqrt(var_prior + existing_sigma * existing_sigma))
                if abs(mu_refined - m.packets[p].mu) < rayleigh_dist
                    unimodal_overlap = true
                    break
                end
            end
            if unimodal_overlap
                best_cands[chan] = CandidateBirth(false, chan, -Inf, 0.0, Inf)
            else
                best_cands[chan] = CandidateBirth(true, chan, best_llr, mu_refined, occam_birth)
            end
        else
            best_cands[chan] = CandidateBirth(false, chan, -Inf, 0.0, Inf)
        end
    end

    # Step 2: Sequential main-thread acceptance, Occam re-check, and motif mutation
    for chan in scan_channels
        cand = best_cands[chan]
        !cand.valid && continue

        if !allow_multi_packets && length(m.chan_to_packets[chan]) >= 1
            continue
        end

        # Re-enforce Rayleigh separation against live packets on this channel
        unimodal_overlap = false
        for p in m.chan_to_packets[chan]
            existing_sigma = sqrt(m.packets[p].var)
            rayleigh_dist = max(tau_refractory, 2.0 * sqrt(2.0) * sigma_prior, 2.0 * sqrt(var_prior + existing_sigma * existing_sigma))
            if abs(cand.mu_refined - m.packets[p].mu) < rayleigh_dist
                unimodal_overlap = true
                break
            end
        end
        unimodal_overlap && continue

        # Existing-Packet Envelope Dilution Tax on Birth
        new_env_left = min(env_left, cand.mu_refined - 4.0 * sigma_prior)
        new_env_right = max(env_right, cand.mu_refined + 4.0 * sigma_prior)
        new_proposal_support = max(new_env_right - new_env_left, 6.0 * sigma_prior)
        tax_chan_live = compute_channel_tax_birth(chan, m.chan_to_packets, length(m.packets), N_ch_eff, new_proposal_support; sigma_prior=sigma_prior)
        isinf(tax_chan_live) && continue

        tax_dilation = 0.0
        P_curr = length(m.packets)
        if channels === nothing && new_proposal_support > proposal_support && proposal_support > 0.0
            tax_dilation = Float64(P_curr) * log(new_proposal_support / proposal_support)
        end

        # Two-Part MDL Span Coding Length Change
        L_curr = P_curr > 1 ? (maximum(m.packets[q].mu for q in 1:P_curr) - minimum(m.packets[q].mu for q in 1:P_curr)) : 0.0
        sig_bar_curr = P_curr > 0 ? mean(sqrt(max(m.packets[q].var, 1e-8)) for q in 1:P_curr) : sigma_prior
        tax_span_curr = Float64(P_curr) * log(1.0 + max(0.0, L_curr) / max(sig_bar_curr, 1e-6))

        all_lats_new = [m.packets[q].mu for q in 1:P_curr]
        push!(all_lats_new, cand.mu_refined)
        L_new = maximum(all_lats_new) - minimum(all_lats_new)
        sig_bar_new = (sig_bar_curr * Float64(P_curr) + sigma_prior) / Float64(P_curr + 1)
        tax_span_new = Float64(P_curr + 1) * log(1.0 + max(0.0, L_new) / max(sig_bar_new, 1e-6))
        delta_tax_span = max(0.0, tax_span_new - tax_span_curr)

        occam_birth_actual = log(new_proposal_support / (sqrt(2π) * sigma_prior)) + log(beta_a + E_m) + tax_loadings + tax_chan_live + tax_dilation + delta_tax_span
        if cand.best_llr <= occam_birth_actual
            continue
        end

        P_new = length(m.packets) + 1
        init_mass = (alpha_a + 1.0) / (beta_a + E_m)
        new_pkt = Packet(P_new, chan, cand.mu_refined, var_prior, max(init_mass, 0.05))
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
                t_pred = occ.tau + cand.mu_refined
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

        lam0_birth_refined = compute_effective_bg_rate(chan, cand.mu_refined, occurrences, bg_rates, ou_ctx)
        a_min_birth = sqrt(2π) * sigma_prior * max(lam0_birth_refined, 1e-12)
        new_pkt.mass = max((alpha_a + Float64(harvested_count)) / (beta_a + E_eff), a_min_birth + 1e-4)
        birthed_any = true
        env_left = new_env_left
        env_right = new_env_right
        proposal_support = new_proposal_support
        s_left = env_left - delta_reach
        s_right = env_right + delta_reach
    end

    if birthed_any
        update_motif_bounds!(m; bg_rates=bg_rates)
    end

    return birthed_any
end
