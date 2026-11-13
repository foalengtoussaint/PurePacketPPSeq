# mechanisms/m2_carrier.jl - Level m=2 Sequence Carrier Composition & Dynamic Envelopes
#
# Generative Role:
# - Composes individual emission packets (m=1) into ordered spatiotemporal sequence carriers.
# - Evaluates continuous dynamic bounding envelopes [delta_left, delta_right] and physical span W_eff.
# - Manages carrier occurrence life cycles: candidate birth via continuous Rao score fields,
#   competitive overwrites, and Occam marginal log-evidence pruning.

const SequenceCarrier = PurePacketMotif

function PurePacketMotif(id::Int, packets::Vector{Packet}, R_max::Int, N_channels::Int; W_window::Float64=0.0)
    P = length(packets)
    W = zeros(Float64, R_max, P)
    c2p = [Int[] for _ in 1:N_channels]
    for (p, pkt) in enumerate(packets)
        push!(c2p[pkt.channel], p)
    end
    chan_active = falses(N_channels)
    chan_bounds = fill((0.0, 0.0), N_channels)
    m = PurePacketMotif(id, packets, W, c2p, W_window, 0.0, 0.0, chan_active, chan_bounds)
    update_motif_bounds!(m)
    return m
end

"""
    update_motif_bounds!(m::PurePacketMotif, lambda_min::Vector{Float64}=Float64[]; bg_rates=nothing, kappa=4.0, kappa_u=3.0)

Dynamically computes:
1. Channel-Local Bounding Envelopes: m.chan_bounds[k] = (delta_left_k, delta_right_k).
2. Global Dynamic Bounding Envelope: [m.delta_left, m.delta_right].
3. Active Channel BitVector: m.chan_active[k].
4. Physical Occam Footprint: m.W_window = max(m.delta_right - m.delta_left, 0.0).
"""
function update_motif_bounds!(
    m::PurePacketMotif,
    lambda_min::Vector{Float64}=Float64[];
    bg_rates::Union{Nothing, Vector{Float64}}=nothing,
    kappa::Float64=4.0,
    kappa_u::Float64=3.0
)
    N_chans = length(m.chan_to_packets)
    if length(m.chan_active) != N_chans
        m.chan_active = falses(N_chans)
        m.chan_bounds = fill((0.0, 0.0), N_chans)
    else
        fill!(m.chan_active, false)
    end

    if isempty(m.packets)
        m.delta_left = 0.0
        m.delta_right = 0.0
        m.W_window = 0.0
        fill!(m.chan_bounds, (0.0, 0.0))
        return m
    end

    R = size(m.W, 1)
    global_left = Inf
    global_right = -Inf

    for k in 1:N_chans
        pkts_k = m.chan_to_packets[k]
        if isempty(pkts_k)
            m.chan_bounds[k] = (0.0, 0.0)
        else
            m.chan_active[k] = true
            c_left = Inf
            c_right = -Inf
            lam0 = (bg_rates !== nothing && k <= length(bg_rates)) ? max(bg_rates[k], 1e-12) : 0.5
            for p in pkts_k
                pkt = m.packets[p]
                sig = sqrt(max(pkt.var, 1e-8))

                if bg_rates !== nothing
                    snr_p = pkt.mass / (sqrt(2π) * sig * lam0)
                    R_p = (snr_p > 1.0) ? (sig * sqrt(2.0 * log(snr_p))) : 0.0
                    R_eff = max(R_p, sig)
                else
                    R_eff = kappa * sig
                end

                var_def = 0.0
                if R > 0 && !isempty(lambda_min)
                    for r in 1:min(R, length(lambda_min))
                        var_def += (m.W[r, p]^2) / max(lambda_min[r], 1e-6)
                    end
                end
                delta_p = kappa_u * sqrt(var_def)
                left_p = pkt.mu - R_eff - delta_p
                right_p = pkt.mu + R_eff + delta_p
                c_left = min(c_left, left_p)
                c_right = max(c_right, right_p)
            end
            m.chan_bounds[k] = (c_left, c_right)
            global_left = min(global_left, c_left)
            global_right = max(global_right, c_right)
        end
    end

    m.delta_left = global_left
    m.delta_right = global_right
    m.W_window = isfinite(global_right - global_left) ? max(global_right - global_left, 0.0) : 0.0
    return m
end

const update_channel_bounds! = update_motif_bounds!

function update_all_motif_bounds!(motifs::Dict{Int, PurePacketMotif}; bg_rates=nothing, kappa::Float64=4.0, kappa_u::Float64=3.0)
    for m in values(motifs)
        update_motif_bounds!(m; bg_rates=bg_rates, kappa=kappa, kappa_u=kappa_u)
    end
end

function update_all_motif_bounds!(motifs::Dict{Int, PurePacketMotif}, legacy_arg; bg_rates=nothing, kappa::Float64=4.0, kappa_u::Float64=3.0)
    update_all_motif_bounds!(motifs; bg_rates=bg_rates, kappa=kappa, kappa_u=kappa_u)
end

function rebuild_inverted_index!(m::PurePacketMotif, N_channels::Int)
    if length(m.chan_to_packets) != N_channels
        m.chan_to_packets = [Int[] for _ in 1:N_channels]
    else
        for c in 1:N_channels
            empty!(m.chan_to_packets[c])
        end
    end
    for (p, pkt) in enumerate(m.packets)
        if pkt.channel >= 1 && pkt.channel <= N_channels
            push!(m.chan_to_packets[pkt.channel], p)
        end
    end
    update_motif_bounds!(m)
end

function update_chan_to_motifs(motifs::Dict{Int, PurePacketMotif}, N_channels::Int)::Vector{Vector{Int}}
    chan_to_motifs = [Int[] for _ in 1:N_channels]
    for (mid, m) in motifs
        for c in 1:N_channels
            if c <= length(m.chan_to_packets) && !isempty(m.chan_to_packets[c])
                push!(chan_to_motifs[c], mid)
            end
        end
    end
    return chan_to_motifs
end

"""
    resample_occurrence_onset!(occ, m, active_mask; T_max=Inf, rng=default_rng())

Gibbs draw for occurrence onset tau_e ~ N(mu_tau, tot_prec^-1)
strictly clamped to [max(0.0, -delta_left), T_max - delta_right].
"""
function resample_occurrence_onset!(
    occ::PureOccurrence,
    m::PurePacketMotif,
    active_mask::BitVector;
    T_max::Float64=Inf,
    rng::AbstractRNG=Random.default_rng()
)
    P = length(m.packets)
    P == 0 && return

    potential = 0.0
    tot_prec = 0.0
    u = occ.u

    for p in 1:P
        c = occ.counts[p]
        c == 0 && continue
        pkt = m.packets[p]
        sig2 = pkt.var
        v_inv = c / sig2
        pred_s = 0.0
        @simd for r in 1:min(size(m.W, 1), length(u), length(active_mask))
            if active_mask[r]
                @inbounds pred_s += m.W[r, p] * u[r]
            end
        end
        tot_prec += v_inv
        potential += (v_inv / c) * (occ.sum_t[p] - c * (pkt.mu + pred_s))
    end

    if tot_prec > 0.0
        mu_tau = potential / tot_prec
        min_valid_tau = isfinite(T_max) ? max(0.0, -m.delta_left) : 0.0
        max_valid_tau = isfinite(T_max) ? max(min_valid_tau, T_max - m.delta_right) : Inf
        occ.tau = clamp(mu_tau + randn(rng) / sqrt(tot_prec), min_valid_tau, max_valid_tau)
    end
end

"""
    calc_marginal_log_evidence(m, counts, members, tau, u, spike_t, assign, bg_rates, active_mask; alpha_c=10.0, beta_c=10.0)

Evaluates the exact Marginal Negative Binomial log-evidence Delta ln M(e) integrating out c_e ~ Gamma(alpha_c, beta_c).
"""
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

    prior_deficit = alpha_c * log(beta_c / (beta_c + A_m))
    N_e == 0 && return prior_deficit

    kernel_sum = 0.0
    for spk_idx in members
        p = assign[spk_idx].packet_idx
        (p <= 0 || p > P) && continue
        pkt = m.packets[p]
        sig2 = pkt.var
        sig = sqrt(sig2)
        lam0 = max(bg_rates[pkt.channel], 1e-12)

        pred_s = 0.0
        @simd for r in 1:min(size(m.W, 1), length(u), length(active_mask))
            if active_mask[r]
                @inbounds pred_s += m.W[r, p] * u[r]
            end
        end

        dt = spike_t[spk_idx] - (tau + pkt.mu + pred_s)
        N_i = exp(-0.5 * (dt * dt) / sig2) / (sqrt(2π) * sig)
        ratio = (pkt.mass * N_i) / lam0
        kernel_sum += log(max(ratio, 1e-12))
    end

    gamma_ratio = loggamma(alpha_c + Float64(N_e)) - loggamma(alpha_c)
    scale_term = - Float64(N_e) * log(beta_c + A_m)

    return kernel_sum + gamma_ratio + scale_term + prior_deficit
end

"""
    evaluate_motif_log_intensity(ctx, mid, t) -> Float64

Evaluates continuous motif engagement log-intensity y_m(t) at time t.
Returns 0.0 if ctx is nothing or if motif OU is inactive.
"""
@inline function evaluate_motif_log_intensity(
    ctx::Union{Nothing, ContinuousOUContext},
    mid::Int,
    t::Float64
)::Float64
    ctx === nothing && return 0.0
    (mid < 1 || mid > size(ctx.y_motif, 1)) && return 0.0
    dt = ctx.dt_motif
    B = ctx.B_motif
    B <= 1 && return 0.0
    idx = clamp(floor(Int, t / dt) + 1, 1, B - 1)
    alpha = (t - (idx - 1) * dt) / dt
    return (1.0 - alpha) * ctx.y_motif[mid, idx] + alpha * ctx.y_motif[mid, idx + 1]
end

"""
    compute_effective_motif_duration(ctx, mid, tau_min, tau_max) -> Float64

Calculates the exact piecewise-linear continuous exposure integral:
    T_eff,m = int_{tau_min}^{tau_max} exp(y_m(t)) dt.
Reduces identically to T_span = tau_max - tau_min when y_m(t) == 0.
"""
function compute_effective_motif_duration(
    ctx::Union{Nothing, ContinuousOUContext},
    mid::Int,
    tau_min::Float64,
    tau_max::Float64
)::Float64
    T_span = max(tau_max - tau_min, 1e-6)
    (ctx === nothing || mid < 1 || mid > size(ctx.y_motif, 1)) && return T_span
    dt = ctx.dt_motif
    B = ctx.B_motif
    B <= 1 && return T_span

    total = 0.0
    for j in 1:(B-1)
        t_lo = (j - 1) * dt
        t_hi = j * dt
        seg_lo = max(t_lo, tau_min)
        seg_hi = min(t_hi, tau_max)
        if seg_hi > seg_lo
            alpha_lo = (seg_lo - t_lo) / dt
            alpha_hi = (seg_hi - t_lo) / dt
            y_lo = (1.0 - alpha_lo) * ctx.y_motif[mid, j] + alpha_lo * ctx.y_motif[mid, j+1]
            y_hi = (1.0 - alpha_hi) * ctx.y_motif[mid, j] + alpha_hi * ctx.y_motif[mid, j+1]
            total += exp_integral_segment(y_lo, y_hi, seg_hi - seg_lo)
        end
    end
    return max(total, 0.01 * T_span)
end


"""
    prune_dead_occurrences!(occurrences, motifs, spike_t, assign, bg_rates, T_max, active_mask; ...)

Prunes occurrences using exact continuous-time RJMCMC reverse jump acceptance:
    ln R_death(e) = -Delta ln M(e) - y_m(tau_e) + ln( (beta_m + T_eff,m) * (E_m / (alpha_m + E_m - 1)) * q_birth(tau_e) )
with Campbell's theorem null-expectation normalized proposal density:
    q_birth(tau_e) = (q_sum(tau_e) + A_m) / (W_m + A_m * T_span)
evaluated on the reference background pool D_bg ∪ S_e.
Accepts death stochastically via log(rand(rng)) < min(0.0, ln_R_death).
"""
function prune_dead_occurrences!(
    occurrences::Dict{Int, PureOccurrence},
    motifs::Dict{Int, PurePacketMotif},
    spike_t::Vector{Float64},
    assign::Vector{SpikeAssignment},
    bg_rates::Vector{Float64},
    T_max::Float64,
    active_mask::BitVector;
    chan_to_spikes::Union{Nothing, Vector{Vector{Int}}}=nothing,
    counts_bg::Union{Nothing, Vector{Int}}=nothing,
    alpha_c::Float64=10.0,
    beta_c::Float64=10.0,
    alpha_m::Float64=0.05,
    beta_m::Float64=1.0,
    gamma_m::Float64=100.0,
    ou_ctx::Union{Nothing, ContinuousOUContext}=nothing,
    rng::AbstractRNG=Random.default_rng()
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

        N_e = sum(occ.counts; init=0)
        if N_e == 0
            push!(pruned_eids, eid)
            continue
        end

        P = length(m.packets)
        tau_min = max(0.0, -m.delta_left)
        tau_max = max(tau_min, T_max - m.delta_right)
        T_span = max(tau_max - tau_min, 1e-6)
        A_m = sum(p.mass for p in m.packets; init=0.0)
        null_mass = A_m * T_span

        # 1. Evaluate Delta ln M(e) while spikes are currently assigned
        delta_ln_M = calc_marginal_log_evidence(
            m, occ.counts, occ.members, occ.tau, occ.u,
            spike_t, assign, bg_rates, active_mask;
            alpha_c=alpha_c, beta_c=beta_c
        )

        # 2. Reference state: temporarily release member spikes back to background (D_bg ∪ S_e)
        orig_assigns = [(s, assign[s]) for s in occ.members]
        for (s, a) in orig_assigns
            assign[s] = SpikeAssignment(0, 0)
            if counts_bg !== nothing && a.packet_idx > 0 && a.packet_idx <= P
                k = m.packets[a.packet_idx].channel
                if k <= length(counts_bg)
                    counts_bg[k] += 1
                end
            end
        end

        # 3. Recompute both W_m and q_sum(tau_e) on the post-removal pool (D_bg ∪ S_e)
        W_m = 0.0
        q_sum = 0.0
        for p in 1:P
            pkt = m.packets[p]
            k = pkt.channel
            lam0 = max(bg_rates[k], 1e-12)
            w_p = pkt.mass / lam0
            sig2 = pkt.var
            sig = sqrt(sig2)
            inv_2sig2 = 0.5 / sig2
            norm_const = 1.0 / (sqrt(2pi) * sig)
            t_pred = occ.tau + pkt.mu

            if chan_to_spikes !== nothing
                spks_k = chan_to_spikes[k]
                N_spks_k = length(spks_k)
                if N_spks_k > 0
                    t_lo = t_pred - 4.0 * sig
                    t_hi = t_pred + 4.0 * sig
                    i_lo = searchsortedfirst(1:N_spks_k, t_lo, lt=(idx, target) -> spike_t[spks_k[idx]] < target)
                    i_hi = searchsortedlast(1:N_spks_k, t_hi, lt=(target, idx) -> target < spike_t[spks_k[idx]])
                    @inbounds for j in i_lo:i_hi
                        s = spks_k[j]
                        if assign[s].event_id == 0
                            dt = spike_t[s] - t_pred
                            N_j = exp(-dt * dt * inv_2sig2) * norm_const
                            q_sum += w_p * N_j
                        end
                    end
                end
                N_bg_k = (counts_bg !== nothing && k <= length(counts_bg)) ? counts_bg[k] : count(s -> assign[s].event_id == 0, spks_k)
                W_m += w_p * Float64(N_bg_k)
            else
                N_bg_k = occ.counts[p]
                for (s, a) in orig_assigns
                    if a.packet_idx == p
                        dt = spike_t[s] - t_pred
                        if abs(dt) <= 4.0 * sig
                            N_j = exp(-dt * dt * inv_2sig2) * norm_const
                            q_sum += w_p * N_j
                        end
                    end
                end
                W_m += w_p * Float64(N_bg_k)
            end
        end

        denom_q = W_m + null_mass
        q_birth = (q_sum + A_m) / max(denom_q, 1e-12)

        y_m = evaluate_motif_log_intensity(ou_ctx, occ.motif, occ.tau)
        T_eff_m = compute_effective_motif_duration(ou_ctx, occ.motif, tau_min, tau_max)

        E_m = get(motif_occ_counts, occ.motif, 1)
        ratio_E = Float64(E_m) / max(alpha_m + Float64(E_m) - 1.0, 1e-12)
        ln_R_death = -delta_ln_M - y_m + log((beta_m + T_eff_m) * ratio_E * q_birth)

        if log(rand(rng)) < min(0.0, ln_R_death)
            # Stochastic acceptance of death: keep spikes released
            push!(pruned_eids, eid)
            motif_occ_counts[occ.motif] -= 1
        else
            # Rejected death: restore original spike assignments
            for (s, orig_a) in orig_assigns
                assign[s] = orig_a
                if counts_bg !== nothing && orig_a.packet_idx > 0 && orig_a.packet_idx <= P
                    k = m.packets[orig_a.packet_idx].channel
                    if k <= length(counts_bg)
                        counts_bg[k] -= 1
                    end
                end
            end
        end
    end

    for eid in pruned_eids
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

"""
    detect_and_birth_occurrences!(occurrences, motifs, spike_t, spike_chan, chan_to_spikes, assign, bg_rates, T_max, R_max; ...)

Data-driven reversible jump MCMC birth of sequence occurrences derived from the continuous Rao score field S_m(tau).
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
    alpha_m::Float64=0.05,
    beta_m::Float64=1.0,
    gamma_m::Float64=100.0,
    delta_margin::Float64=3.0,
    refractory_fraction::Float64=0.5,
    delta_win::Float64=0.030,
    alpha_stopping::Float64=0.025,
    K_prop::Int=50,
    bg_times::Union{Nothing, Vector{Vector{Float64}}}=nothing,
    bg_indices::Union{Nothing, Vector{Vector{Int}}}=nothing,
    active_mask::Union{Nothing, BitVector}=nothing,
    ou_ctx::Union{Nothing, ContinuousOUContext}=nothing,
    rng::AbstractRNG=Random.default_rng()
)
    next_eid = isempty(occurrences) ? 1 : (maximum(keys(occurrences)) + 1)
    intra_sweep_exclusion = Tuple{Int, Float64}[]
    n_birthed = 0
    n_overwritten = 0
    act_mask = active_mask !== nothing ? active_mask : trues(R_max)

    N_channels = length(chan_to_spikes)

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

    cand_spk_indices = Int[]
    cand_pkt_indices = Int[]
    cand_lr = Float64[]
    sizehint!(cand_spk_indices, 256)
    sizehint!(cand_pkt_indices, 256)
    sizehint!(cand_lr, 256)

    for (mid, m) in motifs
        P = length(m.packets)
        P < 2 && continue
        W_eff = max(m.delta_right - m.delta_left, 0.05)
        tau_min = max(0.0, -m.delta_left)
        tau_max = max(tau_min, T_max - m.delta_right)
        T_span = max(tau_max - tau_min, 1e-6)
        total_mass = sum(p.mass for p in m.packets; init=0.0)
        A_m = total_mass
        null_mass = A_m * T_span
        E_m = count(occ -> occ.motif == mid, values(occurrences))
        sig_min = minimum(sqrt(p.var) for p in m.packets)

        prior_deficit = alpha_c * log(beta_c / (beta_c + total_mass))
        N_min = 2

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
        denom_q = W_m + null_mass
        eps_m = denom_q > 0.0 ? (null_mass / denom_q) : 1.0

        p_hit_1 = sum_snr / W_m
        L_m_star = max(ceil(Int, log(1.0 / alpha_stopping) / p_hit_1), 100)
        refract_dist = refractory_fraction * W_eff

        c_reject = 0

        while c_reject < L_m_star
            if rand(rng) < eps_m
                tau_star = rand(rng) * (tau_max - tau_min) + tau_min
            else
                u_draw = rand(rng) * W_m
                p_star = clamp(searchsortedfirst(cum_W, u_draw), 1, P)
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
                tau_star = tau_seed
            end

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

            ws = get_kinematic_workspace()
            K = 0
            for r in 1:length(act_mask)
                if act_mask[r]
                    K += 1
                    ws.act[K] = r
                end
            end
            curr_R = length(act_mask)
            u_star_vec = view(ws.u, 1:curr_R)
            fill!(u_star_vec, 0.0)

            kernel_sum = sum(cand_lr)
            if K > 0 && N_cand >= 2
                A_mat = view(ws.A, 1:K, 1:K)
                b_vec = view(ws.b, 1:K)
                u_sub = view(ws.z, 1:K)
                fill!(A_mat, 0.0)
                fill!(b_vec, 0.0)
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
                        wr1 = m.W[ws.act[r1], p]
                        b_vec[r1] += kap * wr1 * y
                        for r2 in 1:K
                            A_mat[r1, r2] += kap * wr1 * m.W[ws.act[r2], p]
                        end
                    end
                end
                C = cholesky!(Hermitian(A_mat))
                ldiv!(u_sub, C, b_vec)

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
                        pred_s += m.W[ws.act[r], p] * u_sub[r]
                    end
                    dt_warp = spike_t[s] - (tau_star + pkt.mu + pred_s)
                    N_j = exp(-0.5 * (dt_warp * dt_warp) / sig2) / (sqrt(2pi) * sig)
                    ratio = (pkt.mass * N_j) / lam0
                    kernel_sum_warped += log(max(ratio, 1e-12))
                end
                if kernel_sum_warped > kernel_sum
                    kernel_sum = kernel_sum_warped
                    for r in 1:K
                        u_star_vec[ws.act[r]] = u_sub[r]
                    end
                end
            end

            gamma_ratio = loggamma(alpha_c + Float64(N_cand)) - loggamma(alpha_c)
            scale_term = - Float64(N_cand) * log(beta_c + total_mass)
            delta_ln_M_star = kernel_sum + gamma_ratio + scale_term + prior_deficit

            q_birth = (q_sum + A_m) / max(denom_q, 1e-12)
            y_m = evaluate_motif_log_intensity(ou_ctx, mid, tau_star)
            T_eff_m = compute_effective_motif_duration(ou_ctx, mid, tau_min, tau_max)

            # Forward jump: E_m -> E_m + 1
            ratio_E = (Float64(E_m) + 1.0) / (alpha_m + Float64(E_m))
            ln_R = delta_ln_M_star + y_m - log((beta_m + T_eff_m) * ratio_E * q_birth)

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
                    new_occ = PureOccurrence(next_eid, mid, tau_star, curr_R, P; amplitude=1.0)
                    next_eid += 1
                    for r in 1:curr_R
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

                    c_reject = 0

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
                    denom_q = W_m + null_mass
                    eps_m = denom_q > 0.0 ? (null_mass / denom_q) : 1.0
                    if W_m > 0.0
                        cum_W = cumsum(W_mp)
                    end
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
                    for r in 1:min(length(incumbent.u), curr_R)
                        incumbent.u[r] = u_star_vec[r]
                    end
                    for r in (curr_R + 1):length(incumbent.u)
                        incumbent.u[r] = 0.0
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

                    c_reject = 0

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

# Checkpoint backward compatibility deserializer
const MOTIF_FLOAT64_TAG = Int32(0x0e)

function Serialization.deserialize(s::Serialization.AbstractSerializer, t::Type{PurePacketMotif})
    x = ccall(:jl_new_struct_uninit, Any, (Any,), t)
    Serialization.deserialize_cycle(s, x)
    for i in 1:5
        tag = Int32(read(s.io, UInt8)::UInt8)
        if tag != Serialization.UNDEFREF_TAG
            ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), x, i-1, Serialization.handle_deserialize(s, tag))
        end
    end
    N_chans = length(x.chan_to_packets)
    x.chan_active = falses(N_chans)
    x.chan_bounds = fill((0.0, 0.0), N_chans)
    x.delta_left = 0.0
    x.delta_right = 0.0

    mark(s.io)
    if !eof(s.io)
        tag6 = Int32(read(s.io, UInt8)::UInt8)
        if tag6 == MOTIF_FLOAT64_TAG
            x.delta_left = Serialization.handle_deserialize(s, tag6)
            tag7 = Int32(read(s.io, UInt8)::UInt8)
            x.delta_right = Serialization.handle_deserialize(s, tag7)
            
            mark(s.io)
            if !eof(s.io)
                tag8 = Int32(read(s.io, UInt8)::UInt8)
                val8 = Serialization.handle_deserialize(s, tag8)
                if val8 isa BitVector && length(val8) == N_chans
                    x.chan_active = val8
                    tag9 = Int32(read(s.io, UInt8)::UInt8)
                    x.chan_bounds = Serialization.handle_deserialize(s, tag9)
                    unmark(s.io)
                    unmark(s.io)
                    return x
                else
                    reset(s.io)
                end
            end
            unmark(s.io)
            update_motif_bounds!(x)
            return x
        end
    end
    reset(s.io)
    update_motif_bounds!(x)
    return x
end
