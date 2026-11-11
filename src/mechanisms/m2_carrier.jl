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
        if K == 1; pred_s = m.W[k1, p] * u[k1]
        elseif K == 2; pred_s = m.W[k1, p] * u[k1] + m.W[k2, p] * u[k2]
        elseif K == 3; pred_s = m.W[k1, p] * u[k1] + m.W[k2, p] * u[k2] + m.W[k3, p] * u[k3]
        elseif K == 4; pred_s = m.W[k1, p] * u[k1] + m.W[k2, p] * u[k2] + m.W[k3, p] * u[k3] + m.W[k4, p] * u[k4]
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

    gamma_ratio = loggamma(alpha_c + Float64(N_e)) - loggamma(alpha_c)
    scale_term = - Float64(N_e) * log(beta_c + A_m)

    return kernel_sum + gamma_ratio + scale_term + prior_deficit
end

"""
    prune_dead_occurrences!(occurrences, motifs, spike_t, assign, bg_rates, T_max, active_mask; alpha_c=10.0, beta_c=10.0, gamma_m=100.0)

Prunes occurrences that fail the exact Marginal Negative Binomial Bayes Factor test:
    Delta ln M(e) <= Tax_Occam
Releases member spikes back to background.
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

        delta_ln_M = calc_marginal_log_evidence(
            m, occ.counts, occ.members, occ.tau, occ.u,
            spike_t, assign, bg_rates, active_mask;
            alpha_c=alpha_c, beta_c=beta_c
        )

        if delta_ln_M <= tax_occam
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
        P == 0 && continue
        W_eff = max(m.delta_right - m.delta_left, 0.05)
        total_mass = sum(p.mass for p in m.packets; init=0.0)
        E_m = count(occ -> occ.motif == mid, values(occurrences))
        sig_min = minimum(sqrt(p.var) for p in m.packets)

        prior_deficit = alpha_c * log(beta_c / (beta_c + total_mass))
        tax_occam = log(max((T_max - W_eff) / W_eff, 1.0)) + log(max((E_m + 1.0) / gamma_m, 1e-3))
        N_min = max(ceil(Int, (abs(prior_deficit) + tax_occam) / 3.5), 2)

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

        p_hit_1 = sum_snr / W_m
        L_m_star = max(ceil(Int, log(1.0 / alpha_stopping) / p_hit_1), 100)
        refract_dist = refractory_fraction * W_eff
        tau_min = max(0.0, -m.delta_left)
        tau_max = max(tau_min, T_max - m.delta_right)

        c_reject = 0

        while c_reject < L_m_star
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

            gamma_ratio = loggamma(alpha_c + Float64(N_cand)) - loggamma(alpha_c)
            scale_term = - Float64(N_cand) * log(beta_c + total_mass)
            delta_ln_M_star = kernel_sum + gamma_ratio + scale_term + prior_deficit

            q_tau = max(q_sum / W_m, 1e-15)
            ln_R = delta_ln_M_star - tax_occam - log(W_eff * q_tau)

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
