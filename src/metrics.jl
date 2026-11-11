# metrics.jl - Time-Rescaling KS, R-hat, ESS & Joint Posterior Energy

# =====================================================================
# 12. TIME-RESCALING THEOREM (Brown et al. 2002) & MULTI-CHAIN CONVERGENCE (Vehtari et al. 2021)
# =====================================================================

@inline function standard_normal_cdf(x::Float64)::Float64
    return 0.5 * (1.0 + erf(x * 0.7071067811865475))
end


"""
    select_stratified_audit_channels(motifs, bg_rates, N_channels; n_signal=10, n_quiet=10)

Selects a stratified set of audit channels:
- n_signal channels with the highest packet density and motif participation (tests burst timing).
- n_quiet channels with zero packet assignments and baseline tonic rates (tests Poisson background fidelity).
"""
function select_stratified_audit_channels(
    motifs::Dict{Int, PurePacketMotif},
    bg_rates::Vector{Float64},
    N_channels::Int;
    n_signal::Int=10,
    n_quiet::Int=10
)
    chan_pkt_counts = zeros(Int, N_channels)
    for (mid, m) in motifs
        for k in 1:N_channels
            chan_pkt_counts[k] += length(m.chan_to_packets[k])
        end
    end

    signal_chans = sortperm(chan_pkt_counts, rev=true)[1:min(n_signal, N_channels)]
    quiet_candidates = filter(k -> chan_pkt_counts[k] == 0 && bg_rates[k] > 0.1, 1:N_channels)
    if length(quiet_candidates) < n_quiet
        quiet_candidates = sortperm(chan_pkt_counts)[1:min(n_quiet, N_channels)]
    end
    quiet_chans = quiet_candidates[1:min(n_quiet, length(quiet_candidates))]

    return (signal_channels = signal_chans, quiet_channels = quiet_chans, all_audit = [signal_chans; quiet_chans])
end

"""
    compute_time_rescaling_ks(
        motifs, occurrences, spike_t, chan_to_spikes, bg_rates, audit_channels, active_mask
    )

Calculates the exact closed-form Time-Rescaling integral (Brown et al., 2002)
between consecutive spikes on each audit channel:
    tau_i = int_{t_{i-1}}^{t_i} lambda_k(s) ds
          = lambda_0(k) * (t_i - t_{i-1}) + sum_{(e,p)} c_e * a_{m,p} * [Phi((t_i - mu)/sigma) - Phi((t_{i-1} - mu)/sigma)]
Transforms: u_i = 1 - exp(-tau_i) ~ Uniform(0, 1).
Evaluates the Kolmogorov-Smirnov statistic D_KS against the 95% uniform threshold:
    D_95 = 1.36 / sqrt(N_audit).
Zero numerical quadrature: evaluated analytically with SpecialFunctions.erf.
"""
function compute_time_rescaling_ks(
    motifs::Dict{Int, PurePacketMotif},
    occurrences::Dict{Int, PureOccurrence},
    spike_t::Vector{Float64},
    chan_to_spikes::Vector{Vector{Int}},
    bg_rates::Vector{Float64},
    audit_channels::Vector{Int},
    active_mask::BitVector;
    ou_ctx::Union{Nothing, ContinuousOUContext}=nothing
)
    # Pre-gather active packet occurrences on each audit channel
    chan_instances = Dict{Int, Vector{AuditPacketInstance}}()
    for k in audit_channels
        insts = AuditPacketInstance[]
        for occ in values(occurrences)
            haskey(motifs, occ.motif) || continue
            m = motifs[occ.motif]
            pkts_on_k = m.chan_to_packets[k]
            isempty(pkts_on_k) && continue
            
            u = occ.u
            for p in pkts_on_k
                pkt = m.packets[p]
                pred_s = 0.0
                for r in 1:length(active_mask)
                    if active_mask[r]
                        pred_s += m.W[r, p] * u[r]
                    end
                end
                mu_pred = occ.tau + pkt.mu + pred_s
                sig = sqrt(pkt.var)
                tot_mass = occ.amplitude * pkt.mass
                push!(insts, AuditPacketInstance(mu_pred, sig, tot_mass))
            end
        end
        sort!(insts, by=x -> x.mu)
        chan_instances[k] = insts
    end

    all_u = Float64[]

    for k in audit_channels
        spk_indices = chan_to_spikes[k]
        length(spk_indices) < 2 && continue
        
        t_k = view(spike_t, spk_indices)
        N_k = length(t_k)
        lam0 = max(bg_rates[k], 1e-12)
        insts = chan_instances[k]
        N_inst = length(insts)
        inst_mus = [inst.mu for inst in insts]
        
        for i in 2:N_k
            t_prev = t_k[i-1]
            t_curr = t_k[i]
            dt = t_curr - t_prev
            if ou_ctx !== nothing
                tau_int = evaluate_integrated_background(ou_ctx, lam0, k, t_prev, t_curr)
            else
                tau_int = lam0 * dt
            end
            
            if N_inst > 0
                idx_start = searchsortedfirst(inst_mus, t_prev - 0.04)
                idx_end = searchsortedlast(inst_mus, t_curr + 0.04)
                for j in idx_start:idx_end
                    inst = insts[j]
                    dphi = standard_normal_cdf((t_curr - inst.mu) / inst.sigma) - standard_normal_cdf((t_prev - inst.mu) / inst.sigma)
                    if dphi > 0.0
                        tau_int += inst.mass * dphi
                    end
                end
            end
            
            u_val = 1.0 - exp(-tau_int)
            push!(all_u, clamp(u_val, 1e-12, 1.0 - 1e-12))
        end
    end

    N_audit = length(all_u)
    if N_audit == 0
        return (D_ks = 1.0, D_95 = 1.0, is_valid = false, N_intervals = 0)
    end

    sort!(all_u)
    D_ks = 0.0
    for (j, u_val) in enumerate(all_u)
        f_emp = j / N_audit
        f_emp_prev = (j - 1) / N_audit
        d = max(abs(f_emp - u_val), abs(f_emp_prev - u_val))
        if d > D_ks
            D_ks = d
        end
    end

    D_95 = 1.36 / sqrt(N_audit)
    is_valid = D_ks <= D_95

    return (D_ks = D_ks, D_95 = D_95, is_valid = is_valid, N_intervals = N_audit)
end

"""
    compute_rank_normalized_rhat_ess(chains::Vector{Vector{Float64}})

Computes the Rank-Normalized Potential Scale Reduction Factor (split-R_hat)
and Bulk Effective Sample Size (ESS) on MCMC chain values (e.g. Joint Energy U_t)
following Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021).
Splits each chain into 2 halves to detect both between-chain non-mixing and within-chain non-stationarity.
"""
function compute_rank_normalized_rhat_ess(chains::Vector{Vector{Float64}})
    C = length(chains)
    C < 1 && return (rhat = 1.0, ess = 0.0)
    
    # Split each chain into two halves
    split_chains = Vector{Vector{Float64}}()
    for chain in chains
        N_c = length(chain)
        N_c < 4 && continue
        half = div(N_c, 2)
        push!(split_chains, chain[1:half])
        push!(split_chains, chain[(half+1):(2*half)])
    end
    
    M = length(split_chains)
    M < 2 && return (rhat = 1.0, ess = Float64(sum(length.(chains))))
    N = minimum(length.(split_chains))
    
    # Truncate to common length N
    for m in 1:M
        split_chains[m] = split_chains[m][1:N]
    end
    
    # Pool all values and compute normalized ranks
    S = M * N
    flat_vals = zeros(Float64, S)
    idx = 0
    for m in 1:M
        for n in 1:N
            idx += 1
            flat_vals[idx] = split_chains[m][n]
        end
    end
    
    p = sortperm(flat_vals)
    ranks = zeros(Float64, S)
    for (r, flat_i) in enumerate(p)
        ranks[flat_i] = Float64(r)
    end
    
    # Blomblom transformation to standard normal quantiles
    norm_ranks = zeros(Float64, M, N)
    idx = 0
    for m in 1:M
        for n in 1:N
            idx += 1
            u = (ranks[idx] - 0.375) / (S + 0.25)
            norm_ranks[m, n] = quantile(Normal(), u)
        end
    end
    
    # Between-chain and Within-chain variance
    chain_means = vec(mean(norm_ranks, dims=2))
    grand_mean = mean(chain_means)
    
    B = (Float64(N) / Float64(M - 1)) * sum((chain_means .- grand_mean).^2)
    s2 = zeros(Float64, M)
    for m in 1:M
        s2[m] = (1.0 / (N - 1)) * sum((norm_ranks[m, :] .- chain_means[m]).^2)
    end
    W = mean(s2)
    
    var_plus = ((N - 1) / N) * W + (1.0 / N) * B
    rhat = sqrt(max(var_plus / max(W, 1e-12), 1.0))
    
    # Geyer's initial positive sequence estimator for Bulk-ESS
    max_lag = min(N - 2, 25)
    rho = zeros(Float64, max_lag)
    for k in 1:max_lag
        v_k = 0.0
        for m in 1:M
            m_mean = chain_means[m]
            for n in 1:(N - k)
                v_k += (norm_ranks[m, n] - m_mean) * (norm_ranks[m, n + k] - m_mean)
            end
        end
        cov_k = v_k / (M * (N - k))
        rho[k] = 1.0 - (W - cov_k) / max(var_plus, 1e-12)
    end
    
    tau_int = 1.0
    for k in 1:2:(max_lag - 1)
        G_k = rho[k] + rho[k + 1]
        G_k <= 0.0 && break
        tau_int += 2.0 * G_k
    end
    
    ess = Float64(S) / max(tau_int, 1.0)
    return (rhat = rhat, ess = ess)
end

"""
    compute_joint_energy(
        motifs::Dict{Int, PurePacketMotif},
        occurrences::Dict{Int, PureOccurrence},
        total_spikes_per_chan::Vector{Int},
        bg_rates::Vector{Float64},
        ard_state::ARDManifoldState,
        hyperstate::Level5Hyperstate,
        T_total::Float64;
        alpha_a::Float64 = 1.0,
        beta_a::Float64 = 1.0
    ) -> Float64

Computes the exact negative complete-data joint log-posterior energy:
    U_t = -ln p(X, Z | Theta) - ln p(Theta)
for the Level m=0..5 Pure Packet Engine. Evaluates in sub-millisecond time
directly from occurrence summary moments (counts, sum_t, sum_t2) without raw spike loops.
"""
function compute_joint_energy(
    motifs::Dict{Int, PurePacketMotif},
    occurrences::Dict{Int, PureOccurrence},
    total_spikes_per_chan::Vector{Int},
    bg_rates::Vector{Float64},
    ard_state::ARDManifoldState,
    T_total::Float64;
    ou_ctx::Union{Nothing, ContinuousOUContext}=nothing,
    alpha_a::Float64 = 1.0,
    beta_a::Float64 = 1.0
)
    N_channels = length(bg_rates)
    log_lik = 0.0
    log_prior = 0.0

    # 1. Assigned Spikes per Channel from Occurrence Moments
    assigned_spikes_per_chan = zeros(Int, N_channels)
    for occ in values(occurrences)
        haskey(motifs, occ.motif) || continue
        pm = motifs[occ.motif]
        P = length(pm.packets)
        for p in 1:min(P, length(occ.counts))
            c = occ.counts[p]
            if c > 0
                k = pm.packets[p].channel
                if k >= 1 && k <= N_channels
                    assigned_spikes_per_chan[k] += c
                end
            end
        end
    end

    # 2. Background Log-Likelihood & Compensator
    for k in 1:N_channels
        lam0 = max(bg_rates[k], 1e-12)
        n_bg = max(0, total_spikes_per_chan[k] - assigned_spikes_per_chan[k])
        log_lik += n_bg * log(lam0) - lam0 * T_total
    end

    # 3. Foreground Log-Likelihood & Compensator
    for occ in values(occurrences)
        haskey(motifs, occ.motif) || continue
        pm = motifs[occ.motif]
        P = length(pm.packets)
        c_e = occ.amplitude
        tau_e = occ.tau
        u_e = occ.u

        # Prior on onset tau_e ~ Uniform(0, T_total)
        log_prior += -log(max(T_total, 1.0))

        # Prior on amplitude c_e ~ Gamma(alpha_c, beta_c) under continuous vigor anchor E[c_e] = 1.0
        a_c = ou_ctx !== nothing ? ou_ctx.alpha_c : 10.0
        b_c = a_c
        log_prior += a_c * log(b_c) - loggamma(a_c) + (a_c - 1.0) * log(max(c_e, 1e-12)) - b_c * c_e

        # Prior on latent kinematics u_e ~ N(0, tau_e^-1 I)
        tau_prec = ou_ctx !== nothing ? get(ou_ctx.tau_precisions, occ.id, 1.0) : 1.0
        R_dim = ard_state.R_max
        for r in 1:min(R_dim, length(ard_state.active_mask))
            if ard_state.active_mask[r] && r <= length(u_e)
                lam_r = tau_prec
                log_prior += 0.5 * log(lam_r) - 0.5 * log(2π) - 0.5 * lam_r * u_e[r]^2
            end
        end

        # Foreground Emission Log-Likelihood & Compensator
        for p in 1:min(P, length(occ.counts))
            pkt = pm.packets[p]
            c = occ.counts[p]
            var_p = pkt.var
            mass_p = pkt.mass

            # Shift mu_eff
            shift = 0.0
            R_m = size(pm.W, 1)
            for r in 1:min(R_m, length(ard_state.active_mask))
                if ard_state.active_mask[r] && r <= length(u_e)
                    shift += pm.W[r, p] * u_e[r]
                end
            end
            mu_eff = tau_e + pkt.mu + shift

            if c > 0
                ss = occ.sum_t2[p] - 2.0 * mu_eff * occ.sum_t[p] + c * mu_eff * mu_eff
                term = c * log(max(c_e * mass_p, 1e-12)) - 0.5 * c * log(2π * var_p) - 0.5 * ss / var_p
                log_lik += term
            end
            # Compensator for packet emission
            log_lik -= c_e * mass_p
        end
    end

    # 4. Prior on Motif Parameters: W and a_{m,p}
    for pm in values(motifs)
        P = length(pm.packets)
        for p in 1:P
            pkt = pm.packets[p]
            # Prior on mass a_{m,p} ~ Gamma(alpha_a, beta_a)
            log_prior += alpha_a * log(beta_a) - loggamma(alpha_a) + (alpha_a - 1.0) * log(max(pkt.mass, 1e-12)) - beta_a * pkt.mass

            # Prior on loadings W_{r,p} ~ N(0, alpha_r^-1)
            R_m = size(pm.W, 1)
            for r in 1:min(R_m, length(ard_state.active_mask))
                if ard_state.active_mask[r]
                    alpha_r = max(ard_state.alpha[r], 1e-12)
                    log_prior += 0.5 * log(alpha_r) - 0.5 * log(2π) - 0.5 * alpha_r * pm.W[r, p]^2
                end
            end
        end
    end

    joint_energy = -(log_lik + log_prior)
    return joint_energy
end

"""
    compute_joint_energy(motifs, occurrences, total_spikes_per_chan, bg_rates, ard_state, legacy_arg, T_total; kwargs...)

Backward compatibility overload accepting the legacy 7-argument signature (ignoring `legacy_arg` like `hyperstate`).
"""
function compute_joint_energy(
    motifs::Dict{Int, PurePacketMotif},
    occurrences::Dict{Int, PureOccurrence},
    total_spikes_per_chan::Vector{Int},
    bg_rates::Vector{Float64},
    ard_state::ARDManifoldState,
    legacy_arg,
    T_total::Float64;
    kwargs...
)
    return compute_joint_energy(motifs, occurrences, total_spikes_per_chan, bg_rates, ard_state, T_total; kwargs...)
end

