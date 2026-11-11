# ctmc.jl - Discrete Regimes & Continuous-Time CTMC Uniformization

"""
    compact_regimes!(hyperstate::Level5Hyperstate, occurrences)

Re-indexes active regimes to contiguous integers 1...K, keeping Regime 1 fixed as the anchor.
Resizes pi_m to M x K if the number of regimes changed.
"""
function compact_regimes!(
    hyperstate::Level5Hyperstate,
    occurrences::Union{AbstractVector{PureOccurrence}, Dict{Int, PureOccurrence}}
)
    occ_dict = occurrences isa Dict ? occurrences : Dict(occ.id => occ for occ in occurrences)
    old_keys = sort(collect(keys(hyperstate.regimes)))
    # 1 is always the anchor
    id_map = Dict{Int, Int}()
    id_map[1] = 1
    next_id = 2
    for k in old_keys
        k == 1 && continue
        id_map[k] = next_id
        next_id += 1
    end

    needs_reindex = !(length(id_map) == length(old_keys) && all(id_map[k] == k for k in old_keys))

    if needs_reindex
        new_regimes = Dict{Int, TaskRegime}()
        for k in old_keys
            new_k = id_map[k]
            reg = hyperstate.regimes[k]
            reg.id = new_k
            new_regimes[new_k] = reg
        end
        hyperstate.regimes = new_regimes
        hyperstate.K = length(new_regimes)

        for occ in values(occ_dict)
            if haskey(id_map, occ.regime_id)
                occ.regime_id = id_map[occ.regime_id]
            else
                occ.regime_id = 1
            end
        end
    else
        hyperstate.K = length(hyperstate.regimes)
    end

    # Resize pi_m if needed (always ensure columns match hyperstate.K)
    M_rows = size(hyperstate.pi_m, 1)
    if size(hyperstate.pi_m, 2) != hyperstate.K
        new_pi = fill(1.0 / hyperstate.K, M_rows, hyperstate.K)
        for m in 1:M_rows
            for (old_k, new_k) in id_map
                if old_k <= size(hyperstate.pi_m, 2) && new_k <= hyperstate.K
                    new_pi[m, new_k] = hyperstate.pi_m[m, old_k]
                end
            end
            s = sum(new_pi[m, :])
            if s > 0.0
                new_pi[m, :] ./= s
            end
        end
        hyperstate.pi_m = new_pi
    end
end


"""
    sample_dp_concentration!(hyperstate::Level5Hyperstate; rng=Random.default_rng())

Escobar-West (1995) Gibbs sampler for Dirichlet Process concentration parameter alpha_dp.
Given E occurrences and K active regimes:
1. Draws auxiliary variable eta ~ Beta(alpha_dp + 1, E).
2. Computes mixture probability pi_eta = (a_dp + K - 1) / (a_dp + K - 1 + E * (b_dp - log(eta))).
3. Draws alpha_dp from two-component Gamma mixture:
   with probability pi_eta: alpha_dp ~ Gamma(a_dp + K, 1 / (b_dp - log(eta)))
   otherwise:               alpha_dp ~ Gamma(a_dp + K - 1, 1 / (b_dp - log(eta)))
"""
function sample_dp_concentration!(
    hyperstate::Level5Hyperstate;
    rng::AbstractRNG=Random.default_rng()
)
    E = sum(length(reg.assigned_occurrences) for reg in values(hyperstate.regimes); init=0)
    K = length(hyperstate.regimes)
    (E <= 0 || K <= 0) && return hyperstate.alpha_dp

    # 1. Auxiliary beta draw
    eta = rand(rng, Beta(hyperstate.alpha_dp + 1.0, Float64(E)))
    log_eta = log(max(eta, 1e-12))
    rate_term = hyperstate.b_dp - log_eta

    # 2. Mixture weight
    num = hyperstate.a_dp + Float64(K) - 1.0
    denom = num + Float64(E) * rate_term
    pi_eta = denom > 0.0 ? (num / denom) : 0.5

    # 3. Two-component Gamma draw
    shape = rand(rng) < pi_eta ? (hyperstate.a_dp + Float64(K)) : (hyperstate.a_dp + Float64(K) - 1.0)
    shape = max(shape, 0.1)
    hyperstate.alpha_dp = max(0.01, min(50.0, rand(rng, Gamma(shape, 1.0 / max(rate_term, 1e-12)))))
    return hyperstate.alpha_dp
end

"""
    maintain_majority_scale_gauge!(
        hyperstate::Level5Hyperstate,
        occurrences::Union{AbstractVector{PureOccurrence}, Dict{Int, PureOccurrence}},
        motifs::Dict{Int, PurePacketMotif},
        ard_state::ARDManifoldState
    )

Exact majority scale gauge maintenance and anchor handover:
1. Identifies the regime holding the majority of occurrences: k_max = argmax_k |E_k|.
2. If k_max != 1 and |E_k_max| > |E_1|:
   - Rescales kinematic coordinates: u_{e,r} <- u_{e,r} * sqrt(lambda_{k_max, r})
   - Rescales motif loadings: W_{r,p} <- W_{r,p} / sqrt(lambda_{k_max, r})
   - Rescales ARD precisions: alpha_r <- alpha_r * lambda_{k_max, r} (preserves Level 4 prior registration)
   - Rescales secondary kinematic precisions: lambda_{k,r} <- lambda_{k,r} / lambda_{k_max, r}
   - Rescales amplitudes and packet masses:
     c_bar = alpha_{c, k_max} / beta_{c, k_max}
     c_e <- c_e / c_bar
     a_{m,p} <- a_{m,p} * c_bar
     beta_{c,k} <- beta_{c,k} * c_bar (so beta_{c,1} = alpha_{c,1})
   - Swaps labels so k_max becomes Regime 1 (lambda_{1,r} == 1.0, beta_{c,1} == alpha_{c,1}).
"""
function maintain_majority_scale_gauge!(
    hyperstate::Level5Hyperstate,
    occurrences::Union{AbstractVector{PureOccurrence}, Dict{Int, PureOccurrence}},
    motifs::Dict{Int, PurePacketMotif},
    ard_state::ARDManifoldState
)
    K = length(hyperstate.regimes)
    K <= 1 && return 1

    # Find regime with maximum occurrences
    k_max = 1
    max_count = haskey(hyperstate.regimes, 1) ? length(hyperstate.regimes[1].assigned_occurrences) : -1
    for (k, reg) in hyperstate.regimes
        cnt = length(reg.assigned_occurrences)
        if cnt > max_count
            max_count = cnt
            k_max = k
        end
    end

    if k_max == 1 || (haskey(hyperstate.regimes, 1) && max_count <= length(hyperstate.regimes[1].assigned_occurrences))
        return 1
    end

    reg_max = hyperstate.regimes[k_max]
    R = hyperstate.R
    occ_dict = occurrences isa Dict ? occurrences : Dict(occ.id => occ for occ in occurrences)

    # 1. Rescale coordinates u_{e,r} <- u_{e,r} * sqrt(lambda_{k_max, r})
    scale_fac = [sqrt(max(reg_max.lambda_u[r], 1e-6)) for r in 1:R]
    inv_scale_fac = [1.0 / scale_fac[r] for r in 1:R]

    for occ in values(occ_dict)
        for r in 1:R
            occ.u[r] *= scale_fac[r]
        end
    end

    # 2. Rescale motif loadings W_{r,p} <- W_{r,p} / sqrt(lambda_{k_max, r})
    for pm in values(motifs)
        P = length(pm.packets)
        for r in 1:R
            for p in 1:P
                pm.W[r, p] *= inv_scale_fac[r]
            end
        end
    end

    # 3. Rescale ARD precisions alpha_r <- alpha_r * lambda_{k_max, r}
    for r in 1:R
        ard_state.alpha[r] *= (scale_fac[r] * scale_fac[r])
    end

    # 4. Rescale kinematic precisions lambda_{k,r} <- lambda_{k,r} / lambda_{k_max, r}
    for (k, reg) in hyperstate.regimes
        if k != k_max
            for r in 1:R
                if reg.active_dims[r]
                    reg.lambda_u[r] /= (scale_fac[r] * scale_fac[r])
                end
            end
        end
    end
    reg_max.lambda_u .= 1.0
    reg_max.active_dims .= true

    # 5. Rescale Motor Vigor & Packet Masses
    c_bar = reg_max.alpha_c / max(reg_max.beta_c, 1e-12)
    inv_c_bar = 1.0 / c_bar

    for occ in values(occ_dict)
        occ.amplitude *= inv_c_bar
    end

    for pm in values(motifs)
        for pkt in pm.packets
            pkt.mass *= c_bar
        end
    end

    for (k, reg) in hyperstate.regimes
        reg.beta_c *= c_bar
    end
    reg_max.beta_c = reg_max.alpha_c

    # 6. Swap Regime 1 and Regime k_max labels
    reg1 = hyperstate.regimes[1]

    # Swap occurrence regime_ids
    for eid in reg1.assigned_occurrences
        if haskey(occ_dict, eid)
            occ_dict[eid].regime_id = k_max
        end
    end
    for eid in reg_max.assigned_occurrences
        if haskey(occ_dict, eid)
            occ_dict[eid].regime_id = 1
        end
    end

    # Swap ids in structs
    reg1.id = k_max
    reg_max.id = 1
    hyperstate.regimes[1] = reg_max
    hyperstate.regimes[k_max] = reg1

    # Swap mark distribution columns in pi_m
    M_rows = size(hyperstate.pi_m, 1)
    if k_max <= size(hyperstate.pi_m, 2)
        for m in 1:M_rows
            tmp = hyperstate.pi_m[m, 1]
            hyperstate.pi_m[m, 1] = hyperstate.pi_m[m, k_max]
            hyperstate.pi_m[m, k_max] = tmp
        end
    end

    return k_max
end

"""
    resample_occurrence_regimes!(
        hyperstate::Level5Hyperstate,
        occurrences::Union{AbstractVector{PureOccurrence}, Dict{Int, PureOccurrence}},
        motifs::Dict{Int, PurePacketMotif};
        rng::AbstractRNG=Random.default_rng()
    )

Step 5A: Non-parametric Dynamic Occurrence Regime Assignment (Neal 2000, Algorithm 8).
Sequential leave-one-out sampling with closed-form Student-t (G_0 kinematics) and Negative Binomial (vigor).
Includes:
1. Sequential leave-one-out updates: instant cluster visibility, zero redundant singleton spawning.
2. Stiffened base measure G_0 (a_u = b_u = 5.0) suppressing Student-t heavy tail outlier seduction.
3. Candidate staging & categorical dissolution: lone singletons (|E_k| < N_min) dissolved via Categorical redraw over surviving regimes.
4. Escobar-West Gibbs update for alpha_dp.
5. Contiguous regime compaction.
"""
function resample_occurrence_regimes!(
    hyperstate::Level5Hyperstate,
    occurrences::Union{AbstractVector{PureOccurrence}, Dict{Int, PureOccurrence}},
    motifs::Dict{Int, PurePacketMotif};
    rng::AbstractRNG=Random.default_rng()
)
    R = hyperstate.R
    occ_list = occurrences isa Dict ? collect(values(occurrences)) : collect(occurrences)
    E = length(occ_list)
    E == 0 && return

    # Ensure assigned_occurrences is accurate initially
    for reg in values(hyperstate.regimes)
        empty!(reg.assigned_occurrences)
    end
    for occ in occ_list
        if !haskey(hyperstate.regimes, occ.regime_id)
            occ.regime_id = 1
        end
        push!(hyperstate.regimes[occ.regime_id].assigned_occurrences, occ.id)
    end

    motif_masses = Dict{Int, Float64}()
    for (mid, pm) in motifs
        motif_masses[mid] = sum(p.mass for p in pm.packets; init=0.0)
    end

    LN_2PI = 1.8378770664093453 # ln(2*pi)
    a_u = hyperstate.a_u
    b_u = hyperstate.b_u
    # Precompute G0 Student-t normalizer for kinematics:
    # ln(Gamma(a_u + 0.5)) - ln(Gamma(a_u)) - 0.5 * ln(2*pi*b_u)
    lgamma_a_plus_half = loggamma(a_u + 0.5)
    lgamma_a = loggamma(a_u)
    g0_kin_const = lgamma_a_plus_half - lgamma_a - 0.5 * log(2.0 * pi * b_u)

    # Base measure parameters for vigor:
    a_c = hyperstate.a_c
    b_c = hyperstate.b_c
    alpha_c_base = 10.0
    g0_vigor_const = loggamma(alpha_c_base + a_c) - loggamma(alpha_c_base) - loggamma(a_c) + a_c * log(b_c)
    lgamma_alpha_c = loggamma(alpha_c_base)

    # Sequential leave-one-out order
    occ_order = randperm(rng, E)

    for idx in occ_order
        occ = occ_list[idx]
        k_old = occ.regime_id
        mid = occ.motif
        haskey(motifs, mid) || continue
        pm = motifs[mid]
        A_m = get(motif_masses, mid, 1.0)
        N_e = sum(occ.counts; init=0)
        c_e = max(occ.amplitude, 1e-6)
        log_c = log(c_e)

        # 1. Leave-One-Out Decrement
        reg_old = hyperstate.regimes[k_old]
        filter!(id -> id != occ.id, reg_old.assigned_occurrences)
        N_minus_old = length(reg_old.assigned_occurrences)

        # Build candidate list
        # If N_minus_old == 0: reg_old was a singleton; it temporarily occupies the "new" slot
        cand_keys = Int[]
        cand_weights = Float64[]

        for (k, reg) in hyperstate.regimes
            if k == k_old && N_minus_old == 0
                # Do not treat k_old as an existing regime; it IS the candidate "new" slot
                continue
            end
            N_k = length(reg.assigned_occurrences)
            w_mark = (1 <= mid <= size(hyperstate.pi_m, 1) && k <= size(hyperstate.pi_m, 2)) ? hyperstate.pi_m[mid, k] : (1.0 / max(1, hyperstate.K))
            w_k = Float64(N_k) + hyperstate.alpha_pi * w_mark
            if w_k > 0.0
                push!(cand_keys, k)
                push!(cand_weights, w_k)
            end
        end

        # Total candidates: existing regimes + 1 "new" proposal
        n_cands = length(cand_keys) + 1
        log_probs = zeros(Float64, n_cands)

        # Evaluate existing candidates
        for c in 1:(n_cands - 1)
            k = cand_keys[c]
            reg = hyperstate.regimes[k]
            # Prior: ln weight
            lp_prior = log(max(cand_weights[c], 1e-12))

            # Kinematic: ln P(u_e | lambda_k)
            lp_kin = -0.5 * R * LN_2PI
            for r in 1:R
                lam = reg.lambda_u[r]
                lp_kin += 0.5 * (log(max(lam, 1e-12)) - lam * occ.u[r]^2)
            end

            # Vigor: Exact continuous Gamma density over c_e under regime motor vigor (alpha_k, beta_k)
            alpha_k = reg.alpha_c
            beta_k = reg.beta_c
            lp_vigor = alpha_k * log(beta_k) - lgamma_alpha_c + (alpha_k - 1.0) * log_c - beta_k * c_e

            log_probs[c] = lp_prior + lp_kin + lp_vigor
        end

        # Evaluate candidate "new" against G_0
        # Prior: ln alpha_dp
        lp_prior_new = log(max(hyperstate.alpha_dp, 1e-12))

        # Kinematic under stiffened base measure G_0 (Student-t)
        lp_kin_new = 0.0
        for r in 1:R
            lp_kin_new += g0_kin_const - (a_u + 0.5) * log(1.0 + occ.u[r]^2 / (2.0 * b_u))
        end

        # Vigor under base measure G_0: closed-form compound Gamma marginal integrating over beta_c ~ Gamma(a_c, b_c)
        lp_vigor_new = g0_vigor_const + (alpha_c_base - 1.0) * log_c - (alpha_c_base + a_c) * log(b_c + c_e)

        log_probs[n_cands] = lp_prior_new + lp_kin_new + lp_vigor_new

        # Softmax draw
        max_lp = maximum(log_probs)
        sum_exp = 0.0
        for c in 1:n_cands
            log_probs[c] = exp(log_probs[c] - max_lp)
            sum_exp += log_probs[c]
        end
        u_rand = rand(rng) * sum_exp
        cum = 0.0
        chosen_c = n_cands
        for c in 1:n_cands
            cum += log_probs[c]
            if cum >= u_rand
                chosen_c = c
                break
            end
        end

        if chosen_c < n_cands
            # Joined existing candidate k
            chosen_k = cand_keys[chosen_c]
            occ.regime_id = chosen_k
            push!(hyperstate.regimes[chosen_k].assigned_occurrences, occ.id)

            # If k_old became empty (N_minus_old == 0) and wasn't chosen, delete it if k_old != 1
            if N_minus_old == 0 && k_old != 1 && k_old != chosen_k
                delete!(hyperstate.regimes, k_old)
            end
        else
            # Chose "new"!
            if N_minus_old == 0
                # Re-use k_old (no redundant new regime allocated!)
                occ.regime_id = k_old
                push!(hyperstate.regimes[k_old].assigned_occurrences, occ.id)
                # Initialize precisions from conjugate posterior given this occurrence's evidence
                for r in 1:R
                    alpha_post = a_u + 0.5
                    beta_post = b_u + 0.5 * occ.u[r]^2
                    hyperstate.regimes[k_old].lambda_u[r] = rand(rng, Gamma(alpha_post, 1.0 / max(1e-12, beta_post)))
                    hyperstate.regimes[k_old].active_dims[r] = true
                end
                alpha_c_post = hyperstate.a_c + 10.0
                beta_c_post = hyperstate.b_c + occ.amplitude
                hyperstate.regimes[k_old].beta_c = max(0.1, rand(rng, Gamma(alpha_c_post, 1.0 / max(1e-12, beta_c_post))))
            else
                # Allocate brand new regime K_new
                new_k = isempty(hyperstate.regimes) ? 1 : (maximum(keys(hyperstate.regimes)) + 1)
                new_reg = TaskRegime(new_k, R; alpha_c=10.0, beta_c=10.0, lambda_dormant=hyperstate.lambda_dormant)
                for r in 1:R
                    alpha_post = a_u + 0.5
                    beta_post = b_u + 0.5 * occ.u[r]^2
                    new_reg.lambda_u[r] = rand(rng, Gamma(alpha_post, 1.0 / max(1e-12, beta_post)))
                    new_reg.active_dims[r] = true
                end
                alpha_c_post = hyperstate.a_c + 10.0
                beta_c_post = hyperstate.b_c + occ.amplitude
                new_reg.beta_c = max(0.1, rand(rng, Gamma(alpha_c_post, 1.0 / max(1e-12, beta_c_post))))
                push!(new_reg.assigned_occurrences, occ.id)
                hyperstate.regimes[new_k] = new_reg
                occ.regime_id = new_k
            end
        end
    end

    # Step 2: Sweep-End Candidate Staging (N_min >= 2) & Categorical Dissolution
    # Dissolve lone singletons (|E_k| < N_min) for k != 1
    dissolved_keys = Int[]
    for (k, reg) in hyperstate.regimes
        k == 1 && continue
        if length(reg.assigned_occurrences) < hyperstate.N_min
            push!(dissolved_keys, k)
        end
    end

    # Surviving regimes
    surviving_keys = [k for k in keys(hyperstate.regimes) if !(k in dissolved_keys)]
    if isempty(surviving_keys)
        push!(surviving_keys, 1)
    end

    for k in dissolved_keys
        reg = hyperstate.regimes[k]
        for eid in reg.assigned_occurrences
            occ_match = findfirst(o -> o.id == eid, occ_list)
            occ_match === nothing && continue
            occ = occ_list[occ_match]

            # Categorical redraw over surviving regimes with P(new) = 0
            n_surv = length(surviving_keys)
            lps = zeros(Float64, n_surv)
            mid = occ.motif
            A_m = get(motif_masses, mid, 1.0)
            N_e = sum(occ.counts; init=0)

            for s in 1:n_surv
                sk = surviving_keys[s]
                sreg = hyperstate.regimes[sk]
                lp_p = log(max(Float64(length(sreg.assigned_occurrences)), 1.0))
                lp_k = -0.5 * R * LN_2PI
                for r in 1:R
                    lam = sreg.lambda_u[r]
                    lp_k += 0.5 * (log(max(lam, 1e-12)) - lam * occ.u[r]^2)
                end
                alpha_k = sreg.alpha_c
                beta_k = sreg.beta_c
                lp_v = loggamma(alpha_k + N_e) - loggamma(alpha_k) - N_e * log(beta_k + A_m) + alpha_k * log(beta_k / (beta_k + A_m))
                lps[s] = lp_p + lp_k + lp_v
            end

            max_lp = maximum(lps)
            sum_exp = sum(exp(lp - max_lp) for lp in lps)
            u_rand = rand(rng) * sum_exp
            cum = 0.0
            chosen_sk = surviving_keys[end]
            for s in 1:n_surv
                cum += exp(lps[s] - max_lp)
                if cum >= u_rand
                    chosen_sk = surviving_keys[s]
                    break
                end
            end

            occ.regime_id = chosen_sk
            push!(hyperstate.regimes[chosen_sk].assigned_occurrences, occ.id)
        end
        delete!(hyperstate.regimes, k)
    end

    # Step 3: Escobar-West DP concentration sampling
    sample_dp_concentration!(hyperstate; rng=rng)

    # Step 4: Compact regimes so IDs are contiguous 1...K
    compact_regimes!(hyperstate, occurrences)
end

"""
    sample_task_regimes!(
        hyperstate::Level5Hyperstate,
        occurrences::Union{AbstractVector{PureOccurrence}, Dict{Int, PureOccurrence}},
        motifs::Dict{Int, PurePacketMotif};
        T_total::Float64=2114.0,
        rng::AbstractRNG=Random.default_rng()
    )

Conjugate updates for:
- Step 5B: Kinematic precisions lambda_u & unregularized data MLE dispersion check S^{data}_{k,r} > 0.05
- Step 5C: Motor vigor (alpha_c, beta_c) with Regime 1 unit anchor
- Step 5D: Session motif Poisson arrival rates lambda_m & Dirichlet mark distributions pi_m
"""
function sample_task_regimes!(
    hyperstate::Level5Hyperstate,
    occurrences::Union{AbstractVector{PureOccurrence}, Dict{Int, PureOccurrence}},
    motifs::Dict{Int, PurePacketMotif};
    T_total::Float64=2114.0,
    rng::AbstractRNG=Random.default_rng()
)
    K = hyperstate.K
    R = hyperstate.R
    M = hyperstate.M
    occ_dict = occurrences isa Dict ? occurrences : Dict(occ.id => occ for occ in occurrences)

    # Defensive resizing: ensure pi_m columns match hyperstate.K and rows match M
    if size(hyperstate.pi_m, 2) != K || size(hyperstate.pi_m, 1) < M
        M_rows = max(size(hyperstate.pi_m, 1), M)
        new_pi = fill(1.0 / K, M_rows, K)
        for m in 1:min(M_rows, size(hyperstate.pi_m, 1))
            for k in 1:min(K, size(hyperstate.pi_m, 2))
                new_pi[m, k] = hyperstate.pi_m[m, k]
            end
            s = sum(new_pi[m, :])
            if s > 0.0
                new_pi[m, :] ./= s
            end
        end
        hyperstate.pi_m = new_pi
    end

    # Step 5B: Kinematic Precision Updates (lambda_k)
    # Enforce Invariant: Regime 1 is the canonical scale anchor
    reg1 = hyperstate.regimes[1]
    reg1.lambda_u .= 1.0
    reg1.active_dims .= true

    for k in 2:K
        haskey(hyperstate.regimes, k) || continue
        reg = hyperstate.regimes[k]
        E_k = length(reg.assigned_occurrences)
        if E_k == 0
            continue
        end

        for r in 1:R
            # Compute unregularized data MLE coordinate:
            # u_hat_data = b_{e,r} / (sum_p kappa_{e,p} W_{r,p}^2)
            # S^{data}_{k,r} = (1 / |E_k|) sum_{e in E_k} (u_hat_data)^2
            sum_u_mle_sq = 0.0
            sum_u2 = 0.0

            for eid in reg.assigned_occurrences
                haskey(occ_dict, eid) || continue
                occ = occ_dict[eid]
                haskey(motifs, occ.motif) || continue
                pm = motifs[occ.motif]
                P = length(pm.packets)

                b_er = 0.0
                denom_er = 0.0
                for p in 1:P
                    c = occ.counts[p]
                    if c > 0
                        kap = c / pm.packets[p].var
                        mean_t = occ.sum_t[p] / c
                        res = mean_t - (occ.tau + pm.packets[p].mu)
                        w_rp = pm.W[r, p]
                        b_er += kap * res * w_rp
                        denom_er += kap * w_rp * w_rp
                    end
                end

                u_hat_data = denom_er > 1e-12 ? (b_er / denom_er) : 0.0
                sum_u_mle_sq += u_hat_data * u_hat_data
                sum_u2 += occ.u[r]^2
            end

            s_data = sum_u_mle_sq / Float64(E_k)

            if s_data <= hyperstate.force_thresh
                # Clamped to dormant floor
                reg.active_dims[r] = false
                reg.lambda_u[r] = hyperstate.lambda_dormant
            else
                # Active continuous conjugate Gamma update
                reg.active_dims[r] = true
                alpha_post = hyperstate.a_u + 0.5 * Float64(E_k)
                beta_post = hyperstate.b_u + 0.5 * sum_u2
                lam = rand(rng, Gamma(alpha_post, 1.0 / max(1e-12, beta_post)))
                reg.lambda_u[r] = max(1e-6, min(hyperstate.lambda_dormant, lam))
            end
        end
    end

    # Step 5C: Motor Vigor Updates ((alpha_c, beta_c))
    # Enforce Invariant: Regime 1 has beta_c == alpha_c (unit expectation anchor)
    reg1.beta_c = reg1.alpha_c

    for k in 2:K
        haskey(hyperstate.regimes, k) || continue
        reg = hyperstate.regimes[k]
        E_k = length(reg.assigned_occurrences)
        if E_k > 0
            sum_c = 0.0
            for eid in reg.assigned_occurrences
                haskey(occ_dict, eid) || continue
                sum_c += occ_dict[eid].amplitude
            end

            alpha_post = hyperstate.a_c + Float64(E_k) * reg.alpha_c
            beta_post = hyperstate.b_c + sum_c
            reg.beta_c = max(0.1, rand(rng, Gamma(alpha_post, 1.0 / max(1e-12, beta_post))))
        end
    end

    # Step 5D: Arrival Rates (lambda_m) & Mark Proportions (pi_m)
    E_m = zeros(Int, M)
    E_mk = zeros(Int, M, K)
    for occ in values(occ_dict)
        1 <= occ.motif <= M || continue
        E_m[occ.motif] += 1
        1 <= occ.regime_id <= K || continue
        E_mk[occ.motif, occ.regime_id] += 1
    end

    for m in 1:M
        hyperstate.lambda_m[m] = rand(rng, Gamma(hyperstate.a_rate + E_m[m], 1.0 / (hyperstate.b_rate + T_total)))

        dir_sum = 0.0
        for k in 1:K
            g = rand(rng, Gamma(hyperstate.alpha_pi + Float64(E_mk[m, k]), 1.0))
            hyperstate.pi_m[m, k] = g
            dir_sum += g
        end
        if dir_sum > 0.0
            for k in 1:K
                hyperstate.pi_m[m, k] /= dir_sum
            end
        end
    end
end


"""
    sample_occurrence_amplitudes!(occurrences, motifs, hyperstate::Level5Hyperstate; rng)

Conjugate Gamma draw for occurrence amplitudes c_e under regime motor vigor priors:
    c_e ~ Gamma(alpha_{c,k} + N_e, 1.0 / (beta_{c,k} + sum_{p} a_{m_e, p}))
where regime k = occ.regime_id specifies motor vigor prior (alpha_{c,k}, beta_{c,k}).
"""
function sample_occurrence_amplitudes!(
    occurrences::Union{AbstractVector{PureOccurrence}, Dict{Int, PureOccurrence}},
    motifs::Dict{Int, PurePacketMotif},
    hyperstate::Level5Hyperstate;
    rng::AbstractRNG=Random.default_rng()
)
    occ_iter = occurrences isa Dict ? values(occurrences) : occurrences
    motif_total_mass = Dict{Int, Float64}()
    for (mid, m) in motifs
        motif_total_mass[mid] = sum(p.mass for p in m.packets; init=0.0)
    end

    for occ in occ_iter
        reg = hyperstate.regimes[occ.regime_id]
        m_mass = get(motif_total_mass, occ.motif, 1.0)
        N_e = sum(occ.counts; init=0)
        shape_post = reg.alpha_c + Float64(N_e)
        rate_post = reg.beta_c + m_mass
        occ.amplitude = rand(rng, Gamma(shape_post, 1.0 / max(rate_post, 1e-12)))
    end
end


"""
    extend_task_regimes_dimension!(
        hyperstate::Level5Hyperstate,
        motifs::Dict{Int, PurePacketMotif},
        occurrences::Union{AbstractVector{PureOccurrence}, Dict{Int, PureOccurrence}},
        new_R::Int
    )

Dynamic Dimensional Growth Hook (R -> R+1):
When Level 4 detects structured residual covariance and instantiates dimension R+1,
extends lambda_u and active_dims across all regimes.
"""
function extend_task_regimes_dimension!(
    hyperstate::Level5Hyperstate,
    motifs::Dict{Int, PurePacketMotif},
    occurrences::Union{AbstractVector{PureOccurrence}, Dict{Int, PureOccurrence}},
    new_R::Int
)
    occ_dict = occurrences isa Dict ? occurrences : Dict(occ.id => occ for occ in occurrences)

    for (k, reg) in hyperstate.regimes
        reg.R = new_R
        if k == 1
            push!(reg.lambda_u, 1.0)
            push!(reg.active_dims, true)
        else
            E_k = length(reg.assigned_occurrences)
            if E_k == 0
                push!(reg.lambda_u, 1.0)
                push!(reg.active_dims, true)
                continue
            end

            sum_u_mle_sq = 0.0
            for eid in reg.assigned_occurrences
                haskey(occ_dict, eid) || continue
                occ = occ_dict[eid]
                haskey(motifs, occ.motif) || continue
                pm = motifs[occ.motif]
                P = length(pm.packets)
                b_er = 0.0
                denom_er = 0.0
                for p in 1:P
                    c = occ.counts[p]
                    if c > 0
                        kap = c / pm.packets[p].var
                        mean_t = occ.sum_t[p] / c
                        res = mean_t - (occ.tau + pm.packets[p].mu)
                        w_rp = pm.W[new_R, p]
                        b_er += kap * res * w_rp
                        denom_er += kap * w_rp * w_rp
                    end
                end
                u_mle = denom_er > 1e-12 ? (b_er / denom_er) : 0.0
                sum_u_mle_sq += u_mle * u_mle
            end
            s_data = sum_u_mle_sq / Float64(E_k)

            if s_data > hyperstate.force_thresh
                push!(reg.lambda_u, 1.0)
                push!(reg.active_dims, true)
            else
                push!(reg.lambda_u, hyperstate.lambda_dormant)
                push!(reg.active_dims, false)
            end
        end
    end
    hyperstate.R = new_R
end

"""
    compute_active_regime_min_precisions(hyperstate::Union{Level5Hyperstate, Nothing}, R::Int)

Computes the minimum precision lambda_{r,min} across all active regimes for each latent dimension r:
    lambda_{r,min} = min_{k in K_active} lambda_{k,r}
Guarantees that occurrences in high-flexibility / ballistic regimes (lambda_{k,r} < 1.0)
do not suffer tail truncation during continuous bounding envelope evaluation.
"""

function sample_diagonal_regimes!(
    regimes::Dict{Int, TaskRegime},
    occurrences::Vector{PureOccurrence},
    occ_regime_ids::Vector{Int},
    R_max::Int;
    rng::AbstractRNG=Random.default_rng()
)
    for (rid, regime) in regimes
        rid == 1 && continue
        E_k = 0
        S_u = zeros(Float64, R_max)
        for (e, occ) in enumerate(occurrences)
            if occ_regime_ids[e] == rid
                E_k += 1
                for r in 1:R_max
                    S_u[r] += occ.u[r]^2
                end
            end
        end
        if E_k > 0
            for r in 1:R_max
                regime.active_dims[r] || continue
                alpha_post = 2.0 + 0.5 * E_k
                beta_post = 2.0 + 0.5 * S_u[r]
                lam = rand(rng, Gamma(alpha_post, 1.0 / max(1e-12, beta_post)))
                regime.lambda_u[r] = lam
            end
        end
    end
end


"""
    resample_background_pure_packet!(
        bg_rates::Vector{Float64},
        pi_bg::Vector{Float64},
        gamma_buf::Vector{Float64},
        counts_bg::Vector{Int},
        assign::Vector{SpikeAssignment},
        neuron::Vector{Int},
        channel_group::Vector{Int},
        groups::Vector{Vector{Int}},
        bg_a::Vector{Float64},
        bg_b::Vector{Float64},
        beta_bg::Vector{Float64},
        T_total::Float64,
        exposure::Vector{Float64};
        rng::AbstractRNG=Random.default_rng()
    )

Zero-allocation in-place Dirichlet-Gamma background rate resampling.
1. Tallies unassigned spikes (assign[i].event_id == 0) directly into counts_bg[neuron[i]].
2. For each channel group g:
   Draws in-place Gamma samples into gamma_buf[k] ~ Gamma(beta_bg[g] + counts_bg[k], 1.0)
   Normalizes pi_bg[k] = gamma_buf[k] / sum(gamma_buf[k in group]) (exact Dirichlet draw with 0 heap allocations).
   Draws group rate Lambda_g ~ Gamma(bg_a[g] + sum(counts_bg[k in group]), 1.0 / (bg_b[g] + T_total * exposure[g])).
3. Updates bg_rates[k] = Lambda_g * pi_bg[k] in-place with machine-precision stability.
"""

"""
    sample_candidate_times!(ctmc::ContinuousTimeCTMC, T_total::Float64; rng)

Generates auxiliary candidate transition points T_cand ~ Poisson(Omega) on [0, T_total].
Returns sorted vector: [0.0, t_1, t_2, ..., t_B, T_total].
"""
function sample_candidate_times!(
    ctmc::ContinuousTimeCTMC, T_total::Float64;
    rng::AbstractRNG=Random.default_rng()
)
    N_cand = rand(rng, Poisson(ctmc.Omega * T_total))
    raw_times = rand(rng, N_cand) .* T_total
    sort!(raw_times)

    times = Vector{Float64}(undef, N_cand + 2)
    times[1] = 0.0
    times[2:(N_cand+1)] .= raw_times
    times[end] = T_total

    ctmc.cand_times = times
    return times
end

"""
    accumulate_interval_background_counts!(
        N_bg_mop::Vector{Int}, N_bg_cp::Vector{Int},
        cand_times::Vector{Float64},
        spike_t::Vector{Float64}, spike_neurons::Vector{Int},
        assign::Vector{SpikeAssignment}, channel_region::Vector{Int}
    )

Single-pass O(N_spikes + B) linear scan accumulating unassigned background spikes (z_i == 0)
into candidate intervals [cand_times[b], cand_times[b+1]].
Zero heap allocations when output buffers are preallocated.
"""
function accumulate_interval_background_counts!(
    N_bg_mop::Vector{Int}, N_bg_cp::Vector{Int},
    cand_times::Vector{Float64},
    spike_t::Vector{Float64}, spike_neurons::Vector{Int},
    assign::Vector{SpikeAssignment}, channel_region::Vector{Int}
)
    B = length(cand_times) - 1
    fill!(N_bg_mop, 0)
    fill!(N_bg_cp, 0)

    N_spk = length(spike_t)
    b = 1

    @inbounds for i in 1:N_spk
        assign[i].event_id == 0 || continue
        t = spike_t[i]

        while b < B && t > cand_times[b + 1]
            b += 1
        end

        c = spike_neurons[i]
        if 1 <= c <= length(channel_region)
            reg = channel_region[c]
            if reg == 1
                N_bg_mop[b] += 1
            elseif reg == 2
                N_bg_cp[b] += 1
            end
        end
    end
end


"""
    accumulate_interval_motif_marks!(
        log_motif_ev::Matrix{Float64},
        cand_times::Vector{Float64},
        occurrences::Dict{Int, PureOccurrence},
        motifs::Dict{Int, PurePacketMotif},
        hyperstate::Level5Hyperstate
    )

Accumulates Scale n=1 sequence occurrence marks (vigor c_e, kinematics u_e, arrival rate lambda_m)
into candidate intervals [cand_times[b], cand_times[b+1]].
"""
function accumulate_interval_motif_marks!(
    log_motif_ev::Matrix{Float64},
    cand_times::Vector{Float64},
    occurrences::Dict{Int, PureOccurrence},
    motifs::Dict{Int, PurePacketMotif},
    hyperstate::Level5Hyperstate
)
    K = hyperstate.K
    B = length(cand_times) - 1
    fill!(log_motif_ev, 0.0)

    R = hyperstate.R
    LN_2PI = log(2π)

    for occ in values(occurrences)
        tau = occ.tau
        b = searchsortedlast(cand_times, tau)
        if b < 1
            b = 1
        elseif b > B
            b = B
        end

        mid = occ.motif
        c_e = occ.amplitude
        log_c = log(max(c_e, 1e-12))

        for k in 1:K
            reg = hyperstate.regimes[k]
            lam_m = (1 <= mid <= length(hyperstate.lambda_m)) ? hyperstate.lambda_m[mid] : 1.0
            lp_rate = log(max(lam_m, 1e-12))

            lp_kin = -0.5 * R * LN_2PI
            for r in 1:R
                lam = reg.lambda_u[r]
                lp_kin += 0.5 * (log(max(lam, 1e-12)) - lam * occ.u[r]^2)
            end

            alpha_k = reg.alpha_c
            beta_k = reg.beta_c
            lp_vigor = alpha_k * log(beta_k) - loggamma(alpha_k) + (alpha_k - 1.0) * log_c - beta_k * c_e

            log_motif_ev[k, b] += lp_rate + lp_kin + lp_vigor
        end
    end
end


"""
    sample_ctmc_state_trajectory!(
        ctmc::ContinuousTimeCTMC,
        hyperstate::Level5Hyperstate,
        occurrences::Dict{Int, PureOccurrence},
        motifs::Dict{Int, PurePacketMotif},
        spike_t::Vector{Float64}, spike_neurons::Vector{Int},
        assign::Vector{SpikeAssignment}, bg_rates::Vector{Float64},
        T_total::Float64;
        rng::AbstractRNG=Random.default_rng()
    )

Runs exact continuous-time Rao-Teh uniformization FFBS on stack arrays.
Couples occurrences: k_e = S(tau_e).
"""
function sample_ctmc_state_trajectory!(
    ctmc::ContinuousTimeCTMC,
    hyperstate::Level5Hyperstate,
    occurrences::Dict{Int, PureOccurrence},
    motifs::Dict{Int, PurePacketMotif},
    spike_t::Vector{Float64}, spike_neurons::Vector{Int},
    assign::Vector{SpikeAssignment}, bg_rates::Vector{Float64},
    T_total::Float64;
    rng::AbstractRNG=Random.default_rng()
)
    K = ctmc.K

    # 1. Sample auxiliary candidate times T_cand ~ Poisson(Omega)
    sample_candidate_times!(ctmc, T_total; rng=rng)
    B = length(ctmc.cand_times) - 1

    # 2. Subordinated discrete transition matrix: B_mat = I + (1 / Omega) * Q
    Omega = ctmc.Omega
    B_trans = Matrix{Float64}(undef, K, K)
    for k in 1:K
        for j in 1:K
            if j == k
                B_trans[k, k] = max(1e-12, 1.0 - abs(ctmc.Q[k, k]) / Omega)
            else
                B_trans[k, j] = max(0.0, ctmc.Q[k, j] / Omega)
            end
        end
        s = sum(B_trans[k, :])
        if s > 0.0
            B_trans[k, :] ./= s
        end
    end

    # 3. Precompute regional aggregate baseline rates
    Lambda_0_mop = 0.0
    Lambda_0_cp = 0.0
    for c in 1:length(bg_rates)
        reg = (c <= length(ctmc.channel_region)) ? ctmc.channel_region[c] : 1
        if reg == 1
            Lambda_0_mop += bg_rates[c]
        elseif reg == 2
            Lambda_0_cp += bg_rates[c]
        end
    end

    # 4. Count background spikes per interval
    N_bg_mop = zeros(Int, B)
    N_bg_cp = zeros(Int, B)
    accumulate_interval_background_counts!(
        N_bg_mop, N_bg_cp, ctmc.cand_times,
        spike_t, spike_neurons, assign, ctmc.channel_region
    )

    # 5. Accumulate occurrence marks
    log_motif_ev = zeros(Float64, K, B)
    accumulate_interval_motif_marks!(
        log_motif_ev, ctmc.cand_times, occurrences, motifs, hyperstate
    )

    # Total motif rate compensator per regime
    tot_motif_rate = zeros(Float64, K)
    for k in 1:K
        tot_motif_rate[k] = sum(hyperstate.lambda_m)
    end

    # 6. Forward Filtering
    alpha_filt = Matrix{Float64}(undef, K, B)
    log_lik = Matrix{Float64}(undef, K, B)

    for b in 1:B
        dt_b = ctmc.cand_times[b + 1] - ctmc.cand_times[b]
        for k in 1:K
            g_mop = ctmc.gamma_tone[1, k]
            g_cp = ctmc.gamma_tone[2, k]
            lp_bg = N_bg_mop[b] * log(max(g_mop, 1e-12)) - g_mop * Lambda_0_mop * dt_b +
                    N_bg_cp[b] * log(max(g_cp, 1e-12)) - g_cp * Lambda_0_cp * dt_b
            log_lik[k, b] = lp_bg + log_motif_ev[k, b] - dt_b * tot_motif_rate[k]
        end

        max_ll = maximum(log_lik[:, b])

        if b == 1
            for k in 1:K
                alpha_filt[k, 1] = (1.0 / K) * exp(log_lik[k, 1] - max_ll)
            end
        else
            for k in 1:K
                pred = sum(alpha_filt[j, b-1] * B_trans[j, k] for j in 1:K)
                alpha_filt[k, b] = pred * exp(log_lik[k, b] - max_ll)
            end
        end

        norm_c = sum(alpha_filt[:, b])
        if norm_c > 0.0
            alpha_filt[:, b] ./= norm_c
        else
            alpha_filt[:, b] .= 1.0 / K
        end
    end

    # 7. Backward Sampling
    sampled_states = Vector{Int}(undef, B)
    probs_B = alpha_filt[:, B]
    u_rand = rand(rng)
    cum = 0.0
    chosen = K
    for k in 1:K
        cum += probs_B[k]
        if cum >= u_rand
            chosen = k
            break
        end
    end
    sampled_states[B] = chosen

    for b in (B-1):-1:1
        next_k = sampled_states[b + 1]
        back_weights = [alpha_filt[k, b] * B_trans[k, next_k] for k in 1:K]
        sum_bw = sum(back_weights)
        if sum_bw > 0.0
            back_weights ./= sum_bw
        else
            back_weights .= 1.0 / K
        end

        u_rand = rand(rng)
        cum = 0.0
        chosen = K
        for k in 1:K
            cum += back_weights[k]
            if cum >= u_rand
                chosen = k
                break
            end
        end
        sampled_states[b] = chosen
    end

    ctmc.cand_states = sampled_states

    # 8. Couple Occurrences: k_e = S(tau_e)
    for reg in values(hyperstate.regimes)
        empty!(reg.assigned_occurrences)
    end

    for occ in values(occurrences)
        b = searchsortedlast(ctmc.cand_times, occ.tau)
        b = clamp(b, 1, B)
        reg_id = sampled_states[b]
        occ.regime_id = reg_id
        if haskey(hyperstate.regimes, reg_id)
            push!(hyperstate.regimes[reg_id].assigned_occurrences, occ.id)
        end
    end

    # 9. Compute Sufficient Statistics: Dwell times T_k, Transition counts N_kj, and Background tallies
    fill!(ctmc.T_k, 0.0)
    fill!(ctmc.N_trans, 0)
    fill!(ctmc.N_bg_mop_k, 0.0)
    fill!(ctmc.N_bg_cp_k, 0.0)

    for b in 1:B
        k = sampled_states[b]
        dt_b = ctmc.cand_times[b + 1] - ctmc.cand_times[b]
        ctmc.T_k[k] += dt_b
        ctmc.N_bg_mop_k[k] += N_bg_mop[b]
        ctmc.N_bg_cp_k[k] += N_bg_cp[b]
        if b < B
            next_k = sampled_states[b + 1]
            if next_k != k
                ctmc.N_trans[k, next_k] += 1
            end
        end
    end

    return ctmc
end

"""
    update_ctmc_parameters!(
        ctmc::ContinuousTimeCTMC,
        hyperstate::Level5Hyperstate,
        occurrences::Dict{Int, PureOccurrence},
        motifs::Dict{Int, PurePacketMotif},
        bg_rates::Vector{Float64},
        spike_neurons::Vector{Int},
        assign::Vector{SpikeAssignment},
        T_total::Float64;
        rng::AbstractRNG=Random.default_rng()
    )

Exact conjugate Gibbs updates for:
1. Generator matrix Q: q_{kj} ~ Gamma(alpha_q + N_{kj}, beta_q + T_k) with beta_q = (K-1)*alpha_q*W_min
2. Regional multipliers: gamma_{r,k} ~ Gamma(a_gamma + N_bg^r(k), b_gamma + T_k * Lambda_{0,tot}^r) (Regime 1 anchored at 1.0)
3. Uniformization rate: Omega = max_k |q_{kk}| + xi with xi ~ Gamma(a_omega, 1/T_total)
4. Scale gauge invariants: beta_{c,1} == 10.0, Lambda_{u,1} == I_R
"""
function update_ctmc_parameters!(
    ctmc::ContinuousTimeCTMC,
    hyperstate::Level5Hyperstate,
    occurrences::Dict{Int, PureOccurrence},
    motifs::Dict{Int, PurePacketMotif},
    bg_rates::Vector{Float64},
    spike_neurons::Vector{Int},
    assign::Vector{SpikeAssignment},
    T_total::Float64;
    rng::AbstractRNG=Random.default_rng()
)
    K = ctmc.K

    # 1. Update Generator Matrix Q
    ctmc.beta_q = max(1, K - 1) * ctmc.alpha_q * ctmc.W_min
    for k in 1:K
        for j in 1:K
            if j != k
                alpha_post = ctmc.alpha_q + Float64(ctmc.N_trans[k, j])
                beta_post = ctmc.beta_q + ctmc.T_k[k]
                ctmc.Q[k, j] = rand(rng, Gamma(alpha_post, 1.0 / max(1e-12, beta_post)))
            end
        end
        ctmc.Q[k, k] = -sum(ctmc.Q[k, j] for j in 1:K if j != k)
    end

    # 2. Update Uniformization Rate Omega
    max_exit = maximum(abs(ctmc.Q[k, k]) for k in 1:K)
    xi = rand(rng, Gamma(ctmc.a_omega, 1.0 / max(1e-12, T_total)))
    ctmc.Omega = max_exit + xi

    # 3. Update Regional Tone Multipliers gamma_tone
    Lambda_0_mop = 0.0
    Lambda_0_cp = 0.0
    for c in 1:length(bg_rates)
        reg = (c <= length(ctmc.channel_region)) ? ctmc.channel_region[c] : 1
        if reg == 1
            Lambda_0_mop += bg_rates[c]
        elseif reg == 2
            Lambda_0_cp += bg_rates[c]
        end
    end

    # Anchor Regime 1: gamma_{r, 1} == 1.0
    ctmc.gamma_tone[1, 1] = 1.0
    ctmc.gamma_tone[2, 1] = 1.0

    # Conjugate Gamma update for k > 1
    a_gam = 10.0
    b_gam = 10.0
    for k in 2:K
        # MOp
        alpha_mop = a_gam + ctmc.N_bg_mop_k[k]
        beta_mop = b_gam + ctmc.T_k[k] * Lambda_0_mop
        ctmc.gamma_tone[1, k] = max(0.1, rand(rng, Gamma(alpha_mop, 1.0 / max(1e-12, beta_mop))))

        # CP
        alpha_cp = a_gam + ctmc.N_bg_cp_k[k]
        beta_cp = b_gam + ctmc.T_k[k] * Lambda_0_cp
        ctmc.gamma_tone[2, k] = max(0.1, rand(rng, Gamma(alpha_cp, 1.0 / max(1e-12, beta_cp))))
    end

    # 4. Triad Anchors on Level 5 Regimes
    reg1 = hyperstate.regimes[1]
    reg1.beta_c = reg1.alpha_c # E[c | 1] == 1.0
    reg1.lambda_u .= 1.0      # Lambda_{u,1} == I_R
    reg1.active_dims .= true

    # Motor Vigor Updates for k > 1
    for k in 2:K
        haskey(hyperstate.regimes, k) || continue
        reg = hyperstate.regimes[k]
        E_k = length(reg.assigned_occurrences)
        if E_k > 0
            sum_c = sum(occurrences[eid].amplitude for eid in reg.assigned_occurrences if haskey(occurrences, eid); init=0.0)
            alpha_post = hyperstate.a_c + Float64(E_k) * reg.alpha_c
            beta_post = hyperstate.b_c + sum_c
            reg.beta_c = max(0.1, rand(rng, Gamma(alpha_post, 1.0 / max(1e-12, beta_post))))
        end
    end

    # Kinematic Precision Updates for k > 1
    for k in 2:K
        haskey(hyperstate.regimes, k) || continue
        reg = hyperstate.regimes[k]
        E_k = length(reg.assigned_occurrences)
        E_k == 0 && continue
        for r in 1:hyperstate.R
            sum_u2 = sum(occurrences[eid].u[r]^2 for eid in reg.assigned_occurrences if haskey(occurrences, eid); init=0.0)
            alpha_post = hyperstate.a_u + 0.5 * Float64(E_k)
            beta_post = hyperstate.b_u + 0.5 * sum_u2
            lam = rand(rng, Gamma(alpha_post, 1.0 / max(1e-12, beta_post)))
            reg.lambda_u[r] = max(1e-6, min(hyperstate.lambda_dormant, lam))
            reg.active_dims[r] = (reg.lambda_u[r] < hyperstate.lambda_dormant)
        end
    end
end

