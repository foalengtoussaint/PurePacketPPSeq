# kinematics.jl - Continuous Spatial Manifold & Translational Gauge

# =====================================================================
# 3. CONJUGATE GIBBS CONDITIONALS
# =====================================================================

"""
    project_translational_gauge!(m::PurePacketMotif, active_mask::BitVector)

Enforces exact translational gauge invariance:
    sum_{p=1}^{P_m} kappa_{m,p} W_{r,p} = 0
for all active ARD axes r in 1:R_max.
"""
function project_translational_gauge!(m::PurePacketMotif, active_mask::BitVector)
    P = length(m.packets)
    P == 0 && return
    R_max = length(active_mask)
    tot_prec = sum(1.0 / m.packets[p].var for p in 1:P)
    tot_prec <= 0.0 && return

    for r in 1:R_max
        if active_mask[r]
            w_mean = sum((1.0 / m.packets[p].var) * m.W[r, p] for p in 1:P) / tot_prec
            @views m.W[r, :] .-= w_mean
        else
            @views m.W[r, :] .= 0.0
        end
    end
end


"""
    sample_packet_loadings!(m::PurePacketMotif, state_alpha, active_mask, occ_u, residuals, precisions; rng)

Conjugate Gaussian draw for W_{r,p} with exact translational gauge projection:
    sum_{p=1}^{P_m} kappa_{m,p} W_{r,p} = 0
"""
function sample_packet_loadings!(
    m::PurePacketMotif,
    state_alpha::Vector{Float64},
    active_mask::BitVector,
    occ_u::Vector{Vector{Float64}},
    residuals::Vector{Vector{Float64}},
    precisions::Vector{Vector{Float64}};
    rng::AbstractRNG=Random.default_rng()
)
    P = length(m.packets)
    P == 0 && return
    E = length(occ_u)
    R_max = length(active_mask)

    k1 = 0; k2 = 0; k3 = 0; k4 = 0
    K = 0
    @inbounds for r in 1:R_max
        if active_mask[r]
            K += 1
            if K == 1; k1 = r
            elseif K == 2; k2 = r
            elseif K == 3; k3 = r
            elseif K == 4; k4 = r
            end
        end
    end

    K == 0 && return

    @inbounds for p in 1:P
        if K == 1
            a11 = state_alpha[k1]
            b1 = 0.0
            for e in 1:E
                kap = precisions[e][p]
                kap <= 0.0 && continue
                u1 = occ_u[e][k1]
                a11 += kap * u1 * u1
                b1 += kap * u1 * residuals[e][p]
            end
            sig = sqrt(1.0 / a11)
            m.W[k1, p] = (b1 / a11) + sig * randn(rng)
        elseif K == 2
            a11 = state_alpha[k1]; a12 = 0.0; a22 = state_alpha[k2]
            b1 = 0.0; b2 = 0.0
            for e in 1:E
                kap = precisions[e][p]
                kap <= 0.0 && continue
                u1 = occ_u[e][k1]; u2 = occ_u[e][k2]
                a11 += kap * u1 * u1
                a12 += kap * u1 * u2
                a22 += kap * u2 * u2
                y = residuals[e][p]
                b1 += kap * u1 * y
                b2 += kap * u2 * y
            end
            det = a11 * a22 - a12 * a12
            inv11 = a22 / det; inv12 = -a12 / det; inv22 = a11 / det
            mu1 = inv11 * b1 + inv12 * b2
            mu2 = inv12 * b1 + inv22 * b2
            L11 = sqrt(max(inv11, 1e-12))
            L21 = inv12 / L11
            L22 = sqrt(max(inv22 - L21 * L21, 1e-12))
            z1 = randn(rng); z2 = randn(rng)
            m.W[k1, p] = mu1 + L11 * z1
            m.W[k2, p] = mu2 + L21 * z1 + L22 * z2
        else
            # K >= 3: unrolled block inversion
            A = zeros(MMatrix{4, 4, Float64})
            b_vec = zeros(MVector{4, Float64})
            for r in 1:K
                r_idx = (r == 1) ? k1 : ((r == 2) ? k2 : ((r == 3) ? k3 : k4))
                A[r, r] = state_alpha[r_idx]
            end
            for e in 1:E
                kap = precisions[e][p]
                kap <= 0.0 && continue
                u = occ_u[e]
                y = residuals[e][p]
                for r1 in 1:K
                    r1_idx = (r1 == 1) ? k1 : ((r1 == 2) ? k2 : ((r1 == 3) ? k3 : k4))
                    ur1 = u[r1_idx]
                    b_vec[r1] += kap * ur1 * y
                    for r2 in 1:K
                        r2_idx = (r2 == 1) ? k1 : ((r2 == 2) ? k2 : ((r2 == 3) ? k3 : k4))
                        A[r1, r2] += kap * ur1 * u[r2_idx]
                    end
                end
            end
            A_sub = SMatrix{K, K, Float64}(view(A, 1:K, 1:K))
            b_sub = SVector{K, Float64}(view(b_vec, 1:K))
            cov_sub = inv(A_sub)
            mu_sub = cov_sub * b_sub
            C_chol = cholesky(Hermitian(cov_sub))
            w_samp = mu_sub + C_chol.L * randn(rng, SVector{K, Float64})
            for r in 1:K
                r_idx = (r == 1) ? k1 : ((r == 2) ? k2 : ((r == 3) ? k3 : k4))
                m.W[r_idx, p] = w_samp[r]
            end
        end
    end

    project_translational_gauge!(m, active_mask)
    update_motif_bounds!(m)
end

"""
    sample_occurrence_coordinate!(u_out, m::PurePacketMotif, active_mask, res, prec; prior_precisions, rng)

Conjugate Gaussian draw for u_e in R^R (0 heap allocations).
"""
function sample_occurrence_coordinate!(
    u_out::Vector{Float64},
    m::PurePacketMotif,
    active_mask::BitVector,
    residuals::Vector{Float64},
    precisions::Vector{Float64};
    prior_precisions::Union{Nothing, AbstractVector{Float64}}=nothing,
    tau_precision::Float64=1.0,
    rng::AbstractRNG=Random.default_rng()
)
    P = length(m.packets)
    P == 0 && return
    R_max = length(active_mask)

    k1 = 0; k2 = 0; k3 = 0; k4 = 0
    K = 0
    @inbounds for r in 1:R_max
        if active_mask[r]
            K += 1
            if K == 1; k1 = r
            elseif K == 2; k2 = r
            elseif K == 3; k3 = r
            elseif K == 4; k4 = r
            end
        end
    end

    if K == 0
        u_out .= 0.0
        return
    end

    has_prior = prior_precisions !== nothing
    @inline p_prec(idx) = (has_prior && length(prior_precisions) >= idx) ? (prior_precisions[idx] * tau_precision) : tau_precision

    if K == 1
        a11 = p_prec(k1)
        b1 = 0.0
        @inbounds for p in 1:P
            kap = precisions[p]
            kap <= 0.0 && continue
            w1 = m.W[k1, p]
            a11 += kap * w1 * w1
            b1 += kap * w1 * residuals[p]
        end
        sig = sqrt(1.0 / a11)
        u_out[k1] = (b1 / a11) + sig * randn(rng)
        for r in 1:R_max; r != k1 && (u_out[r] = 0.0); end
    elseif K == 2
        a11 = p_prec(k1); a12 = 0.0; a22 = p_prec(k2)
        b1 = 0.0; b2 = 0.0
        @inbounds for p in 1:P
            kap = precisions[p]
            kap <= 0.0 && continue
            w1 = m.W[k1, p]; w2 = m.W[k2, p]
            a11 += kap * w1 * w1
            a12 += kap * w1 * w2
            a22 += kap * w2 * w2
            y = residuals[p]
            b1 += kap * w1 * y
            b2 += kap * w2 * y
        end
        det = a11 * a22 - a12 * a12
        inv11 = a22 / det; inv12 = -a12 / det; inv22 = a11 / det
        mu1 = inv11 * b1 + inv12 * b2
        mu2 = inv12 * b1 + inv22 * b2
        L11 = sqrt(max(inv11, 1e-12))
        L21 = inv12 / L11
        L22 = sqrt(max(inv22 - L21 * L21, 1e-12))
        z1 = randn(rng); z2 = randn(rng)
        u_out[k1] = mu1 + L11 * z1
        u_out[k2] = mu2 + L21 * z1 + L22 * z2
        for r in 1:R_max; (r != k1 && r != k2) && (u_out[r] = 0.0); end
    else
        A = zeros(MMatrix{4, 4, Float64})
        b_vec = zeros(MVector{4, Float64})
        for r in 1:K
            r_idx = (r == 1) ? k1 : ((r == 2) ? k2 : ((r == 3) ? k3 : k4))
            A[r, r] = p_prec(r_idx)
        end
        @inbounds for p in 1:P
            kap = precisions[p]
            kap <= 0.0 && continue
            y = residuals[p]
            for r1 in 1:K
                r1_idx = (r1 == 1) ? k1 : ((r1 == 2) ? k2 : ((r1 == 3) ? k3 : k4))
                wr1 = m.W[r1_idx, p]
                b_vec[r1] += kap * wr1 * y
                for r2 in 1:K
                    r2_idx = (r2 == 1) ? k1 : ((r2 == 2) ? k2 : ((r2 == 3) ? k3 : k4))
                    A[r1, r2] += kap * wr1 * m.W[r2_idx, p]
                end
            end
        end
        A_sub = SMatrix{K, K, Float64}(view(A, 1:K, 1:K))
        b_sub = SVector{K, Float64}(view(b_vec, 1:K))
        cov_sub = inv(A_sub)
        mu_sub = cov_sub * b_sub
        C_chol = cholesky(Hermitian(cov_sub))
        u_samp = mu_sub + C_chol.L * randn(rng, SVector{K, Float64})
        for r in 1:R_max; u_out[r] = 0.0; end
        for r in 1:K
            r_idx = (r == 1) ? k1 : ((r == 2) ? k2 : ((r == 3) ? k3 : k4))
            u_out[r_idx] = u_samp[r]
        end
    end
end


"""
    resample_occurrence_onset!(occ::PureOccurrence, m::PurePacketMotif, active_mask; T_max=Inf, rng)

Exact conjugate Gaussian Gibbs draw for occurrence onset tau_e subtracting manifold deformation s_v(p):
    tau_e ~ N(mu_tau, 1 / tot_prec)
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
    sample_packet_masses!(m::PurePacketMotif, occurrences::Vector{PureOccurrence}; alpha_a=1.0, beta_a=1.0, rng)

Conjugate Gamma draw for packet emission mass a_{m,p}:
    a_{m,p} ~ Gamma(alpha_a + c_{m,p}, 1.0 / (beta_a + E_tilde))
where E_tilde = sum_{e in E_m} c_e is the total integrated occurrence gain and c_{m,p} is the total assigned spike count.
"""

function sample_global_ard_precisions!(
    state::ARDManifoldState,
    motifs::Vector{PurePacketMotif};
    rng::AbstractRNG=Random.default_rng()
)
    P_tot = sum(length(m.packets) for m in motifs)
    a_post = state.a0 + 0.5 * P_tot

    for r in 1:state.R_max
        if state.active_mask[r]
            sum_w2 = 0.0
            for m in motifs
                for p in 1:length(m.packets)
                    sum_w2 += m.W[r, p]^2
                end
            end
            b_post = state.b0 + 0.5 * sum_w2
            scale = 1.0 / max(b_post, 1e-12)
            state.alpha[r] = rand(rng, Gamma(a_post, scale))
        else
            state.alpha[r] = 1.0e6
        end
    end
end


function prune_inactive_axes!(state::ARDManifoldState, motifs::Vector{PurePacketMotif})
    for r in 1:state.R_max
        if state.active_mask[r]
            if state.alpha[r] > state.prune_threshold
                state.active_mask[r] = false
                state.alpha[r] = 1.0e6
                for m in motifs
                    @views m.W[r, :] .= 0.0
                end
            end
        end
    end
end


function check_and_birth_axis!(
    state::ARDManifoldState,
    motifs::Dict{Int, PurePacketMotif},
    occurrences::Vector{PureOccurrence};
    rng::AbstractRNG=Random.default_rng(),
    sigma_threshold::Float64=3.0
)
    dormant_r = 0
    for r in 1:state.R_max
        if !state.active_mask[r]
            dormant_r = r
            break
        end
    end
    dormant_r == 0 && return (false, 1.0)

    k1 = 0; k2 = 0; k3 = 0; k4 = 0
    K = 0
    for r in 1:state.R_max
        if state.active_mask[r]
            K += 1
            if K == 1; k1 = r
            elseif K == 2; k2 = r
            elseif K == 3; k3 = r
            elseif K == 4; k4 = r
            end
        end
    end

    total_chi2 = 0.0
    dof = 0

    for occ in occurrences
        m = motifs[occ.motif]
        u = occ.u
        P = length(m.packets)
        for p in 1:P
            c = occ.counts[p]
            c == 0 && continue
            mean_t = occ.sum_t[p] / c
            target_t = occ.tau + m.packets[p].mu
            pred_s = 0.0
            if K == 1; pred_s = m.W[k1, p] * u[k1]
            elseif K == 2; pred_s = m.W[k1, p] * u[k1] + m.W[k2, p] * u[k2]
            elseif K == 3; pred_s = m.W[k1, p] * u[k1] + m.W[k2, p] * u[k2] + m.W[k3, p] * u[k3]
            elseif K == 4; pred_s = m.W[k1, p] * u[k1] + m.W[k2, p] * u[k2] + m.W[k3, p] * u[k3] + m.W[k4, p] * u[k4]
            end
            res = (mean_t - target_t) - pred_s
            kap = c / m.packets[p].var
            total_chi2 += kap * res * res
            dof += 1
        end
    end

    dof <= 0 && return (false, 1.0)
    chi2_ratio = total_chi2 / dof
    z_score = (total_chi2 - dof) / sqrt(2.0 * dof)

    if z_score > sigma_threshold
        state.active_mask[dormant_r] = true
        state.alpha[dormant_r] = 500.0
        for occ in occurrences
            occ.u[dormant_r] = randn(rng)
        end
        for m in values(motifs)
            @views m.W[dormant_r, :] .= 0.0
        end
        return (true, chi2_ratio)
    end

    return (false, chi2_ratio)
end

# =====================================================================
# 9. LEVEL m=5 TASK REGIMES & HYPERSTATE
# =====================================================================
