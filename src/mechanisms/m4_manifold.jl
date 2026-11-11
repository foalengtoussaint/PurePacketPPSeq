# mechanisms/m4_manifold.jl - Level m=4 Continuous Latent Kinematics & ARD Manifold
#
# Generative Role:
# - Continuous low-dimensional temporal warping:
#     mu_{e,p} = tau_e + mu_p + sum_r W_{r,p} u_{e,r}
# - Zero-allocation analytical solve for occurrence coordinate u_e in R^R.
# - Conjugate Gaussian draws for carrier loadings W_{r,p}.
# - Automatic Relevance Determination (ARD) manifold precisions alpha_r.
# - Cramér-Rao manifold pruning: deactivates axes exceeding alpha_prune = 6500 s^-2.

"""
    sample_packet_loadings!(m::PurePacketMotif, state_alpha, active_mask, occ_u, residuals, precisions; rng=default_rng())

Conjugate Gaussian draw for manifold loadings W_{r,p} followed by exact translational gauge projection.
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
    sample_occurrence_coordinate!(u_out, m, active_mask, residuals, precisions; prior_precisions=nothing, tau_precision=1.0, rng=default_rng())

Zero-allocation conjugate Gaussian draw for latent coordinate u_e in R^R:
    u_e | tau_e ~ N(mu_u, Sigma_u)
where prior precision is scaled by Student-t robustness weight tau_e.
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
    sample_global_ard_precisions!(state::ARDManifoldState, motifs::Vector{PurePacketMotif}; rng=default_rng())

Conjugate Gamma draws for ARD precision hyperprior:
    alpha_r ~ Gamma(a_0 + 0.5 * sum_m P_m, 1.0 / (b_0 + 0.5 * sum_{m,p} W_{r,p}^2)).
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

"""
    prune_inactive_axes!(state::ARDManifoldState, motifs::Vector{PurePacketMotif})

Prunes dormant manifold dimensions exceeding Cramér-Rao threshold alpha_prune = 6500 s^-2.
"""
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

"""
    check_and_birth_axis!(state::ARDManifoldState, motifs, occurrences; rng=default_rng(), sigma_threshold=3.0)

Data-driven dimensional birth operator R -> R+1.
"""
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
