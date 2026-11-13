# mechanisms/m4_manifold.jl - Level m=4 Continuous Latent Kinematics & ARD Manifold
#
# Generative Role:
# - Continuous low-dimensional temporal warping:
#     mu_{e,p} = tau_e + mu_p + sum_r W_{r,p} u_{e,r}
# - Zero-allocation analytical and in-place Cholesky solves for occurrence coordinate u_e in R^R.
# - Conjugate Gaussian draws for carrier loadings W_{r,p}.
# - Automatic Relevance Determination (ARD) manifold precisions alpha_r.
# - Dynamic data-driven dimensional growth R -> R+1 and Cramér-Rao pruning (alpha_prune = 6500 s^-2).

"""
    KinematicWorkspace

Pre-allocated reusable scratchpad buffers ensuring exact 0-allocation linear solves
for arbitrary kinematic dimensions R.
"""
mutable struct KinematicWorkspace
    A::Matrix{Float64}
    b::Vector{Float64}
    u::Vector{Float64}
    z::Vector{Float64}
    act::Vector{Int}
end

function KinematicWorkspace(max_dim::Int=64)
    return KinematicWorkspace(
        zeros(Float64, max_dim, max_dim),
        zeros(Float64, max_dim),
        zeros(Float64, max_dim),
        zeros(Float64, max_dim),
        zeros(Int, max_dim)
    )
end

const GLOBAL_KINEMATIC_WS = [KinematicWorkspace(64) for _ in 1:max(Threads.maxthreadid(), 16)]

@inline function get_kinematic_workspace()
    tid = Threads.threadid()
    if tid <= length(GLOBAL_KINEMATIC_WS)
        return GLOBAL_KINEMATIC_WS[tid]
    else
        return GLOBAL_KINEMATIC_WS[1]
    end
end

"""
    sample_packet_loadings!(m::PurePacketMotif, state_alpha, active_mask, occ_u, residuals, precisions; rng=default_rng())

Conjugate Gaussian draw for manifold loadings W_{r,p} followed by exact translational gauge projection.
Operates dynamically for any dimension R with 0 heap allocations.
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
    ws = get_kinematic_workspace()

    K = 0
    @inbounds for r in 1:length(active_mask)
        if active_mask[r]
            K += 1
            ws.act[K] = r
        end
    end

    K == 0 && return

    @inbounds for p in 1:P
        if K == 1
            k1 = ws.act[1]
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
            k1 = ws.act[1]; k2 = ws.act[2]
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
            A = view(ws.A, 1:K, 1:K)
            b_vec = view(ws.b, 1:K)
            u_vec = view(ws.u, 1:K)
            z_vec = view(ws.z, 1:K)
            fill!(A, 0.0)
            fill!(b_vec, 0.0)
            for r in 1:K
                A[r, r] = state_alpha[ws.act[r]]
            end
            for e in 1:E
                kap = precisions[e][p]
                kap <= 0.0 && continue
                u = occ_u[e]
                y = residuals[e][p]
                for r1 in 1:K
                    ur1 = u[ws.act[r1]]
                    b_vec[r1] += kap * ur1 * y
                    for r2 in 1:K
                        A[r1, r2] += kap * ur1 * u[ws.act[r2]]
                    end
                end
            end
            C = cholesky!(Hermitian(A))
            ldiv!(u_vec, C, b_vec)
            for r in 1:K
                z_vec[r] = randn(rng)
            end
            ldiv!(C.L, z_vec)
            u_vec .+= z_vec
            for r in 1:K
                m.W[ws.act[r], p] = u_vec[r]
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
Operates dynamically for any dimension R with 0 heap allocations.
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
    ws = get_kinematic_workspace()

    K = 0
    @inbounds for r in 1:length(active_mask)
        if active_mask[r]
            K += 1
            ws.act[K] = r
        end
    end

    if K == 0
        u_out .= 0.0
        return
    end

    has_prior = prior_precisions !== nothing
    @inline p_prec(idx) = (has_prior && length(prior_precisions) >= idx) ? (prior_precisions[idx] * tau_precision) : tau_precision

    if K == 1
        k1 = ws.act[1]
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
        for r in 1:length(u_out); r != k1 && (u_out[r] = 0.0); end
    elseif K == 2
        k1 = ws.act[1]; k2 = ws.act[2]
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
        for r in 1:length(u_out); (r != k1 && r != k2) && (u_out[r] = 0.0); end
    else
        A = view(ws.A, 1:K, 1:K)
        b_vec = view(ws.b, 1:K)
        u_vec = view(ws.u, 1:K)
        z_vec = view(ws.z, 1:K)
        fill!(A, 0.0)
        fill!(b_vec, 0.0)
        for r in 1:K
            A[r, r] = p_prec(ws.act[r])
        end
        @inbounds for p in 1:P
            kap = precisions[p]
            kap <= 0.0 && continue
            y = residuals[p]
            for r1 in 1:K
                wr1 = m.W[ws.act[r1], p]
                b_vec[r1] += kap * wr1 * y
                for r2 in 1:K
                    A[r1, r2] += kap * wr1 * m.W[ws.act[r2], p]
                end
            end
        end
        C = cholesky!(Hermitian(A))
        ldiv!(u_vec, C, b_vec)
        for r in 1:K
            z_vec[r] = randn(rng)
        end
        ldiv!(C.L, z_vec)
        u_vec .+= z_vec
        for r in 1:length(u_out); u_out[r] = 0.0; end
        for r in 1:K
            u_out[ws.act[r]] = u_vec[r]
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
    P_tot = sum(length(m.packets) for m in motifs; init=0)
    a_post = state.a0 + 0.5 * P_tot

    for r in 1:length(state.active_mask)
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
    for r in 1:length(state.active_mask)
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

Data-driven dimensional birth operator R -> R+1. Dynamically grows the manifold dimension without
any arbitrary hardcoded ceiling.
"""
function check_and_birth_axis!(
    state::ARDManifoldState,
    motifs::Dict{Int, PurePacketMotif},
    occurrences::Vector{PureOccurrence};
    rng::AbstractRNG=Random.default_rng(),
    sigma_threshold::Float64=3.0
)
    dormant_r = findfirst(!, state.active_mask)
    need_grow = (dormant_r === nothing)
    target_r = need_grow ? (length(state.active_mask) + 1) : dormant_r

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
            @simd for r in 1:min(size(m.W, 1), length(u), length(state.active_mask))
                if state.active_mask[r]
                    @inbounds pred_s += m.W[r, p] * u[r]
                end
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
        if need_grow
            push!(state.active_mask, true)
            push!(state.alpha, 500.0)
            state.R_max = length(state.active_mask)
            for m in values(motifs)
                P = length(m.packets)
                new_W = zeros(Float64, state.R_max, P)
                n_copy = min(size(m.W, 1), state.R_max - 1)
                if n_copy > 0
                    @views new_W[1:n_copy, :] .= m.W[1:n_copy, :]
                end
                m.W = new_W
            end
            for occ in occurrences
                push!(occ.u, randn(rng))
            end
        else
            state.active_mask[target_r] = true
            state.alpha[target_r] = 500.0
            for occ in occurrences
                if target_r <= length(occ.u)
                    occ.u[target_r] = randn(rng)
                else
                    push!(occ.u, randn(rng))
                end
            end
            for m in values(motifs)
                if target_r <= size(m.W, 1)
                    @views m.W[target_r, :] .= 0.0
                else
                    P = length(m.packets)
                    new_W = zeros(Float64, target_r, P)
                    n_copy = min(size(m.W, 1), target_r)
                    if n_copy > 0
                        @views new_W[1:n_copy, :] .= m.W[1:n_copy, :]
                    end
                    m.W = new_W
                end
            end
        end
        return (true, chi2_ratio)
    end

    return (false, chi2_ratio)
end
