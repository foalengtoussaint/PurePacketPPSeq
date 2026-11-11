# types.jl - Core Point-Process Data Structures

"""
    Packet

A single spatiotemporal emission packet on physical neuron `channel`.
"""
mutable struct Packet
    id::Int                 # Local packet index p in 1:P_m
    channel::Int            # Physical neuron k in 1:N
    mu::Float64             # Canonical latency in seconds (mu_1 == 0)
    var::Float64            # Jitter variance sigma_p^2 in seconds^2
    mass::Float64           # Scalar emission amplitude a_{m,p}
end

"""
    PurePacketMotif

A motif defined strictly as a collection of P_m packets.
"""
mutable struct PurePacketMotif
    id::Int
    packets::Vector{Packet}              # Length P_m
    W::Matrix{Float64}                   # R_max x P_m deformation loadings
    chan_to_packets::Vector{Vector{Int}} # Inverted index: channel k -> [p1, p2, ...]
    W_window::Float64                    # Motif duration window in seconds
    delta_left::Float64                  # Cached global left bounding offset (<= 0.0)
    delta_right::Float64                 # Cached global right bounding offset (>= 0.0)
    chan_active::BitVector               # Length N: true if channel has packets in this motif
    chan_bounds::Vector{Tuple{Float64, Float64}} # Length N: (delta_left, delta_right) per channel
end


"""
    PureOccurrence

Dense occurrence state tracking only its motif's P_m active packets.
"""
mutable struct PureOccurrence
    id::Int
    motif::Int
    tau::Float64                         # Global onset time in seconds
    u::Vector{Float64}                   # Latent coordinates in R^R
    counts::Vector{Int}                  # Length P_m: spike counts c_{e,p}
    sum_t::Vector{Float64}               # Length P_m: sum of spike times
    sum_t2::Vector{Float64}              # Length P_m: sum of squared times
    members::Vector{Int}                 # Indices of spikes assigned to this occurrence
    amplitude::Float64                   # Occurrence gain / amplitude c_e
    regime_id::Int                       # Level m=5 task regime assignment z_e
end

function PureOccurrence(id::Int, motif::Int, tau::Float64, R_max::Int, P::Int; amplitude::Float64=1.0, regime_id::Int=1)
    return PureOccurrence(
        id, motif, tau, zeros(Float64, R_max),
        zeros(Int, P), zeros(Float64, P), zeros(Float64, P),
        Int[], amplitude, regime_id
    )
end

"""
    SpikeAssignment


Dual-index assignment of spike i to occurrence and packet.
"""
struct SpikeAssignment
    event_id::Int      # 0 for background, >0 for occurrence
    packet_idx::Int    # 0 for background, p in 1:P_m for packet
end

"""
    ContinuousOUContext

Continuous-time Level m=5 Conjugate Triad point-process adaptation engine.
Replaces discrete Markov chain regime switching and arbitrary time slicing
with continuous Gauss-Markov log-gains:
1. Leg 1:
   - Regional network arousal: x_region (MOp: r=1, CP: r=2), tau_net >= W_min, regular finite-element knots dt_net <= W_min / 5
   - Single-cell micro-drift: delta_x (C sorted units), tau_cell ~ 60s, knots dt_cell ~ 10s
   - Motif engagement: y_motif (M motifs), tau_motif ~ 20s, knots dt_motif ~ 5s
2. Leg 2: Motor vigor shape alpha_c with unit-mean pin beta_c == alpha_c => E[c_e] == 1.0
3. Leg 3: Continuous Student-t kinematics precision mixture tau_e ~ Gamma(nu/2, nu/2)
"""
mutable struct ContinuousOUContext
    # Network regional arousal: x_region[r, j], r=1 (MOp), r=2 (CP)
    dt_net::Float64
    B_net::Int
    t_net::Vector{Float64}
    x_region::Matrix{Float64}         # 2 x B_net
    tau_net::Float64                  # default 0.5 s (tau_net >= W_min)
    sigma_net::Float64                # default 0.4

    # Single-cell micro-drift: delta_x[c, j], c in 1:C
    dt_cell::Float64
    B_cell::Int
    t_cell::Vector{Float64}
    delta_x::Matrix{Float64}          # C x B_cell
    tau_cell::Float64                 # default 60.0 s
    sigma_cell::Float64               # default 0.15

    # Motif engagement: y_motif[m, j], m in 1:M
    dt_motif::Float64
    B_motif::Int
    t_motif::Vector{Float64}
    y_motif::Matrix{Float64}          # M x B_motif
    tau_motif::Float64                # default 20.0 s
    sigma_motif::Float64              # default 0.3

    # Channel anatomical region mapping: 1=MOp, 2=CP
    channel_region::Vector{Int}       # Length C (strictly C=411 pure neural units)

    # Leg 2 Vigor shape (beta_c = alpha_c, E[c_e] = 1.0)
    alpha_c::Float64                  # default 10.0

    # Leg 3 Student-t kinematics degrees of freedom
    nu_student::Float64               # default 4.0
    tau_precisions::Dict{Int, Float64} # occ.id => tau_e

    # Precomputed regional segment integrals: 2 x (B_net - 1)
    seg_integral_region::Matrix{Float64}

    # Preallocated Thomas algorithm workspace vectors to ensure ZERO heap allocations during sweeps
    diag_buf::Vector{Float64}
    off_buf::Vector{Float64}
    rhs_buf::Vector{Float64}
    c_prime::Vector{Float64}
    d_prime::Vector{Float64}
    counts_buf::Vector{Int}
end

function ContinuousOUContext(
    N_channels::Int,
    T_total::Float64;
    M_motifs::Int=50,
    cp_range::UnitRange{Int}=99:301,
    dt_net::Float64=0.05,
    tau_net::Float64=0.5,
    sigma_net::Float64=0.4,
    dt_cell::Float64=10.0,
    tau_cell::Float64=60.0,
    sigma_cell::Float64=0.15,
    dt_motif::Float64=5.0,
    tau_motif::Float64=20.0,
    sigma_motif::Float64=0.3,
    alpha_c::Float64=10.0,
    nu_student::Float64=4.0
)
    # Knot counts
    B_net = max(2, ceil(Int, T_total / dt_net) + 1)
    t_net = [(j - 1) * dt_net for j in 1:B_net]
    x_region = zeros(Float64, 2, B_net)

    B_cell = max(2, ceil(Int, T_total / dt_cell) + 1)
    t_cell = [(j - 1) * dt_cell for j in 1:B_cell]
    delta_x = zeros(Float64, N_channels, B_cell)

    B_motif = max(2, ceil(Int, T_total / dt_motif) + 1)
    t_motif = [(j - 1) * dt_motif for j in 1:B_motif]
    y_motif = zeros(Float64, M_motifs, B_motif)

    chan_reg = ones(Int, N_channels)
    for c in 1:N_channels
        if c in cp_range
            chan_reg[c] = 2 # CP
        else
            chan_reg[c] = 1 # MOp
        end
    end

    tau_precisions = Dict{Int, Float64}()
    seg_integral_region = zeros(Float64, 2, B_net - 1)

    max_B = max(B_net, B_cell, B_motif)
    diag_buf = zeros(Float64, max_B)
    off_buf = zeros(Float64, max_B - 1)
    rhs_buf = zeros(Float64, max_B)
    c_prime = zeros(Float64, max_B)
    d_prime = zeros(Float64, max_B)
    counts_buf = zeros(Int, max_B)

    return ContinuousOUContext(
        dt_net, B_net, t_net, x_region, tau_net, sigma_net,
        dt_cell, B_cell, t_cell, delta_x, tau_cell, sigma_cell,
        dt_motif, B_motif, t_motif, y_motif, tau_motif, sigma_motif,
        chan_reg, alpha_c, nu_student, tau_precisions, seg_integral_region,
        diag_buf, off_buf, rhs_buf, c_prime, d_prime, counts_buf
    )
end


mutable struct ARDManifoldState
    R_max::Int
    active_mask::BitVector
    alpha::Vector{Float64}
    a0::Float64
    b0::Float64
    prune_threshold::Float64
end

function ARDManifoldState(R_max::Int; a0::Float64=1e-3, b0::Float64=1.0e-6, prune_threshold::Float64=6500.0)
    active_mask = falses(R_max)
    active_mask[1] = true
    alpha = fill(1.0e6, R_max)
    alpha[1] = 500.0
    return ARDManifoldState(R_max, active_mask, alpha, a0, b0, prune_threshold)
end



"""
    TaskRegime

Discrete behavioral or kinematic task regime governing:
1. Continuous kinematic subspace recruitment: lambda_u in R^R (clamped to lambda_dormant if dormant)
2. Motor vigor: occurrence amplitude prior c_e ~ Gamma(alpha_c, beta_c)
3. Scale gauge anchoring: Regime 1 strictly maintains lambda_1,r == 1.0 and beta_c == alpha_c
"""
mutable struct TaskRegime
    id::Int
    R::Int
    # Kinematic subspace: dimensionless relative precision per axis
    lambda_u::Vector{Float64}         # length R, anchored to 1.0 for id == 1
    active_dims::BitVector             # true if lambda_u[r] < lambda_dormant
    # Motor vigor (Amplitude prior)
    alpha_c::Float64
    beta_c::Float64                   # anchored to alpha_c for id == 1
    # Running sufficient statistics
    assigned_occurrences::Vector{Int}  # List of occurrence IDs
end

function TaskRegime(
    id::Int, R::Int;
    alpha_c::Float64=10.0, beta_c::Float64=10.0,
    alpha_u::Float64=2.0, beta_u::Float64=2.0,
    lambda_dormant::Float64=1000.0
)
    lambda_u = fill(1.0, R)
    active_dims = trues(R)
    return TaskRegime(id, R, lambda_u, active_dims, alpha_c, beta_c, Int[])
end

# Backward-compatibility alias
const DiagonalRegime = TaskRegime

"""
    Level5Hyperstate

Maintains discrete task regimes, session motif Poisson arrival rates,
and Dirichlet-Categorical mark distributions.

"""
mutable struct Level5Hyperstate
    K::Int
    R::Int
    M::Int
    regimes::Dict{Int, TaskRegime}
    lambda_dormant::Float64            # default: 1000.0
    # Hyperpriors
    a_u::Float64; b_u::Float64         # Gamma hyperprior on lambda_u (stiffened: 5.0, 5.0)
    a_c::Float64; b_c::Float64         # Hyperprior on beta_c (10.0, 10.0)
    alpha_pi::Float64                  # Dirichlet prior on pi_m
    # Session-level motif Poisson rates & mark distributions
    lambda_m::Vector{Float64}          # length M (rates in Hz)
    pi_m::Matrix{Float64}              # M x K Dirichlet-Categorical regime weights
    a_rate::Float64; b_rate::Float64   # Gamma prior on lambda_m
    force_thresh::Float64              # Threshold on unregularized data MLE dispersion S^{data}_{k,r} (0.05)
    # Nonparametric DP parameters
    alpha_dp::Float64                  # DP concentration parameter
    a_dp::Float64; b_dp::Float64       # Hyperprior on alpha_dp ~ Gamma(a_dp, 1/b_dp)
    N_min::Int                         # Minimum cluster size to ratify (2)
end

function Level5Hyperstate(
    K::Int, R::Int, M::Int=100;
    lambda_dormant::Float64=1000.0,
    a_u::Float64=5.0, b_u::Float64=5.0,
    a_c::Float64=10.0, b_c::Float64=10.0,
    alpha_pi::Float64=1.0,
    a_rate::Float64=1.0, b_rate::Float64=1.0,
    force_thresh::Float64=0.05,
    alpha_dp::Float64=1.0,
    a_dp::Float64=1.0, b_dp::Float64=1.0,
    N_min::Int=2
)
    M_safe = max(M, 50)
    regimes = Dict{Int, TaskRegime}()
    for k in 1:K
        regimes[k] = TaskRegime(k, R; alpha_c=10.0, beta_c=10.0, lambda_dormant=lambda_dormant)
    end
    lambda_m = fill(1.0, M_safe)
    pi_m = fill(1.0 / K, M_safe, K)
    return Level5Hyperstate(
        K, R, M_safe, regimes, lambda_dormant,
        a_u, b_u, a_c, b_c, alpha_pi,
        lambda_m, pi_m, a_rate, b_rate, force_thresh,
        alpha_dp, a_dp, b_dp, N_min
    )
end

"""
    ContinuousTimeCTMC

Exact Continuous-Time Markov Chain (CTMC) state field S(t) in {1...K}
coupled to marked point process via Rao-Teh uniformization and Region-Rank-1 baseline.

Invariants & Closed Bounds:
1. Structural Nyquist Bound: beta_q = (K - 1) * alpha_q * W_min
   Guarantees E[|q_kk|] <= 1 / W_min under cold-start.
2. Information Horizon: Region-Rank-1 baseline divides probe into MOp and CP,
   cancelling channel nuisance parameters in likelihood ratios.
3. Triad Scale Anchor: Regime 1 strictly pins:
   gamma_{MOp, 1} == 1.0, gamma_{CP, 1} == 1.0, beta_{c, 1} == 10.0, Lambda_{u, 1} == I_R.
4. Auxiliary Uniformization: Omega = max_k |q_kk| + xi, with xi ~ Gamma(a_omega, 1 / T_total).
"""
mutable struct ContinuousTimeCTMC
    K::Int
    Q::Matrix{Float64}                 # K x K generator matrix (q_kj >= 0, q_kk = -sum_{j!=k} q_kj)
    Omega::Float64                     # Current uniformization rate (Hz)
    W_min::Float64                     # Shortest motif carrier duration (s)
    alpha_q::Float64                   # Gamma shape on q_kj
    beta_q::Float64                    # Gamma rate on q_kj: (K-1) * alpha_q * W_min
    a_omega::Float64                   # Gamma shape for Omega buffer xi (default: 120.0)
    channel_region::Vector{Int}        # Length C: 1=MOp, 2=CP
    gamma_tone::Matrix{Float64}        # 2 x K regional baseline multipliers (Regime 1 anchored at [1.0, 1.0])
    cand_times::Vector{Float64}        # Auxiliary candidate transition times in [0, T]
    cand_states::Vector{Int}           # Sampled discrete state S_b for interval [cand_times[b], cand_times[b+1]]
    T_k::Vector{Float64}               # Length K: empirical dwell times
    N_trans::Matrix{Int}               # K x K empirical transition counts
    N_bg_mop_k::Vector{Float64}        # Length K: background spike counts in MOp per regime
    N_bg_cp_k::Vector{Float64}         # Length K: background spike counts in CP per regime
end

function ContinuousTimeCTMC(
    K::Int, N_channels::Int;
    W_min::Float64=0.25,
    alpha_q::Float64=2.0,
    a_omega::Float64=120.0,
    T_total::Float64=2114.0,
    cp_range::UnitRange{Int}=99:301
)
    Q = zeros(Float64, K, K)
    beta_q = max(1, K - 1) * alpha_q * W_min
    prior_rate = 1.0 / (max(1, K - 1) * W_min)
    for k in 1:K
        for j in 1:K
            if j != k
                Q[k, j] = prior_rate
            end
        end
        Q[k, k] = -sum(Q[k, j] for j in 1:K if j != k)
    end

    max_exit = maximum(abs(Q[k, k]) for k in 1:K)
    Omega = max_exit + a_omega / max(1.0, T_total)

    chan_reg = ones(Int, N_channels)
    for c in 1:N_channels
        if c in cp_range
            chan_reg[c] = 2 # CP
        else
            chan_reg[c] = 1 # MOp
        end
    end

    gamma_tone = ones(Float64, 2, K) # [region, k], Regime 1 anchored to 1.0

    cand_times = Float64[0.0, T_total]
    cand_states = Int[1]
    T_k = zeros(Float64, K)
    T_k[1] = T_total
    N_trans = zeros(Int, K, K)
    N_bg_mop_k = zeros(Float64, K)
    N_bg_cp_k = zeros(Float64, K)

    return ContinuousTimeCTMC(
        K, Q, Omega, W_min, alpha_q, beta_q, a_omega,
        chan_reg, gamma_tone, cand_times, cand_states,
        T_k, N_trans, N_bg_mop_k, N_bg_cp_k
    )
end



struct AuditPacketInstance
    mu::Float64
    sigma::Float64
    mass::Float64
end
