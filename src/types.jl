# types.jl - Core Point-Process Data Structures

# =====================================================================
# ABSTRACT TWO-SCALE MARKED POINT-PROCESS HIERARCHY
# =====================================================================

abstract type AbstractPointProcessGenerator end
abstract type AbstractPointProcessEvent end

"""
    NeuronGenerator <: AbstractPointProcessGenerator

Scale n = 0 Generator: Physical neuron c in 1:N.
Its ensemble template (m=2) is trivially size 1 (itself), its gauge (m=3) is identity,
its manifold (m=4) is identity (R=0), and its continuous rate (m=5) is delta_x_c(t).
"""
struct NeuronGenerator <: AbstractPointProcessGenerator
    id::Int                 # Channel index in 1:N
    region::Int             # 1 for MOp, 2 for CP
end

"""
    SpikeEvent <: AbstractPointProcessEvent

Scale n = 0 Event: Physical spike click at time t_i.
Carries zero internal payload (unmarked).
"""
struct SpikeEvent <: AbstractPointProcessEvent
    t::Float64              # Timestamp in seconds
    generator_id::Int       # Physical neuron that emitted it
end

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
    PurePacketMotif <: AbstractPointProcessGenerator

Scale n = 1 Generator: A motif defined strictly as a collection of P_m packets.
"""
mutable struct PurePacketMotif <: AbstractPointProcessGenerator
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
    PureOccurrence <: AbstractPointProcessEvent

Scale n = 1 Event: Dense occurrence state tracking only its motif's P_m active packets.
Marked event carrying internal payload: vigor c_e, spatial coordinate u_e, precision tau_e.
"""
mutable struct PureOccurrence <: AbstractPointProcessEvent
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



# =====================================================================
# LEGACY CHECKPOINT COMPATIBILITY STRUCTS
# (Deprecated: Replaced by Continuous OU Context & Conjugate Triad)
# =====================================================================

"""
    TaskRegime (Deprecated)
Retained strictly for binary deserialization backward-compatibility with older .jls checkpoints.
"""
mutable struct TaskRegime
    id::Int
    R::Int
    lambda_u::Vector{Float64}
    active_dims::BitVector
    alpha_c::Float64
    beta_c::Float64
    assigned_occurrences::Vector{Int}
end

function TaskRegime(id::Int, R::Int; alpha_c::Float64=10.0, beta_c::Float64=10.0, lambda_dormant::Float64=1000.0)
    return TaskRegime(id, R, fill(1.0, R), trues(R), alpha_c, beta_c, Int[])
end

const DiagonalRegime = TaskRegime

"""
    Level5Hyperstate (Deprecated)
Retained strictly for binary deserialization backward-compatibility with older .jls checkpoints.
"""
mutable struct Level5Hyperstate
    K::Int
    R::Int
    M::Int
    regimes::Dict{Int, TaskRegime}
    lambda_dormant::Float64
    a_u::Float64; b_u::Float64
    a_c::Float64; b_c::Float64
    alpha_pi::Float64
    lambda_m::Vector{Float64}
    pi_m::Matrix{Float64}
    a_rate::Float64; b_rate::Float64
    force_thresh::Float64
    alpha_dp::Float64
    a_dp::Float64; b_dp::Float64
    N_min::Int
end

function Level5Hyperstate(K::Int, R::Int, M::Int=100; lambda_dormant::Float64=1000.0, a_u::Float64=5.0, b_u::Float64=5.0, a_c::Float64=10.0, b_c::Float64=10.0, alpha_pi::Float64=1.0, a_rate::Float64=1.0, b_rate::Float64=1.0, force_thresh::Float64=0.05, alpha_dp::Float64=1.0, a_dp::Float64=1.0, b_dp::Float64=1.0, N_min::Int=2)
    M_safe = max(M, 50)
    regimes = Dict{Int, TaskRegime}()
    for k in 1:K
        regimes[k] = TaskRegime(k, R; alpha_c=10.0, beta_c=10.0, lambda_dormant=lambda_dormant)
    end
    return Level5Hyperstate(K, R, M_safe, regimes, lambda_dormant, a_u, b_u, a_c, b_c, alpha_pi, fill(1.0, M_safe), fill(1.0 / K, M_safe, K), a_rate, b_rate, force_thresh, alpha_dp, a_dp, b_dp, N_min)
end



struct AuditPacketInstance
    mu::Float64
    sigma::Float64
    mass::Float64
end
