# scales.jl - Hierarchical Scale Engine & m-Mechanism Encodings
#
# Generative Point-Process Architecture:
# - m levels (m in 0..5): Universal generative mechanisms (Rate Floor, Packet Emission,
#   Carrier Composition, Gauge Symmetry, Kinematic Manifold, Continuous OU).
# - n levels (n in 0, 1, 2...): Scale instantiations defined strictly by:
#   1. Which m mechanisms are active vs degenerate.
#   2. How the scale relates to incoming data (raw spikes vs lower-scale occurrences).

# =====================================================================
# 1. UNIVERSAL m-MECHANISMS (m in 0..5)
# =====================================================================

abstract type AbstractMechanism end

"""
    Level0RateFloor <: AbstractMechanism

Mechanism m = 0: Stationary Baseline Intensity & Poisson Compensator.
Eliminates non-identifiable rate collapse, providing session rate anchor
and unassigned spike likelihood comparison.
"""
struct Level0RateFloor <: AbstractMechanism end

"""
    Level1EmissionPacket <: AbstractMechanism

Mechanism m = 1: Pure Spatiotemporal Point-Process Emission Kernel.
Parameterizes emission mass a_p, canonical latency mu_p, and jitter variance var_p = sigma_p^2.
Governs SNR dominance radius R_p = sigma_p * sqrt(2 * ln^+(SNR_p)).
"""
struct Level1EmissionPacket <: AbstractMechanism end

"""
    Level2CarrierComposition <: AbstractMechanism

Mechanism m = 2: Sequence Carrier Composition & Dynamic Bounding Envelopes.
Organizes packets into temporal sequence carriers with dynamic envelope
[delta_left, delta_right] and physical Occam footprint W_eff.
"""
struct Level2CarrierComposition <: AbstractMechanism end

"""
    Level3GaugeSymmetry <: AbstractMechanism

Mechanism m = 3: Continuous Translational Gauge Invariance.
Strictly projects manifold loadings W onto orthogonal subspace:
    sum_p kappa_p W_{r,p} == 0
ensuring zero continuous drift between onset tau_e and kinematic deformation.
"""
struct Level3GaugeSymmetry <: AbstractMechanism end

"""
    Level4KinematicManifold <: AbstractMechanism

Mechanism m = 4: Low-Dimensional Continuous Latent Kinematics & ARD.
Projects occurrence timing residuals onto latent axes u_e in R^R with
Automatic Relevance Determination precisions alpha_r and Cramér-Rao pruning.
"""
struct Level4KinematicManifold <: AbstractMechanism end

"""
    Level5ContinuousOU <: AbstractMechanism

Mechanism m = 5: Continuous Ornstein-Uhlenbeck Conjugate Triad.
Provides continuous non-stationary arousal x_r(t), single-cell micro-drift delta_x_c(t),
motor vigor anchor (alpha_c == beta_c == 10.0), and Student-t scale mixture precisions tau_e | u_e.
"""
struct Level5ContinuousOU <: AbstractMechanism end

# Semantic Mechanism Aliases
const RateFloor = Level0RateFloor
const EmissionPacketMechanism = Level1EmissionPacket
const CarrierComposition = Level2CarrierComposition
const GaugeSymmetry = Level3GaugeSymmetry
const KinematicManifold = Level4KinematicManifold
const ContinuousOU = Level5ContinuousOU

# =====================================================================
# 2. SCALE INSTANTIATIONS (n in 0, 1, 2...)
# =====================================================================

abstract type AbstractScaleLevel end

"""
    ScaleLevel{N} <: AbstractScaleLevel

Scale N instantiation of the marked point process:
- ScaleLevel{0}: Continuous background Poisson point process on physical units.
- ScaleLevel{1}: Stereotyped sequence foreground occurrences on spikes.
- ScaleLevel{2}: Meta-sequence repertoire on Scale 1 sequence occurrences.
"""
struct ScaleLevel{N} <: AbstractScaleLevel end

# Query active m levels
active_m_levels(::ScaleLevel{0}) = (0, 5)
active_m_levels(::ScaleLevel{1}) = (0, 1, 2, 3, 4, 5)
active_m_levels(::ScaleLevel{2}) = (0, 1, 2, 3, 4, 5)

"""
    is_m_active(scale::ScaleLevel{N}, m::Int) -> Bool

Returns true if mechanism m in 0..5 is active at Scale N.
"""
@inline is_m_active(scale::ScaleLevel{N}, m::Int) where N = m in active_m_levels(scale)

# Query incoming and outgoing data contracts
incoming_data_type(::ScaleLevel{0}) = SpikeEvent
incoming_data_type(::ScaleLevel{1}) = SpikeEvent
incoming_data_type(::ScaleLevel{2}) = PureOccurrence

outgoing_data_type(::ScaleLevel{0}) = Float64
outgoing_data_type(::ScaleLevel{1}) = PureOccurrence
outgoing_data_type(::ScaleLevel{2}) = MetaOccurrence

# =====================================================================
# 3. HIGHER-ORDER SCALE n=2 EVENT CONTRACT (META-OCCURRENCE)
# =====================================================================

"""
    MetaOccurrence <: AbstractPointProcessEvent

Scale n = 2 Event: Higher-order sequence motif occurrence (repertoire chunk).
Consumes Scale n=1 occurrences as its input point-process events.
Carries higher-order timing Tau, meta-motif ID, latent deformation U, and member occurrence IDs.
"""
mutable struct MetaOccurrence <: AbstractPointProcessEvent
    id::Int
    meta_motif::Int
    tau::Float64                         # Global onset time in seconds
    u::Vector{Float64}                   # Higher-order latent coordinates in R^R
    members::Vector{Int}                 # IDs of Scale n=1 occurrences assigned to this meta-occurrence
    amplitude::Float64                   # Meta-occurrence vigor C
end

function MetaOccurrence(id::Int, meta_motif::Int, tau::Float64, R_max::Int=2; amplitude::Float64=1.0)
    return MetaOccurrence(id, meta_motif, tau, zeros(Float64, R_max), Int[], amplitude)
end

# =====================================================================
# 4. SCALE-DISPATCHED INTENSITY EVALUATION
# =====================================================================

"""
    evaluate_scale_intensity(::ScaleLevel{0}, channel::Int, t::Float64, bg_rates::Vector{Float64}, ou_ctx::ContinuousOUContext) -> Float64

Evaluates Scale n=0 continuous background intensity:
    lambda_c^{(0)}(t) = bar{lambda}_{0,c} * exp(x_r(t) + delta x_c(t))
"""
@inline function evaluate_scale_intensity(
    ::ScaleLevel{0},
    channel::Int,
    t::Float64,
    bg_rates::Vector{Float64},
    ou_ctx::ContinuousOUContext
)::Float64
    log_intensity = evaluate_background_log_intensity(ou_ctx, bg_rates, channel, t)
    return exp(log_intensity)
end

"""
    evaluate_scale_intensity(::ScaleLevel{1}, carrier::PurePacketMotif, occ::PureOccurrence, channel::Int, t::Float64, active_mask::BitVector) -> Float64

Evaluates Scale n=1 sequence foreground intensity on channel c at time t:
    lambda_{e,c}^{(1)}(t) = c_e * sum_{p: chan(p)=c} a_{m,p} * N(t; tau_e + mu_p + W_{:,p}^T u_e, sigma_p^2)
"""
function evaluate_scale_intensity(
    ::ScaleLevel{1},
    carrier::PurePacketMotif,
    occ::PureOccurrence,
    channel::Int,
    t::Float64,
    active_mask::BitVector
)::Float64
    channel in 1:length(carrier.chan_to_packets) || return 0.0
    pkts = carrier.chan_to_packets[channel]
    isempty(pkts) && return 0.0

    R_m = size(carrier.W, 1)
    c_e = occ.amplitude
    total_rate = 0.0

    for p in pkts
        pkt = carrier.packets[p]
        shift = 0.0
        for r in 1:min(R_m, length(active_mask))
            if active_mask[r] && r <= length(occ.u)
                shift += carrier.W[r, p] * occ.u[r]
            end
        end
        mu_eff = occ.tau + pkt.mu + shift
        dt = t - mu_eff
        # Gaussian emission kernel: a_{m,p} * N(dt; 0, var_p)
        intensity = pkt.mass * exp(-0.5 * dt * dt / pkt.var) / sqrt(2π * pkt.var)
        total_rate += intensity
    end

    return c_e * total_rate
end
