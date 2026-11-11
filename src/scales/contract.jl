# scales/contract.jl - Formal Mathematical Scale Contract in Point-Process Theory
#
# Generative Foundation:
# A Scale S is an autonomous generative layer in a marked point-process hierarchy.
# It defines how incoming point events X_in are consumed, how active physical
# mechanisms M_active are combined, how generators G parameterize emissions,
# and how marked occurrences X_out are produced for the scale above.
#
# This file defines the pure abstract contract with zero coupling to specific
# electrophysiological recording hardware or spike formats.

# =====================================================================
# 1. UNIVERSAL m-MECHANISMS (m in 0..5)
# =====================================================================

"""
    AbstractMechanism

Universal generative mechanism family defining the physical laws of the point process.
"""
abstract type AbstractMechanism end

"""
    Level0RateFloor <: AbstractMechanism

Mechanism m = 0: Stationary Baseline Floor & Poisson Compensator.
Guarantees session rate anchor and unassigned spike likelihood comparison.
"""
struct Level0RateFloor <: AbstractMechanism end

"""
    Level1EmissionPacket <: AbstractMechanism

Mechanism m = 1: Pure Spatiotemporal Point-Process Emission Kernel.
Parameterizes latency mu_p, jitter variance var_p, and emission mass a_p.
"""
struct Level1EmissionPacket <: AbstractMechanism end

"""
    Level2CarrierComposition <: AbstractMechanism

Mechanism m = 2: Carrier Composition & Dynamic Envelopes.
Composes packets into sequence carriers with bounding envelopes [delta_left, delta_right].
"""
struct Level2CarrierComposition <: AbstractMechanism end

"""
    Level3GaugeSymmetry <: AbstractMechanism

Mechanism m = 3: Continuous Translational Gauge Invariance.
Strictly projects loadings W onto orthogonal subspace: sum_p kappa_p W_{r,p} == 0.
"""
struct Level3GaugeSymmetry <: AbstractMechanism end

"""
    Level4KinematicManifold <: AbstractMechanism

Mechanism m = 4: Low-Dimensional Continuous Latent Kinematics & ARD.
Projects occurrence timing residuals onto latent axes u_e in R^R.
"""
struct Level4KinematicManifold <: AbstractMechanism end

"""
    Level5ContinuousOU <: AbstractMechanism

Mechanism m = 5: Continuous Ornstein-Uhlenbeck Conjugate Triad.
Provides continuous network arousal, single-cell micro-drift, motor vigor anchor,
and Student-t scale mixture precisions.
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
# 2. ABSTRACT SCALE CONTRACT (WHAT A SCALE IS AT ALL)
# =====================================================================

"""
    AbstractScale

Abstract supertype for an autonomous scale level in a marked point-process hierarchy.
Any concrete scale must implement the formal scale contract:
- `scale_order(s)` -> Integer depth n >= 0
- `incoming_data_type(s)` -> Type of point-process events consumed
- `outgoing_data_type(s)` -> Type of marked entities emitted
- `active_m_levels(s)` -> Tuple of active mechanisms m in 0..5
- `scale_generator_type(s)` -> Type of generator structure parameterizing emissions
- `scale_timescale(s)` -> Characteristic temporal support window (T_min, T_max) in seconds
- `evaluate_scale_intensity(s, ...)` -> Instantaneous event arrival rate lambda*(t, mark | H_t)
"""
abstract type AbstractScale end

const AbstractScaleLevel = AbstractScale # Backward compatibility alias

# Default Fallbacks & Trait Queries

"""
    scale_order(scale::AbstractScale) -> Int

Returns the hierarchical scale index n (e.g. 0 for baseline channel level, 1 for sequence motif level).
"""
function scale_order end

"""
    incoming_data_type(scale::AbstractScale) -> Type

Returns the type of point-process events consumed by this scale from the scale below.
"""
function incoming_data_type end

"""
    outgoing_data_type(scale::AbstractScale) -> Type

Returns the type of marked events / intensities emitted by this scale to the scale above.
"""
function outgoing_data_type end

"""
    active_m_levels(scale::AbstractScale) -> Tuple{Vararg{Int}}

Returns the tuple of universal mechanisms m in 0..5 that are active at this scale.
"""
function active_m_levels end

"""
    scale_generator_type(scale::AbstractScale) -> Type

Returns the generator type that parameterizes emissions at this scale.
"""
function scale_generator_type end

"""
    scale_timescale(scale::AbstractScale) -> Tuple{Float64, Float64}

Returns the characteristic temporal resolution / support window (T_min, T_max) in seconds.
"""
function scale_timescale end

"""
    is_m_active(scale::AbstractScale, m::Int) -> Bool

Returns true if mechanism m in 0..5 is active at this scale.
"""
@inline is_m_active(scale::AbstractScale, m::Int)::Bool = m in active_m_levels(scale)

"""
    evaluate_scale_intensity(scale::AbstractScale, args...)

Evaluates the conditional point-process intensity functional lambda*(t, mark | H_t)
by executing the active m-mechanisms for this scale.
"""
function evaluate_scale_intensity end

"""
    verify_scale_contract(scale::AbstractScale)::Bool

Validates that a scale implementation satisfies the formal abstract contract.
"""
function verify_scale_contract(scale::AbstractScale)::Bool
    n = scale_order(scale)
    n >= 0 || return false
    m_acts = active_m_levels(scale)
    all(m in 0:5 for m in m_acts) || return false
    in_t = incoming_data_type(scale)
    in_t isa Type || return false
    out_t = outgoing_data_type(scale)
    out_t isa Type || return false
    gen_t = scale_generator_type(scale)
    gen_t isa Type || return false
    t_min, t_max = scale_timescale(scale)
    t_max >= t_min >= 0.0 || return false
    return true
end
