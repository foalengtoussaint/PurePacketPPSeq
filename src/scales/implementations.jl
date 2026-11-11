# scales/implementations.jl - Concrete Scale Implementations for Neural Point-Processes
#
# Specific Domain Instantiations of AbstractScale:
# - ScaleLevel{0} (NeuronBaselineScale): Continuous background Poisson point process
#   on physical recording units (MOp/CP neurons).
# - ScaleLevel{1} (NeuralSequenceScale): First-order spatiotemporal neural sequence motifs
#   on spike trains.

# =====================================================================
# 1. CONCRETE SCALE TYPES & SEMANTIC ALIASES
# =====================================================================

"""
    ScaleLevel{N} <: AbstractScale

Parameterized scale instantiation of the marked point process:
- ScaleLevel{0}: Continuous background point process on physical units.
- ScaleLevel{1}: Stereotyped sequence foreground occurrences on spikes.
"""
struct ScaleLevel{N} <: AbstractScale end

# Semantic Scale Aliases
const NeuronBaselineScale = ScaleLevel{0}
const NeuralSequenceScale = ScaleLevel{1}

# =====================================================================
# 2. SCALE 0: NEURON BASELINE IMPLEMENTATION (n = 0)
# =====================================================================

scale_order(::ScaleLevel{0}) = 0
active_m_levels(::ScaleLevel{0}) = (0, 5)
incoming_data_type(::ScaleLevel{0}) = SpikeEvent
outgoing_data_type(::ScaleLevel{0}) = Float64
scale_generator_type(::ScaleLevel{0}) = NeuronGenerator
scale_timescale(::ScaleLevel{0}) = (0.0, Inf)

"""
    evaluate_scale_intensity(::ScaleLevel{0}, channel::Int, t::Float64, bg_rates::Vector{Float64}, ou_ctx::ContinuousOUContext) -> Float64

Evaluates Scale n=0 continuous background intensity on physical channel c:
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

# =====================================================================
# 3. SCALE 1: NEURAL SEQUENCE MOTIF IMPLEMENTATION (n = 1)
# =====================================================================

scale_order(::ScaleLevel{1}) = 1
active_m_levels(::ScaleLevel{1}) = (0, 1, 2, 3, 4, 5)
incoming_data_type(::ScaleLevel{1}) = SpikeEvent
outgoing_data_type(::ScaleLevel{1}) = PureOccurrence
scale_generator_type(::ScaleLevel{1}) = PurePacketMotif
scale_timescale(::ScaleLevel{1}) = (0.010, 2.0)

"""
    evaluate_scale_intensity(::ScaleLevel{1}, carrier::PurePacketMotif, occ::PureOccurrence, channel::Int, t::Float64, active_mask::BitVector) -> Float64

Evaluates Scale n=1 sequence foreground intensity on channel c at time t during occurrence e:
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
