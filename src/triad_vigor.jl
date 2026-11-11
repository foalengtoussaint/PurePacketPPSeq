# triad_vigor.jl - Conjugate Triad Motor Vigor & Student-t Precisions

function sample_occurrence_amplitudes!(
    occurrences::Union{AbstractVector{PureOccurrence}, Dict{Int, PureOccurrence}},
    motifs::Dict{Int, PurePacketMotif};
    alpha_c::Float64=25.0,
    rng::AbstractRNG=Random.default_rng()
)
    occ_iter = occurrences isa Dict ? values(occurrences) : occurrences
    motif_total_mass = Dict{Int, Float64}()
    for (mid, m) in motifs
        motif_total_mass[mid] = sum(p.mass for p in m.packets; init=0.0)
    end

    for occ in occ_iter
        m_mass = get(motif_total_mass, occ.motif, 1.0)
        N_e = sum(occ.counts; init=0)
        shape_post = alpha_c + Float64(N_e)
        rate_post = alpha_c + m_mass
        occ.amplitude = rand(rng, Gamma(shape_post, 1.0 / max(rate_post, 1e-12)))
    end
end


"""
    update_packet_corridors!(m::PurePacketMotif, occurrences::Vector{PureOccurrence}, active_mask; anchor_origin=true, damp=0.2, nu_0=10.0, S_0=0.00025)

Updates packet canonical latencies mu_p and jitter variances var_p with origin anchoring mu_1 == 0.
Returns the origin_offset to shift occurrences (anchor handover).
"""

function update_vigor_shape!(
    ctx::ContinuousOUContext,
    occurrences::Dict{Int, PureOccurrence};
    max_iter::Int=10,
    tol::Float64=1e-6
)
    # Leg 2 Motor Vigor Pin: Strictly anchored at alpha_c == 10.0 (beta_c == 10.0 => E[c_e] == 1.0)
    # Prevents variance inflation and ghost occurrence proliferation across random noise.
    ctx.alpha_c = 10.0
    return ctx.alpha_c
end

"""
    sample_continuous_occurrence_amplitudes!(
        occurrences::Dict{Int, PureOccurrence},
        motifs::Dict{Int, PurePacketMotif},
        alpha_c::Float64;
        rng::AbstractRNG=Random.default_rng()
    )

Conjugate Gamma draws for occurrence amplitudes with unit-mean prior beta_c == alpha_c:
    c_e ~ Gamma(alpha_c + sum_p c_{e,p}, 1 / (alpha_c + sum_p a_{m,p})).
"""
function sample_continuous_occurrence_amplitudes!(
    occurrences::Dict{Int, PureOccurrence},
    motifs::Dict{Int, PurePacketMotif},
    alpha_c::Float64;
    rng::AbstractRNG=Random.default_rng()
)
    for occ in values(occurrences)
        haskey(motifs, occ.motif) || continue
        pm = motifs[occ.motif]
        tot_spikes = sum(occ.counts)
        tot_mass = sum(p.mass for p in pm.packets; init=0.0)

        shape_post = alpha_c + Float64(tot_spikes)
        rate_post = alpha_c + tot_mass
        occ.amplitude = rand(rng, Gamma(shape_post, 1.0 / max(rate_post, 1e-12)))
    end
end

"""
    sample_student_t_precisions!(
        ctx::ContinuousOUContext,
        occurrences::Dict{Int, PureOccurrence},
        R::Int;
        rng::AbstractRNG=Random.default_rng()
    )

Leg 3 continuous scale-mixture Student-t kinematics precision draws:
    tau_e | u_e ~ Gamma((nu + R) / 2, 2 / (nu + ||u_e||^2)).
"""
function sample_student_t_precisions!(
    ctx::ContinuousOUContext,
    occurrences::Dict{Int, PureOccurrence},
    R::Int;
    rng::AbstractRNG=Random.default_rng()
)
    nu = ctx.nu_student
    shape_post = 0.5 * (nu + Float64(R))
    for occ in values(occurrences)
        norm_sq = sum(abs2, occ.u[1:R])
        rate_post = 0.5 * (nu + norm_sq)
        ctx.tau_precisions[occ.id] = rand(rng, Gamma(shape_post, 1.0 / max(rate_post, 1e-12)))
    end
end

