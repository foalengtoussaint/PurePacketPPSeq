# mechanisms/m5_ou.jl - Level m=5 Continuous Ornstein-Uhlenbeck Adaptation & Conjugate Triad
#
# Generative Role:
# - Continuous-time Gauss-Markov stochastic log-gains:
#     Leg 1: Regional arousal x_r(t), single-cell micro-drift delta_x_c(t), motif engagement y_m(t).
#     Leg 2: Motor vigor shape anchor alpha_c == beta_c == 10.0 (E[c_e] == 1.0).
#     Leg 3: Continuous Student-t scale mixture precisions tau_e | u_e ~ Gamma((nu+R)/2, (nu+||u_e||^2)/2).
# - Universal O(B) continuous Newton-Raphson Laplace smoother via Thomas tridiagonal algorithm.

# Continuous Ornstein-Uhlenbeck tridiagonal solver & intensity updates

"""
    solve_sym_tridiagonal!(y, diag, off, b, c_p, d_p)

Exact O(N) Thomas algorithm solver for symmetric tridiagonal system A y = b.
Zero heap allocations when workspace buffers are preallocated.
"""
function solve_sym_tridiagonal!(
    y::AbstractVector{Float64},
    diag::AbstractVector{Float64},
    off::AbstractVector{Float64},
    b::AbstractVector{Float64},
    c_p::AbstractVector{Float64},
    d_p::AbstractVector{Float64}
)
    N = length(diag)
    N == 0 && return y
    if N == 1
        y[1] = b[1] / diag[1]
        return y
    end

    c_p[1] = off[1] / diag[1]
    d_p[1] = b[1] / diag[1]
    @inbounds for j in 2:N
        denom = diag[j] - off[j-1] * c_p[j-1]
        if j < N
            c_p[j] = off[j] / denom
        end
        d_p[j] = (b[j] - off[j-1] * d_p[j-1]) / denom
    end

    y[N] = d_p[N]
    @inbounds for j in (N-1):-1:1
        y[j] = d_p[j] - c_p[j] * y[j+1]
    end
    return y
end

"""
    smooth_point_process_intensity!(log_gains, event_times, event_generator_ids, baseline_rates, dt, tau, sig, diag_buf, off_buf, rhs_buf, c_prime, d_prime; ...)

Universal Level m=5 Gauss-Markov Laplace Smoother.
Operates identically across Scale n=0 (unmarked physical spikes) and Scale n=1 (marked occurrences).
Solves exact continuous O(B) Newton-Raphson tridiagonal system with zero-mean gauge anchoring.
"""
function smooth_point_process_intensity!(
    log_gains::Matrix{Float64},
    event_times::Vector{Float64},
    event_generator_ids::Vector{Int},
    baseline_rates::Vector{Float64},
    dt::Float64,
    tau::Float64,
    sig::Float64,
    diag_buf::AbstractVector{Float64},
    off_buf::AbstractVector{Float64},
    rhs_buf::AbstractVector{Float64},
    c_prime::AbstractVector{Float64},
    d_prime::AbstractVector{Float64};
    active_ids::Union{Nothing, Vector{Int}}=nothing,
    max_iter::Int=2,
    zero_mean_gauge::Bool=true,
    clamp_val::Float64=2.5
)
    E_tot = size(log_gains, 1)
    B = size(log_gains, 2)
    B <= 1 && return log_gains

    phi = exp(-dt / tau)
    v = (0.5 * sig * sig * tau) * (1.0 - phi * phi)
    v_inf = 0.5 * sig * sig * tau
    inv_v = 1.0 / max(1e-12, v)
    inv_v_inf = 1.0 / max(1e-12, v_inf)
    off_val = -phi * inv_v

    # 1. Linear O(N_events) streaming accumulation
    counts = zeros(Int, E_tot, B)
    N_ev = length(event_times)
    @inbounds for i in 1:N_ev
        gid = event_generator_ids[i]
        (1 <= gid <= E_tot) || continue
        t = event_times[i]
        j = clamp(floor(Int, t / dt) + 1, 1, B)
        counts[gid, j] += 1
    end

    d_buf = view(diag_buf, 1:B)
    o_buf = view(off_buf, 1:(B-1))
    r_buf = view(rhs_buf, 1:B)
    cp_buf = view(c_prime, 1:B)
    dp_buf = view(d_prime, 1:B)

    target_ids = active_ids === nothing ? (1:E_tot) : active_ids

    for gid in target_ids
        (1 <= gid <= E_tot) || continue
        lam_base = (gid <= length(baseline_rates)) ? max(baseline_rates[gid], 1e-12) : 1e-12
        lam_base <= 0.0 && continue

        for iter in 1:max_iter
            @inbounds for j in 1:B
                dt_eff = (j == 1 || j == B) ? (0.5 * dt) : dt
                x_val = log_gains[gid, j]
                d_j = lam_base * dt_eff * exp(x_val)

                j_diag = (j == 1) ? (inv_v_inf + phi * phi * inv_v) : ((j == B) ? inv_v : (1.0 + phi * phi) * inv_v)
                d_buf[j] = j_diag + d_j
                if j < B
                    o_buf[j] = off_val
                end
                r_buf[j] = Float64(counts[gid, j]) - d_j + d_j * x_val
            end

            solve_sym_tridiagonal!(
                view(log_gains, gid, :),
                d_buf, o_buf, r_buf, cp_buf, dp_buf
            )
        end

        if zero_mean_gauge
            mean_val = mean(view(log_gains, gid, :))
            log_gains[gid, :] .-= mean_val
        end
        if clamp_val > 0.0
            clamp!(view(log_gains, gid, :), -clamp_val, clamp_val)
        end
    end

    return log_gains
end

"""
    evaluate_background_log_intensity(ctx, bg_rates, c, t) -> Float64

Evaluates continuous background log-intensity on channel c at time t:
    ln lambda_c^{(0)}(t) = ln lambda_{0,c} + x_r(t) + delta x_c(t).
"""
@inline function evaluate_background_log_intensity(
    ctx::ContinuousOUContext,
    bg_rates::Vector{Float64},
    c::Int,
    t::Float64
)::Float64
    T_total_ou = isempty(ctx.t_net) ? 2114.29 : ctx.t_net[end]
    min_prior_bg = 1.0 / (1.0 + T_total_ou)
    base_rate = max(bg_rates[c], min_prior_bg)
    reg = (c <= length(ctx.channel_region)) ? ctx.channel_region[c] : 1

    # Regional arousal x_r(t)
    dt_net = ctx.dt_net
    idx_net = clamp(floor(Int, t / dt_net) + 1, 1, ctx.B_net - 1)
    alpha_net = (t - (idx_net - 1) * dt_net) / dt_net
    x_r = (1.0 - alpha_net) * ctx.x_region[reg, idx_net] + alpha_net * ctx.x_region[reg, idx_net + 1]

    # Cell-intrinsic micro-drift delta_x_c(t)
    dt_cell = ctx.dt_cell
    idx_cell = clamp(floor(Int, t / dt_cell) + 1, 1, ctx.B_cell - 1)
    alpha_cell = (t - (idx_cell - 1) * dt_cell) / dt_cell
    dx_c = (1.0 - alpha_cell) * ctx.delta_x[c, idx_cell] + alpha_cell * ctx.delta_x[c, idx_cell + 1]

    return log(base_rate) + x_r + dx_c
end

"""
    update_regional_ou!(ctx, assign, spike_t, spike_neurons, bg_rates; max_iter=2)

Leg 1 slow regional arousal x_region[r, :] for r in {1, 2} (MOp and CP).
"""
function update_regional_ou!(
    ctx::ContinuousOUContext,
    assign::Vector{SpikeAssignment},
    spike_t::Vector{Float64},
    spike_neurons::Vector{Int},
    bg_rates::Vector{Float64};
    max_iter::Int=2
)
    lambda_bar_reg = zeros(Float64, 2)
    for c in 1:length(bg_rates)
        reg = (c <= length(ctx.channel_region)) ? ctx.channel_region[c] : 1
        lambda_bar_reg[reg] += bg_rates[c]
    end

    N_spk = length(assign)
    reg_t = Float64[]
    reg_ids = Int[]
    sizehint!(reg_t, N_spk)
    sizehint!(reg_ids, N_spk)
    @inbounds for i in 1:N_spk
        if assign[i].event_id == 0
            c = spike_neurons[i]
            reg = (c <= length(ctx.channel_region)) ? ctx.channel_region[c] : 1
            push!(reg_t, spike_t[i])
            push!(reg_ids, reg)
        end
    end

    smooth_point_process_intensity!(
        ctx.x_region, reg_t, reg_ids, lambda_bar_reg,
        ctx.dt_net, ctx.tau_net, ctx.sigma_net,
        ctx.diag_buf, ctx.off_buf, ctx.rhs_buf, ctx.c_prime, ctx.d_prime;
        max_iter=max_iter, zero_mean_gauge=true, clamp_val=2.5
    )
end

"""
    update_cell_ou!(ctx, assign, spike_t, spike_neurons, bg_rates; max_iter=2)

Leg 1 single-cell micro-drift delta_x[c, :] across physical units c in 1:C.
"""
function update_cell_ou!(
    ctx::ContinuousOUContext,
    assign::Vector{SpikeAssignment},
    spike_t::Vector{Float64},
    spike_neurons::Vector{Int},
    bg_rates::Vector{Float64};
    max_iter::Int=2
)
    N_spk = length(assign)
    bg_t = Float64[]
    bg_c = Int[]
    sizehint!(bg_t, N_spk)
    sizehint!(bg_c, N_spk)
    @inbounds for i in 1:N_spk
        if assign[i].event_id == 0
            push!(bg_t, spike_t[i])
            push!(bg_c, spike_neurons[i])
        end
    end

    smooth_point_process_intensity!(
        ctx.delta_x, bg_t, bg_c, bg_rates,
        ctx.dt_cell, ctx.tau_cell, ctx.sigma_cell,
        ctx.diag_buf, ctx.off_buf, ctx.rhs_buf, ctx.c_prime, ctx.d_prime;
        max_iter=max_iter, zero_mean_gauge=true, clamp_val=2.5
    )
end

"""
    update_motif_ou!(ctx, occurrences, motifs, rates_arg=nothing; max_iter=2)

Scale n = 1: Motif engagement y_motif[m, :] across motifs.
"""
function update_motif_ou!(
    ctx::ContinuousOUContext,
    occurrences::Dict{Int, PureOccurrence},
    motifs::Dict{Int, PurePacketMotif},
    rates_arg=nothing;
    max_iter::Int=2
)
    occ_t = Float64[occ.tau for occ in values(occurrences)]
    occ_m = Int[occ.motif for occ in values(occurrences)]
    active_motifs = collect(keys(motifs))
    M = size(ctx.y_motif, 1)

    rates = if rates_arg isa Vector{Float64}
        rates_arg
    elseif rates_arg isa Level5Hyperstate
        rates_arg.lambda_m
    else
        fill(1.0, M)
    end

    smooth_point_process_intensity!(
        ctx.y_motif, occ_t, occ_m, rates,
        ctx.dt_motif, ctx.tau_motif, ctx.sigma_motif,
        ctx.diag_buf, ctx.off_buf, ctx.rhs_buf, ctx.c_prime, ctx.d_prime;
        active_ids=active_motifs, max_iter=max_iter, zero_mean_gauge=true, clamp_val=2.5
    )
end

"""
    update_vigor_shape!(ctx, occurrences; max_iter=10, tol=1e-6)

Leg 2 Motor Vigor Pin: Strictly anchored at alpha_c == 10.0 (beta_c == 10.0 => E[c_e] == 1.0).
Prevents variance inflation and ghost occurrence proliferation.
"""
function update_vigor_shape!(
    ctx::ContinuousOUContext,
    occurrences::Dict{Int, PureOccurrence};
    max_iter::Int=10,
    tol::Float64=1e-6
)
    ctx.alpha_c = 10.0
    return ctx.alpha_c
end

"""
    sample_occurrence_amplitudes!(occurrences, motifs; alpha_c=25.0, rng=default_rng())
    sample_continuous_occurrence_amplitudes!(occurrences, motifs, alpha_c; rng=default_rng())

Leg 2 motor vigor draws:
    c_e ~ Gamma(alpha_c + sum_p c_{e,p}, 1 / (alpha_c + sum_p a_{m,p})).
Supports both Vector and Dict containers and optional or positional alpha_c.
"""
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

function sample_continuous_occurrence_amplitudes!(
    occurrences::Dict{Int, PureOccurrence},
    motifs::Dict{Int, PurePacketMotif},
    alpha_c::Float64;
    rng::AbstractRNG=Random.default_rng()
)
    sample_occurrence_amplitudes!(occurrences, motifs; alpha_c=alpha_c, rng=rng)
end

function sample_continuous_occurrence_amplitudes!(
    occurrences::Union{AbstractVector{PureOccurrence}, Dict{Int, PureOccurrence}},
    motifs::Dict{Int, PurePacketMotif};
    alpha_c::Float64=25.0,
    rng::AbstractRNG=Random.default_rng()
)
    sample_occurrence_amplitudes!(occurrences, motifs; alpha_c=alpha_c, rng=rng)
end

"""
    sample_student_t_precisions!(ctx, occurrences, R; rng=default_rng())

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
