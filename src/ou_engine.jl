# ou_engine.jl - Continuous Ornstein-Uhlenbeck Adaptation Engine

# =====================================================================
# 11. CONTINUOUS ORNSTEIN-UHLENBECK CONJUGATE TRIAD ADAPTATION ENGINE
# =====================================================================

"""
    exp_integral_segment(x1::Float64, x2::Float64, dt::Float64)

Analytical piecewise-linear exponential integral:
    int_0^dt exp(x1 + (s/dt)*(x2 - x1)) ds = dt * (exp(x2) - exp(x1)) / (x2 - x1)
With Taylor expansion safeguard for |Delta x| < 10^-4 achieving full IEEE-754 machine precision.
"""
@inline function exp_integral_segment(x1::Float64, x2::Float64, dt::Float64)::Float64
    dx = x2 - x1
    if abs(dx) < 1e-4
        factor = 1.0 + dx * (0.5 + dx * (1.0/6.0 + dx / 24.0))
        return dt * exp(x1) * factor
    else
        return dt * (exp(x2) - exp(x1)) / dx
    end
end

"""
    solve_sym_tridiagonal!(y, diag, off, b, c_p, d_p)

Exact O(N) Thomas algorithm solver for symmetric tridiagonal system A y = b.
Zero heap allocations when workspace vectors are preallocated.
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
    evaluate_background_log_intensity(ctx::ContinuousOUContext, bg_rates::Vector{Float64}, c::Int, t::Float64)

O(1) continuous background log-intensity evaluation:
    ln lambda_c^(0)(t) = ln lambda_0,c + x_region(c)(t) + delta_x_c(t).
"""
@inline function evaluate_background_log_intensity(
    ctx::ContinuousOUContext,
    bg_rates::Vector{Float64},
    c::Int,
    t::Float64
)::Float64
    base_rate = max(bg_rates[c], 1e-12)
    reg = (c <= length(ctx.channel_region)) ? ctx.channel_region[c] : 1

    # 1. Regional arousal x_r(t)
    dt_net = ctx.dt_net
    idx_net = clamp(floor(Int, t / dt_net) + 1, 1, ctx.B_net - 1)
    alpha_net = (t - (idx_net - 1) * dt_net) / dt_net
    x_r = (1.0 - alpha_net) * ctx.x_region[reg, idx_net] + alpha_net * ctx.x_region[reg, idx_net + 1]

    # 2. Cell-intrinsic micro-drift delta_x_c(t)
    dt_cell = ctx.dt_cell
    idx_cell = clamp(floor(Int, t / dt_cell) + 1, 1, ctx.B_cell - 1)
    alpha_cell = (t - (idx_cell - 1) * dt_cell) / dt_cell
    dx_c = (1.0 - alpha_cell) * ctx.delta_x[c, idx_cell] + alpha_cell * ctx.delta_x[c, idx_cell + 1]

    return log(base_rate) + x_r + dx_c
end

"""
    evaluate_integrated_background(ctx::ContinuousOUContext, bg_rate_c::Float64, c::Int, t1::Float64, t2::Float64)

Evaluates int_{t1}^{t2} lambda_c^(0)(s) ds for Time-Rescaling theorem audit.
"""
@inline function evaluate_integrated_background(
    ctx::ContinuousOUContext,
    bg_rate_c::Float64,
    c::Int,
    t1::Float64,
    t2::Float64
)::Float64
    t2 <= t1 && return 0.0
    dt = t2 - t1
    reg = (c <= length(ctx.channel_region)) ? ctx.channel_region[c] : 1

    # Midpoint log-gain
    t_mid = 0.5 * (t1 + t2)
    dt_net = ctx.dt_net
    idx_net = clamp(floor(Int, t_mid / dt_net) + 1, 1, ctx.B_net - 1)
    alpha_net = (t_mid - (idx_net - 1) * dt_net) / dt_net
    x_r = (1.0 - alpha_net) * ctx.x_region[reg, idx_net] + alpha_net * ctx.x_region[reg, idx_net + 1]

    dt_cell = ctx.dt_cell
    idx_cell = clamp(floor(Int, t_mid / dt_cell) + 1, 1, ctx.B_cell - 1)
    alpha_cell = (t_mid - (idx_cell - 1) * dt_cell) / dt_cell
    dx_c = (1.0 - alpha_cell) * ctx.delta_x[c, idx_cell] + alpha_cell * ctx.delta_x[c, idx_cell + 1]

    return bg_rate_c * dt * exp(x_r + dx_c)
end

"""
    smooth_point_process_intensity!(
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

Universal Level m=5 Gauss-Markov Laplace Smoother:
Operates identically across Scale n=0 (unmarked physical spikes from neurons) and Scale n=1 (marked occurrences from motifs).
Solves the exact O(B) continuous Newton-Raphson Laplace approximation with zero-mean gauge anchoring.
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

    # 1. Linear O(N_events) streaming event accumulation
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
    update_regional_ou!(
        ctx::ContinuousOUContext,
        assign::Vector{SpikeAssignment},
        spike_t::Vector{Float64},
        spike_neurons::Vector{Int},
        bg_rates::Vector{Float64};
        max_iter::Int=2
    )

Regional arousal x_region[r, :] for r in {1, 2} (MOp and CP) evaluated via smooth_point_process_intensity!.
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
    update_cell_ou!(
        ctx::ContinuousOUContext,
        assign::Vector{SpikeAssignment},
        spike_t::Vector{Float64},
        spike_neurons::Vector{Int},
        bg_rates::Vector{Float64};
        max_iter::Int=2
    )

Scale n = 0: Single-cell micro-drift delta_x[c, :] across all units c in 1:C evaluated via smooth_point_process_intensity!.
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
    update_motif_ou!(
        ctx::ContinuousOUContext,
        occurrences::Dict{Int, PureOccurrence},
        motifs::Dict{Int, PurePacketMotif},
        rates_arg=nothing;
        max_iter::Int=2
    )

Scale n = 1: Motif engagement y_motif[m, :] across motifs evaluated via smooth_point_process_intensity!.
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
    update_continuous_background_rates!(
        bg_rates::Vector{Float64},
        ctx::ContinuousOUContext,
        counts_bg::Vector{Int},
        assign::Vector{SpikeAssignment},
        neuron::Vector{Int},
        T_total::Float64;
        alpha_bg::Float64=1.0,
        beta_bg::Float64=1.0,
        rng::AbstractRNG=Random.default_rng()
    )

Exact conjugate Gamma rate draws for pure neural units c in 1:C:
    lambda_0,c ~ Gamma(alpha_bg + N_bg(c), 1 / (beta_bg + T_eff,c))
"""
function update_continuous_background_rates!(
    bg_rates::Vector{Float64},
    ctx::ContinuousOUContext,
    counts_bg::Vector{Int},
    assign::Vector{SpikeAssignment},
    neuron::Vector{Int},
    T_total::Float64;
    alpha_bg::Float64=1.0,
    beta_bg::Float64=1.0,
    rng::AbstractRNG=Random.default_rng()
)
    C = length(bg_rates)
    fill!(counts_bg, 0)
    @inbounds for i in eachindex(assign)
        if assign[i].event_id == 0
            c = neuron[i]
            if 1 <= c <= C
                counts_bg[c] += 1
            end
        end
    end

    # 1. Precompute regional segment integrals
    B_net = ctx.B_net
    dt_net = ctx.dt_net
    for reg in 1:2
        @inbounds for j in 1:(B_net - 1)
            ctx.seg_integral_region[reg, j] = exp_integral_segment(
                ctx.x_region[reg, j], ctx.x_region[reg, j+1], dt_net
            )
        end
    end

    # 2. Precompute regional sums per cell interval
    B_cell = ctx.B_cell
    dt_cell = ctx.dt_cell
    seg_per_cell = dt_cell / dt_net
    reg_sum_per_cell = zeros(Float64, 2, B_cell - 1)
    for reg in 1:2
        for k in 1:(B_cell - 1)
            j_start = floor(Int, (k - 1) * seg_per_cell) + 1
            j_end = min(B_net - 1, floor(Int, k * seg_per_cell))
            s = 0.0
            @inbounds for j in j_start:j_end
                s += ctx.seg_integral_region[reg, j]
            end
            reg_sum_per_cell[reg, k] = s
        end
    end

    # 3. Draw conjugate lambda_0,c
    for c in 1:C
        reg = (c <= length(ctx.channel_region)) ? ctx.channel_region[c] : 1
        T_eff_c = 0.0
        @inbounds for k in 1:(B_cell - 1)
            dx_mid = 0.5 * (ctx.delta_x[c, k] + ctx.delta_x[c, k+1])
            T_eff_c += exp(dx_mid) * reg_sum_per_cell[reg, k]
        end
        T_eff_c = max(1e-6, T_eff_c)

        shape_c = alpha_bg + Float64(counts_bg[c])
        rate_c = beta_bg + T_eff_c
        bg_rates[c] = rand(rng, Gamma(shape_c, 1.0 / rate_c))
    end
end

"""
    update_vigor_shape!(ctx::ContinuousOUContext, occurrences::Dict{Int, PureOccurrence}; max_iter::Int=10, tol::Float64=1e-6)

Leg 2 unit-mean motor vigor pin: beta_c == alpha_c => E[c_e] == 1.0.
Updates shape alpha_c via 1D Newton-Raphson on Gamma log-likelihood.
"""
