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
    update_regional_ou!(
        ctx::ContinuousOUContext,
        assign::Vector{SpikeAssignment},
        spike_t::Vector{Float64},
        spike_neurons::Vector{Int},
        bg_rates::Vector{Float64};
        max_iter::Int=2
    )

Newton-Raphson Laplace smoothing on regional arousal x_region[r, :] for r in {1, 2} (MOp and CP).
Exact O(B_net) tridiagonal solve via Thomas algorithm.
"""
function update_regional_ou!(
    ctx::ContinuousOUContext,
    assign::Vector{SpikeAssignment},
    spike_t::Vector{Float64},
    spike_neurons::Vector{Int},
    bg_rates::Vector{Float64};
    max_iter::Int=2
)
    B = ctx.B_net
    dt = ctx.dt_net
    tau = ctx.tau_net
    sig = ctx.sigma_net

    phi = exp(-dt / tau)
    v = (0.5 * sig * sig * tau) * (1.0 - phi * phi)
    v_inf = 0.5 * sig * sig * tau
    inv_v = 1.0 / max(1e-12, v)
    inv_v_inf = 1.0 / max(1e-12, v_inf)
    off_val = -phi * inv_v

    # Sum baseline rates per region
    lambda_bar_mop = 0.0
    lambda_bar_cp = 0.0
    for c in 1:length(bg_rates)
        reg = (c <= length(ctx.channel_region)) ? ctx.channel_region[c] : 1
        if reg == 1
            lambda_bar_mop += bg_rates[c]
        else
            lambda_bar_cp += bg_rates[c]
        end
    end

    N_spk = length(assign)

    for reg in 1:2
        lambda_bar = reg == 1 ? lambda_bar_mop : lambda_bar_cp
        lambda_bar <= 0.0 && continue

        # 1. Accumulate unassigned background spikes for region
        fill!(view(ctx.counts_buf, 1:B), 0)
        @inbounds for i in 1:N_spk
            assign[i].event_id == 0 || continue
            c = spike_neurons[i]
            r_c = (c <= length(ctx.channel_region)) ? ctx.channel_region[c] : 1
            r_c == reg || continue

            t = spike_t[i]
            j = clamp(floor(Int, t / dt) + 1, 1, B)
            ctx.counts_buf[j] += 1
        end

        # 2. Newton-Raphson iterations
        for iter in 1:max_iter
            @inbounds for j in 1:B
                dt_eff = (j == 1 || j == B) ? (0.5 * dt) : dt
                x_val = ctx.x_region[reg, j]
                d_j = lambda_bar * dt_eff * exp(x_val)

                j_diag = (j == 1) ? (inv_v_inf + phi * phi * inv_v) : ((j == B) ? inv_v : (1.0 + phi * phi) * inv_v)

                ctx.diag_buf[j] = j_diag + d_j
                if j < B
                    ctx.off_buf[j] = off_val
                end
                ctx.rhs_buf[j] = Float64(ctx.counts_buf[j]) - d_j + d_j * x_val
            end

            solve_sym_tridiagonal!(
                view(ctx.x_region, reg, :),
                view(ctx.diag_buf, 1:B),
                view(ctx.off_buf, 1:(B-1)),
                view(ctx.rhs_buf, 1:B),
                view(ctx.c_prime, 1:B),
                view(ctx.d_prime, 1:B)
            )
        end

        # Zero-mean gauge condition: (1/B) sum x_region == 0
        mean_x = mean(view(ctx.x_region, reg, :))
        ctx.x_region[reg, :] .-= mean_x
        clamp!(view(ctx.x_region, reg, :), -2.5, 2.5)
    end
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

Newton-Raphson Laplace smoothing on single-cell micro-drift delta_x[c, :] across all units c in 1:C.
"""
function update_cell_ou!(
    ctx::ContinuousOUContext,
    assign::Vector{SpikeAssignment},
    spike_t::Vector{Float64},
    spike_neurons::Vector{Int},
    bg_rates::Vector{Float64};
    max_iter::Int=2
)
    C = size(ctx.delta_x, 1)
    B = ctx.B_cell
    dt = ctx.dt_cell
    tau = ctx.tau_cell
    sig = ctx.sigma_cell

    phi = exp(-dt / tau)
    v = (0.5 * sig * sig * tau) * (1.0 - phi * phi)
    v_inf = 0.5 * sig * sig * tau
    inv_v = 1.0 / max(1e-12, v)
    inv_v_inf = 1.0 / max(1e-12, v_inf)
    off_val = -phi * inv_v

    cell_counts = zeros(Int, C, B)
    N_spk = length(assign)
    @inbounds for i in 1:N_spk
        assign[i].event_id == 0 || continue
        c = spike_neurons[i]
        (1 <= c <= C) || continue
        t = spike_t[i]
        j = clamp(floor(Int, t / dt) + 1, 1, B)
        cell_counts[c, j] += 1
    end

    d_buf = view(ctx.diag_buf, 1:B)
    o_buf = view(ctx.off_buf, 1:(B-1))
    r_buf = view(ctx.rhs_buf, 1:B)
    cp_buf = view(ctx.c_prime, 1:B)
    dp_buf = view(ctx.d_prime, 1:B)

    for c in 1:C
        lam0 = max(bg_rates[c], 1e-12)
        for iter in 1:max_iter
            @inbounds for j in 1:B
                dt_eff = (j == 1 || j == B) ? (0.5 * dt) : dt
                x_val = ctx.delta_x[c, j]
                d_j = lam0 * dt_eff * exp(x_val)

                j_diag = (j == 1) ? (inv_v_inf + phi * phi * inv_v) : ((j == B) ? inv_v : (1.0 + phi * phi) * inv_v)
                d_buf[j] = j_diag + d_j
                if j < B
                    o_buf[j] = off_val
                end
                r_buf[j] = Float64(cell_counts[c, j]) - d_j + d_j * x_val
            end

            solve_sym_tridiagonal!(
                view(ctx.delta_x, c, :),
                d_buf, o_buf, r_buf, cp_buf, dp_buf
            )
        end

        # Zero-mean gauge condition: (1/B) sum delta_x == 0
        mean_dx = mean(view(ctx.delta_x, c, :))
        ctx.delta_x[c, :] .-= mean_dx
        clamp!(view(ctx.delta_x, c, :), -2.5, 2.5)
    end
end

"""
    update_motif_ou!(
        ctx::ContinuousOUContext,
        occurrences::Dict{Int, PureOccurrence},
        motifs::Dict{Int, PurePacketMotif},
        hyperstate::Level5Hyperstate;
        max_iter::Int=2
    )

Newton-Raphson Laplace smoothing on motif engagement y_motif[m, :] across motifs.
"""
function update_motif_ou!(
    ctx::ContinuousOUContext,
    occurrences::Dict{Int, PureOccurrence},
    motifs::Dict{Int, PurePacketMotif},
    hyperstate::Level5Hyperstate;
    max_iter::Int=2
)
    M = size(ctx.y_motif, 1)
    B = ctx.B_motif
    dt = ctx.dt_motif
    tau = ctx.tau_motif
    sig = ctx.sigma_motif

    phi = exp(-dt / tau)
    v = (0.5 * sig * sig * tau) * (1.0 - phi * phi)
    v_inf = 0.5 * sig * sig * tau
    inv_v = 1.0 / max(1e-12, v)
    inv_v_inf = 1.0 / max(1e-12, v_inf)
    off_val = -phi * inv_v

    occ_counts = zeros(Int, M, B)
    for occ in values(occurrences)
        mid = occ.motif
        (1 <= mid <= M) || continue
        j = clamp(floor(Int, occ.tau / dt) + 1, 1, B)
        occ_counts[mid, j] += 1
    end

    d_buf = view(ctx.diag_buf, 1:B)
    o_buf = view(ctx.off_buf, 1:(B-1))
    r_buf = view(ctx.rhs_buf, 1:B)
    cp_buf = view(ctx.c_prime, 1:B)
    dp_buf = view(ctx.d_prime, 1:B)

    for mid in keys(motifs)
        (1 <= mid <= M) || continue
        lam_m = (mid <= length(hyperstate.lambda_m)) ? hyperstate.lambda_m[mid] : 1.0
        lam_m <= 0.0 && continue

        for iter in 1:max_iter
            @inbounds for j in 1:B
                dt_eff = (j == 1 || j == B) ? (0.5 * dt) : dt
                y_val = ctx.y_motif[mid, j]
                d_j = lam_m * dt_eff * exp(y_val)

                j_diag = (j == 1) ? (inv_v_inf + phi * phi * inv_v) : ((j == B) ? inv_v : (1.0 + phi * phi) * inv_v)
                d_buf[j] = j_diag + d_j
                if j < B
                    o_buf[j] = off_val
                end
                r_buf[j] = Float64(occ_counts[mid, j]) - d_j + d_j * y_val
            end

            solve_sym_tridiagonal!(
                view(ctx.y_motif, mid, :),
                d_buf, o_buf, r_buf, cp_buf, dp_buf
            )
        end

        # Zero-mean gauge condition: (1/B) sum y_motif == 0
        mean_y = mean(view(ctx.y_motif, mid, :))
        ctx.y_motif[mid, :] .-= mean_y
        clamp!(view(ctx.y_motif, mid, :), -2.5, 2.5)
    end
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
