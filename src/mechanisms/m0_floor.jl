# mechanisms/m0_floor.jl - Level m=0 Stationary Baseline Floor & Poisson Compensator
#
# Generative Role:
# - Anchors absolute spike rates per physical channel.
# - Evaluates continuous Poisson compensator volume int lambda_0,c exp(x_r + delta x_c) dt.
# - Provides likelihood floor for dual-index spike assignment vs foreground sequences.

"""
    resample_background_pure_packet!(
        bg_rates, pi_bg, gamma_buf, counts_bg, assign, neuron,
        channel_group, groups, bg_a, bg_b, beta_bg, T_total, exposure;
        rng=default_rng()
    )

Level m=0 Zero-allocation discrete baseline rate draws across channel groups:
    lambda_{0,k} = lambda_g * pi_{bg,k}
"""
function resample_background_pure_packet!(
    bg_rates::Vector{Float64},
    pi_bg::Vector{Float64},
    gamma_buf::Vector{Float64},
    counts_bg::Vector{Int},
    assign::AbstractVector{SpikeAssignment},
    neuron::AbstractVector{Int},
    channel_group::AbstractVector{Int},
    groups::AbstractVector,
    bg_a::AbstractVector{Float64},
    bg_b::AbstractVector{Float64},
    beta_bg::AbstractVector{Float64},
    T_total::Float64,
    exposure::AbstractVector{Float64};
    rng::AbstractRNG=Random.default_rng()
)
    fill!(counts_bg, 0)
    @inbounds for i in eachindex(assign)
        if assign[i].event_id == 0
            counts_bg[neuron[i]] += 1
        end
    end

    for (g, grp) in enumerate(groups)
        exp_g = exposure[g]
        exp_g <= 0.0 && continue
        b_bg = beta_bg[g]
        tot_grp_counts = 0
        sum_gamma = 0.0

        @inbounds for k in grp
            c_k = counts_bg[k]
            tot_grp_counts += c_k
            g_k = rand(rng, Gamma(b_bg + c_k, 1.0))
            gamma_buf[k] = g_k
            sum_gamma += g_k
        end

        inv_sum_gamma = sum_gamma > 0.0 ? (1.0 / sum_gamma) : 0.0
        @inbounds for k in grp
            pi_bg[k] = gamma_buf[k] * inv_sum_gamma
        end

        shape_g = bg_a[g] + tot_grp_counts
        rate_g = bg_b[g] + T_total * exp_g
        lambda_g = rand(rng, Gamma(shape_g, 1.0 / max(rate_g, 1e-12)))

        @inbounds for k in grp
            bg_rates[k] = lambda_g * pi_bg[k]
        end
    end
end

"""
    update_continuous_background_rates!(
        bg_rates, ctx, counts_bg, assign, neuron, T_total;
        alpha_bg=1.0, beta_bg=1.0, rng=default_rng()
    )

Level m=0 & m=5: Exact conjugate Gamma rate draws with continuous OU exposure integrals:
    lambda_{0,c} ~ Gamma(alpha_bg + N_bg(c), 1 / (beta_bg + T_{eff,c}))
where T_{eff,c} = int_0^T exp(x_r(t) + delta x_c(t)) dt.
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

    # 1. Regional segment integrals
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

    # 3. Draw conjugate lambda_{0,c}
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
    evaluate_integrated_background(ctx, bg_rate_c, c, t1, t2) -> Float64

Evaluates int_{t1}^{t2} lambda_c^{(0)}(s) ds for Time-Rescaling theorem validation.
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


