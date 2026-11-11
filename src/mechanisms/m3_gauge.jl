# mechanisms/m3_gauge.jl - Level m=3 Continuous Translational Gauge Symmetry
#
# Generative Role:
# - Enforces exact translational gauge symmetry:
#     sum_{p=1}^{P_m} kappa_{m,p} W_{r,p} == 0
#   where kappa_{m,p} = 1.0 / var_p is the precision of emission packet p.
# - Decouples global carrier onset tau_e from continuous manifold deformation W u_e,
#   preventing non-identifiable translational drift across MCMC sweeps.

"""
    project_translational_gauge!(m::PurePacketMotif, active_mask::BitVector)

Strictly projects manifold deformation loadings W onto the orthogonal subspace:
    sum_p kappa_{m,p} W_{r,p} == 0
for all active ARD dimensions r.
Zeroes out dormant dimensions.
"""
function project_translational_gauge!(m::PurePacketMotif, active_mask::BitVector)
    P = length(m.packets)
    P == 0 && return
    R_max = length(active_mask)
    tot_prec = sum(1.0 / m.packets[p].var for p in 1:P)
    tot_prec <= 0.0 && return

    for r in 1:R_max
        if active_mask[r]
            w_mean = sum((1.0 / m.packets[p].var) * m.W[r, p] for p in 1:P) / tot_prec
            @views m.W[r, :] .-= w_mean
        else
            @views m.W[r, :] .= 0.0
        end
    end
end

"""
    compute_gauge_error(m::PurePacketMotif, active_mask::BitVector) -> Float64

Evaluates maximum normalized gauge error across active ARD axes:
    max_r |sum_p kappa_p W_{r,p}| / sum_p kappa_p
Verifies exact machine-zero gauge invariance.
"""
function compute_gauge_error(m::PurePacketMotif, active_mask::BitVector)::Float64
    P = length(m.packets)
    P == 0 && return 0.0
    tot_prec = sum(1.0 / pkt.var for pkt in m.packets)
    tot_prec <= 0.0 && return 0.0

    max_err = 0.0
    for r in 1:length(active_mask)
        if active_mask[r]
            err = abs(sum((1.0 / m.packets[p].var) * m.W[r, p] for p in 1:P) / tot_prec)
            max_err = max(max_err, err)
        end
    end
    return max_err
end
