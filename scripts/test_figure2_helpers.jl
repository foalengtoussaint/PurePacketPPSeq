using PurePacketPPSeq, Serialization, Printf, Statistics, NPZ, DelimitedFiles, LinearAlgebra

println("Testing Figure 2 helpers...")

ck = open(deserialize, "scripts/checkpoint_behavioral_correspondence.jls")
data_dir = "/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/results/top4_task_20261008/data/lgcy0288__20230519_114115__ephys"
raw_npz = npzread(joinpath(data_dir, "joint.npz"))
ts = Float64.(raw_npz["timestamp"])
ch = Int.(raw_npz["channel"])

units_path = joinpath(data_dir, "units.csv")
units_raw, _ = readdlm(units_path, ',', header=true)
regions = String.(units_raw[:, 3])
is_mop(c) = (1 <= c <= 411) && occursin("MOp", regions[c])
is_cp(c)  = (1 <= c <= 411) && occursin("CP", regions[c])

m1 = ck.motifs[1]
bg = ck.bg_rates
occs = ck.occurrences

function get_spine_data(occ_id)
    occ = occs[occ_id]
    idx_start = searchsortedfirst(ts, occ.tau - 0.2)
    idx_end = searchsortedlast(ts, occ.tau + 2.5)
    t_sub = ts[idx_start:idx_end]
    ch_sub = ch[idx_start:idx_end]
    
    chan_map = Dict{Int, Vector{Int}}()
    for (loc_i, c) in enumerate(ch_sub)
        push!(get!(chan_map, c, Int[]), loc_i)
    end
    
    res = []
    for (p_idx, pkt) in enumerate(m1.packets)
        c = pkt.channel
        haskey(chan_map, c) || continue
        spk_locs = chan_map[c]
        
        pred_s = 0.0
        for r in 1:min(6, length(occ.u), size(m1.W, 1))
            pred_s += m1.W[r, p_idx] * occ.u[r]
        end
        t_pred = occ.tau + pkt.mu + pred_s
        sig2 = pkt.var
        sig = sqrt(sig2)
        norm_const = pkt.mass / (sqrt(2π) * sig)
        
        sum_r = 0.0
        sum_rt = 0.0
        for loc_i in spk_locs
            t_i = t_sub[loc_i]
            dt = t_i - t_pred
            lam_p = norm_const * exp(-0.5 * dt * dt / sig2)
            lam_tot = bg[c]
            for (p2_idx, pkt2) in enumerate(m1.packets)
                if pkt2.channel == c
                    p2_s = 0.0
                    for r in 1:min(6, length(occ.u), size(m1.W, 1))
                        p2_s += m1.W[r, p2_idx] * occ.u[r]
                    end
                    dt2 = t_i - (occ.tau + pkt2.mu + p2_s)
                    lam_tot += (pkt2.mass / (sqrt(2π*pkt2.var))) * exp(-0.5 * dt2 * dt2 / pkt2.var)
                end
            end
            r_i = lam_p / max(lam_tot, 1e-12)
            sum_r += r_i
            sum_rt += r_i * t_i
        end
        
        if sum_r > 0.5
            hat_t = sum_rt / sum_r
            residual_ms = ((hat_t - occ.tau) - pkt.mu) * 1000.0
            pred_w2_ms = m1.W[2, p_idx] * occ.u[2] * 1000.0
            pred_w4_ms = m1.W[4, p_idx] * occ.u[4] * 1000.0
            push!(res, (p_idx=p_idx, ch=c, mu=pkt.mu, is_beh=(c >= 412), is_mop=is_mop(c), is_cp=is_cp(c),
                        res_ms=residual_ms, pred_w2_ms=pred_w2_ms, pred_w4_ms=pred_w4_ms, sig_ms=sig*1000.0, mass=sum_r))
        end
    end
    return res
end

res_slow = get_spine_data(404)
res_fast = get_spine_data(142)

println("Slow Trial 404: $(length(res_slow)) packets (Beh: $(count(x->x.is_beh, res_slow)))")
println("Fast Trial 142: $(length(res_fast)) packets (Beh: $(count(x->x.is_beh, res_fast)))")

corr_slow = cor([x.res_ms for x in res_slow], [x.pred_w2_ms for x in res_slow])
corr_fast = cor([x.res_ms for x in res_fast], [x.pred_w2_ms for x in res_fast])
println(@sprintf("Corr(residual, pred_W2*u2): Slow = %+.4f, Fast = %+.4f", corr_slow, corr_fast))

println("\nComputing SE for behavioral packets in Motif 1:")
occs_m1 = [occ for occ in values(ck.occurrences) if occ.motif == 1]
alpha_beh = ck.alpha_beh
active_mask = ck.ard_state.active_mask
act_dims = findall(active_mask)
K = length(act_dims)
beh_pkts = [(i, p) for (i, p) in enumerate(m1.packets) if p.channel >= 412]

for (p_idx, pkt) in beh_pkts
    sig2 = max(pkt.var, 1e-8)
    inv_Sigma_0 = diagm([alpha_beh[r] for r in act_dims])
    A_p = zeros(Float64, K, K)
    b_p = zeros(Float64, K)
    for occ in occs_m1
        c = occ.counts[p_idx]
        c == 0 && continue
        mean_t = occ.sum_t[p_idx] / c
        y = mean_t - (occ.tau + pkt.mu)
        kap = c / sig2
        u_k = [occ.u[r] for r in act_dims]
        for i in 1:K
            b_p[i] += kap * u_k[i] * y
            for j in 1:K
                A_p[i, j] += kap * u_k[i] * u_k[j]
            end
        end
    end
    inv_Sigma_post = inv_Sigma_0 + A_p
    E = eigen(Hermitian(inv_Sigma_post))
    evals = max.(E.values, 1e-6)
    Sigma_post = E.vectors * diagm(1.0 ./ evals) * adjoint(E.vectors)
    se_w2 = sqrt(Sigma_post[2, 2])
    se_w4 = sqrt(Sigma_post[4, 4])
    println(@sprintf("  Ch %3d (%-14s): W2 = %+6.2f +/- %4.2f ms | W4 = %+6.2f +/- %4.2f ms | mass = %.2f",
            pkt.channel, ck.beh_names[pkt.channel], m1.W[2, p_idx]*1000, se_w2*1000, m1.W[4, p_idx]*1000, se_w4*1000, pkt.mass))
end
