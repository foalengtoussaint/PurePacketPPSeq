# analyze_regional_directionality.jl
# Computes MOp vs CP regional sequencing directionality across all 20 converged motifs

using PurePacketPPSeq
using Serialization, Statistics, Printf, SpecialFunctions

using DelimitedFiles

ck_path = joinpath(@__DIR__, "checkpoint_pure_packet_cold_start_100sweeps.jls")
raw = open(ck_path, "r") do io
    deserialize(io)
end

println("Loaded Checkpoint: $(length(raw.motifs)) Motifs, $(length(raw.occurrences)) Occurrences.")

# Load exact ground-truth unit regional mapping from units.csv:
# MOp: channels 1:98 and 302:411 (208 units total)
# CP:  channels 99:301 (203 units total)
units_path = "/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/results/top4_task_20261008/data/lgcy0288__20230519_114115__ephys/units.csv"
units_raw, _ = readdlm(units_path, ',', header=true)
unit_regions = String.(units_raw[:, 3])

is_mop(c) = (1 <= c <= 411) && occursin("MOp", unit_regions[c])
is_cp(c)  = (1 <= c <= 411) && occursin("CP", unit_regions[c])

println("="^110)
@printf("%-6s | %-5s | %-12s | %-12s | %-10s | %-10s | %-12s | %-10s | %-8s\n",
        "Motif", "Occs", "MOp Pkts/Chs", "CP Pkts/Chs", "Med Mu MOp", "Med Mu CP", "Delta(MOp->CP)", "Mann-Whit p", "Dir")
println("-"^110)

all_deltas = Float64[]
mop_lead_count = 0
cp_lead_count = 0
balanced_count = 0

for (mid, pm) in sort(collect(raw.motifs), by=x->x[1])
    n_occs = count(occ -> occ.motif == mid, values(raw.occurrences))
    
    mop_pkts = [p for p in pm.packets if is_mop(p.channel)]
    cp_pkts  = [p for p in pm.packets if is_cp(p.channel)]
    
    n_mop_p = length(mop_pkts)
    n_cp_p  = length(cp_pkts)
    
    n_mop_ch = length(unique([p.channel for p in mop_pkts]))
    n_cp_ch  = length(unique([p.channel for p in cp_pkts]))
    
    mop_str = @sprintf("%3d / %3d", n_mop_p, n_mop_ch)
    cp_str  = @sprintf("%3d / %3d", n_cp_p, n_cp_ch)
    
    if n_mop_p == 0 || n_cp_p == 0
        @printf("M%-4d | %5d | %-12s | %-12s | %-10s | %-10s | %-12s | %-10s | %-8s\n",
                mid, n_occs, mop_str, cp_str, "N/A", "N/A", "Pure Regional", "N/A", n_mop_p > 0 ? "MOp-only" : "CP-only")
        continue
    end
    
    mu_mop = [p.mu for p in mop_pkts]
    mu_cp  = [p.mu for p in cp_pkts]
    
    med_mop = median(mu_mop)
    med_cp  = median(mu_cp)
    delta_t = med_cp - med_mop # Positive means MOp fires before CP
    push!(all_deltas, delta_t)
    
    # Simple Mann-Whitney U / rank-sum test between mu distributions
    # Compute rank-sum W
    combined = [(mu, :mop) for mu in mu_mop]
    append!(combined, [(mu, :cp) for mu in mu_cp])
    sort!(combined, by=x->x[1])
    ranks_mop = [i for (i, item) in enumerate(combined) if item[2] == :mop]
    R_mop = sum(ranks_mop)
    U_mop = R_mop - n_mop_p * (n_mop_p + 1) / 2
    U_cp  = n_mop_p * n_cp_p - U_mop
    U_min = min(U_mop, U_cp)
    
    # Normal approximation for large samples
    mu_U = n_mop_p * n_cp_p / 2
    sigma_U = sqrt(n_mop_p * n_cp_p * (n_mop_p + n_cp_p + 1) / 12)
    z = (U_mop - mu_U) / (sigma_U > 0 ? sigma_U : 1.0)
    # p-value approx using error function
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
    
    dir = if delta_t > 0.010 # > 10ms MOp lead
        global mop_lead_count += 1
        "MOp->CP"
    elseif delta_t < -0.010 # > 10ms CP lead
        global cp_lead_count += 1
        "CP->MOp"
    else
        global balanced_count += 1
        "Synchr"
    end
    
    @printf("M%-4d | %5d | %-12s | %-12s | %7.1f ms | %7.1f ms | %+8.1f ms  | %10.4f  | %-8s\n",
            mid, n_occs, mop_str, cp_str, med_mop * 1000, med_cp * 1000, delta_t * 1000, p_val, dir)
end
println("="^110)

println("\nSummary of Cross-Regional Motifs (with both MOp and CP participation):")
@printf("  Total Cross-Regional Motifs: %d / %d\n", length(all_deltas), length(raw.motifs))
@printf("  MOp Leads CP (> +10ms):     %d\n", mop_lead_count)
@printf("  CP Leads MOp (< -10ms):     %d\n", cp_lead_count)
@printf("  Synchronous (|Delta| <= 10ms):%d\n", balanced_count)
if !isempty(all_deltas)
    @printf("  Mean Regional Delay Delta:   %+.2f ms\n", mean(all_deltas) * 1000)
    @printf("  Median Regional Delay Delta: %+.2f ms\n", median(all_deltas) * 1000)
end
