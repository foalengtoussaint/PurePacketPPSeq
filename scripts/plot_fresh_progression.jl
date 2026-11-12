# plot_fresh_progression.jl
# Rich publication-quality progression plot comparing the Fresh RJMCMC Run against the old Runaway Run

using PyPlot
using Printf

fresh_log = "/home/foaleng/Documents/PurePacketPPSeq/scripts/run_awake_scratch_fresh.log"
old_log1 = "/home/foaleng/.gemini/antigravity/brain/326fd4bb-683a-4e18-90e7-47edba633672/.system_generated/tasks/task-26658.log"
old_log2 = "/home/foaleng/.gemini/antigravity/brain/326fd4bb-683a-4e18-90e7-47edba633672/.system_generated/tasks/task-26876.log"

function parse_log_file(path; is_fresh=false)
    sweeps = Int[]
    events = Int[]
    prune_occs = Int[]
    birth_occs = Int[]
    spikes = Int[]
    packets = Int[]
    vigor = Float64[]
    motifs = Int[]

    active_mids = Set{Int}([1, 2, 3, 4, 6, 7, 8, 10, 15])
    row_pattern = r"^(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+([0-9\.]+)"

    isfile(path) || return nothing
    open(path, "r") do f
        for line in eachline(f)
            mb = match(r"Motif Birthed\] Motif ID:\s*(\d+)", line)
            if mb !== nothing
                push!(active_mids, parse(Int, mb.captures[1]))
            end
            
            mp = match(r"Motif Pruned\] Motif ID:\s*(\d+)", line)
            if mp !== nothing
                delete!(active_mids, parse(Int, mp.captures[1]))
            end

            m = match(row_pattern, strip(line))
            if m !== nothing
                sw = parse(Int, m.captures[1])
                if !is_fresh
                    if sw == 3
                        delete!(active_mids, 15)
                    elseif sw == 18
                        delete!(active_mids, 32)
                    end
                else
                    if sw == 2
                        delete!(active_mids, 15)
                    end
                end

                push!(sweeps, sw)
                push!(events, parse(Int, m.captures[2]))
                push!(prune_occs, parse(Int, m.captures[3]))
                push!(birth_occs, parse(Int, m.captures[4]))
                push!(spikes, parse(Int, m.captures[5]))
                push!(packets, parse(Int, m.captures[6]))
                push!(vigor, parse(Float64, m.captures[7]))
                push!(motifs, length(active_mids))
            end
        end
    end
    return (
        sweeps=sweeps, events=events, prune_occs=prune_occs, birth_occs=birth_occs,
        spikes=spikes, packets=packets, vigor=vigor, motifs=motifs
    )
end

function parse_old_run()
    d1 = parse_log_file(old_log1; is_fresh=false)
    d2 = parse_log_file(old_log2; is_fresh=false)
    
    sweeps = vcat(d1.sweeps, d2.sweeps)
    events = vcat(d1.events, d2.events)
    spikes = vcat(d1.spikes, d2.spikes)
    packets = vcat(d1.packets, d2.packets)
    motifs = vcat(d1.motifs, d2.motifs)
    
    return (sweeps=sweeps, events=events, spikes=spikes, packets=packets, motifs=motifs)
end

fresh = parse_log_file(fresh_log; is_fresh=true)
old = parse_old_run()

println("Fresh Run: $(length(fresh.sweeps)) sweeps. Latest: Sweep $(fresh.sweeps[end]) | M=$(fresh.motifs[end]) | P=$(fresh.packets[end]) | E=$(fresh.events[end]) | Spikes=$(fresh.spikes[end])")

PyPlot.rc("font", family="sans-serif", size=10)
PyPlot.rc("axes", labelsize=11, titlesize=12)
PyPlot.rc("xtick", labelsize=9)
PyPlot.rc("ytick", labelsize=9)

fig, axs = PyPlot.subplots(2, 2, figsize=(14, 10), dpi=300)

# Panel A: Motifs (M)
ax_m = axs[1, 1]
ax_m.plot(old.sweeps, old.motifs, color="#d62728", linestyle="--", alpha=0.6, linewidth=1.8, label="Old Runaway (No Parameter Occam Tax)")
ax_m.plot(fresh.sweeps, fresh.motifs, color="#1f77b4", linewidth=2.8, label="Fresh RJMCMC (Exact Occam Tax + Singleton Death)")
ax_m.scatter([fresh.sweeps[end]], [fresh.motifs[end]], color="#1f77b4", s=80, zorder=5, label=@sprintf("Fresh Sweep %d: M = %d (Stable Plateau)", fresh.sweeps[end], fresh.motifs[end]))
ax_m.set_xlabel("MCMC Sweep")
ax_m.set_ylabel("Active Motif Count (M)")
ax_m.set_title("A. Nonparametric Motif Dictionary Dynamics (M)", fontweight="bold")
ax_m.set_ylim(0, max(maximum(old.motifs) + 5, 20))
ax_m.grid(true, linestyle="--", alpha=0.5)
ax_m.legend(loc="upper left", framealpha=0.9)

# Panel B: Total Packets (P)
ax_p = axs[1, 2]
ax_p.plot(old.sweeps, old.packets, color="#d62728", linestyle="--", alpha=0.6, linewidth=1.8, label="Old Runaway (+250 pkts/sweep)")
ax_p.plot(fresh.sweeps, fresh.packets, color="#9467bd", linewidth=2.8, label="Fresh RJMCMC (Concave Saturation)")
ax_p.scatter([fresh.sweeps[end]], [fresh.packets[end]], color="#9467bd", s=80, zorder=5, label=@sprintf("Fresh Sweep %d: P = %d (~%.0f pkts/motif)", fresh.sweeps[end], fresh.packets[end], fresh.packets[end] / fresh.motifs[end]))
ax_p.set_xlabel("MCMC Sweep")
ax_p.set_ylabel("Total Spatiotemporal Packets (P)")
ax_p.set_title("B. Spatiotemporal Emission Packets Expansion (P)", fontweight="bold")
ax_p.grid(true, linestyle="--", alpha=0.5)
ax_p.legend(loc="upper left", framealpha=0.9)

# Panel C: Foreground Assigned Spikes
ax_s = axs[2, 1]
spk_k = fresh.spikes ./ 1000.0
pct_mass = (fresh.spikes ./ 2476213.0) .* 100.0
line_s = ax_s.plot(fresh.sweeps, spk_k, color="#2ca02c", linewidth=2.8, label="Fresh Assigned Spikes")
ax_s.scatter([fresh.sweeps[end]], [spk_k[end]], color="#2ca02c", s=80, zorder=5)
ax_s.set_xlabel("MCMC Sweep")
ax_s.set_ylabel("Foreground Assigned Spikes (×10³)")
ax_s.set_title("C. Continuous Foreground Spike Recruitment & Saturation", fontweight="bold")
ax_s.grid(true, linestyle="--", alpha=0.5)

ax_pct = ax_s.twinx()
ax_pct.plot(fresh.sweeps, pct_mass, color="#17becf", linestyle=":", linewidth=2.0, label="Foreground Mass (%)")
ax_pct.set_ylabel("Foreground Mass (%) of 2.47M Total Spikes", color="#17becf")
ax_pct.tick_params(axis="y", labelcolor="#17becf")
ax_pct.set_ylim(0, 35)

lines_c = [line_s[1], ax_pct.lines[1]]
labels_c = [@sprintf("Spikes: %d", fresh.spikes[end]), @sprintf("Foreground Mass: %.1f%%", pct_mass[end])]
ax_s.legend(lines_c, labels_c, loc="lower right", framealpha=0.9)

# Panel D: Total Occurrence Number (E)
ax_e = axs[2, 2]
line_e = ax_e.plot(fresh.sweeps, fresh.events, color="#e377c2", linewidth=2.8, label="Active Occurrences (E)")
ax_e.scatter([fresh.sweeps[end]], [fresh.events[end]], color="#e377c2", s=80, zorder=5)
ax_e.set_xlabel("MCMC Sweep")
ax_e.set_ylabel("Active Replay Occurrences (E)", color="#c51b8a")
ax_e.tick_params(axis="y", labelcolor="#c51b8a")
ax_e.set_title("D. Active Occurrence Population Dynamics (E)", fontweight="bold")
ax_e.grid(true, linestyle="--", alpha=0.5)

ax_flux = ax_e.twinx()
net_flux = fresh.birth_occs .- fresh.prune_occs
line_b = ax_flux.plot(fresh.sweeps, fresh.birth_occs, color="#2ca02c", linestyle=":", alpha=0.7, linewidth=1.4, label="Proposed Births")
line_pr = ax_flux.plot(fresh.sweeps, fresh.prune_occs, color="#d62728", linestyle=":", alpha=0.7, linewidth=1.4, label="Pruned Occurrences")
ax_flux.set_ylabel("Per-Sweep Occurrence Proposals (Birth / Prune)", color="#555555")
ax_flux.tick_params(axis="y", labelcolor="#555555")

lines_d = [line_e[1], line_b[1], line_pr[1]]
event_rate = fresh.events[end] / 2114.288
labels_d = [
    @sprintf("Active Occurrences: %d (%.2f Hz)", fresh.events[end], event_rate),
    @sprintf("Births: %d/sw", fresh.birth_occs[end]),
    @sprintf("Pruned: %d/sw (Net: %+d)", fresh.prune_occs[end], net_flux[end])
]
ax_e.legend(lines_d, labels_d, loc="center right", framealpha=0.9)

PyPlot.tight_layout()

out_path = "/home/foaleng/.gemini/antigravity/brain/326fd4bb-683a-4e18-90e7-47edba633672/fig_progression_fresh_vs_runaway_v3.png"
PyPlot.savefig(out_path, dpi=300, bbox_inches="tight")
PyPlot.close()

println("Successfully saved progression figure to: $out_path")
