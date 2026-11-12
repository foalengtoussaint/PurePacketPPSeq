# plot_convergence_progression.jl
# Rich publication-quality progression plot of Motifs (M), Packets (P), Foreground Spikes, and Occurrences (E)

using PurePacketPPSeq
using PyPlot
using Printf

log1 = "/home/foaleng/.gemini/antigravity/brain/326fd4bb-683a-4e18-90e7-47edba633672/.system_generated/tasks/task-26658.log"
log2 = "/home/foaleng/.gemini/antigravity/brain/326fd4bb-683a-4e18-90e7-47edba633672/.system_generated/tasks/task-26876.log"

function parse_logs(log1, log2)
    sweeps = Int[]
    events = Int[]
    prune_occs = Int[]
    birth_occs = Int[]
    spikes = Int[]
    packets = Int[]
    vigor = Float64[]
    motifs = Int[]
    bg_rates_mean = Float64[]

    active_mids = Set{Int}([1, 2, 3, 4, 6, 7, 8, 10, 15]) # Initial 9 seed motifs
    row_pattern = r"^(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+(\d+)\s+\|\s+([0-9\.]+)"
    bg_pattern = r"mean\(bg_rates\)=([0-9\.]+)\s*Hz"

    curr_bg = 2.4013

    for (p_idx, path) in enumerate([log1, log2])
        isfile(path) || continue
        open(path, "r") do f
            for line in eachline(f)
                # Background rate tracking
                mbg = match(bg_pattern, line)
                if mbg !== nothing
                    curr_bg = parse(Float64, mbg.captures[1])
                end

                # Motif birth tracking
                mb = match(r"Motif Birthed\] Motif ID:\s*(\d+)", line)
                if mb !== nothing
                    push!(active_mids, parse(Int, mb.captures[1]))
                end
                
                # Motif death tracking
                mp = match(r"Motif Pruned\] Motif ID:\s*(\d+)", line)
                if mp !== nothing
                    delete!(active_mids, parse(Int, mp.captures[1]))
                end

                # Row data
                m = match(row_pattern, strip(line))
                if m !== nothing
                    sw = parse(Int, m.captures[1])
                    # In Sweeps 1-50, Motif 15 died at sweep 3, Motif 32 died at sweep 18
                    if p_idx == 1
                        if sw == 3
                            delete!(active_mids, 15)
                        elseif sw == 18
                            delete!(active_mids, 32)
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
                    push!(bg_rates_mean, curr_bg)
                end
            end
        end
    end
    return (
        sweeps=sweeps, events=events, prune_occs=prune_occs, birth_occs=birth_occs,
        spikes=spikes, packets=packets, vigor=vigor, motifs=motifs, bg_rates_mean=bg_rates_mean
    )
end

parsed = parse_logs(log1, log2)
sweeps = parsed.sweeps
events = parsed.events
prune_occs = parsed.prune_occs
birth_occs = parsed.birth_occs
spikes = parsed.spikes
packets = parsed.packets
vigor = parsed.vigor
motifs = parsed.motifs
bg_rates_mean = parsed.bg_rates_mean

N_sweeps = length(sweeps)
println("Extracted $(N_sweeps) sweeps. Latest: Sweep $(sweeps[end]), Events=$(events[end]), Spikes=$(spikes[end]), Packets=$(packets[end]), Motifs=$(motifs[end])")

# Setup Matplotlib styling
PyPlot.rc("font", family="sans-serif", size=10)
PyPlot.rc("axes", labelsize=11, titlesize=12)
PyPlot.rc("xtick", labelsize=9)
PyPlot.rc("ytick", labelsize=9)

fig, axs = PyPlot.subplots(2, 2, figsize=(14, 10), dpi=300)

# Panel A: Motifs (M)
ax_m = axs[1, 1]
ax_m.plot(sweeps, motifs, color="#1f77b4", linewidth=2.5, label="Active Motifs (M)")
if N_sweeps >= 50
    ax_m.scatter([50], [motifs[50]], color="#d62728", s=60, zorder=5, label=@sprintf("Sweep 50: M = %d", motifs[50]))
end
ax_m.scatter([sweeps[end]], [motifs[end]], color="#2ca02c", s=70, zorder=5, label=@sprintf("Sweep %d: M = %d", sweeps[end], motifs[end]))
ax_m.set_xlabel("MCMC Sweep")
ax_m.set_ylabel("Active Motif Count (M)")
ax_m.set_title("A. Nonparametric Motif Dictionary Dynamics (M)", fontweight="bold")
ax_m.grid(true, linestyle="--", alpha=0.5)
ax_m.legend(loc="lower right", framealpha=0.9)

# Panel B: Total Packets (P) & Mean Packets per Motif
ax_p = axs[1, 2]
ax_p.plot(sweeps, packets, color="#9467bd", linewidth=2.5, label="Total Emission Packets (P)")
if N_sweeps >= 50
    ax_p.scatter([50], [packets[50]], color="#d62728", s=60, zorder=5, label=@sprintf("Sweep 50: P = %d", packets[50]))
end
ax_p.scatter([sweeps[end]], [packets[end]], color="#2ca02c", s=70, zorder=5, label=@sprintf("Sweep %d: P = %d", sweeps[end], packets[end]))
ax_p.set_xlabel("MCMC Sweep")
ax_p.set_ylabel("Total Emission Packets (P)")
ax_p.set_title("B. Spatiotemporal Emission Packets Expansion (P)", fontweight="bold")
ax_p.grid(true, linestyle="--", alpha=0.5)

# Secondary axis for Packets per Motif
ax_ppm = ax_p.twinx()
ppm = packets ./ motifs
ax_ppm.plot(sweeps, ppm, color="#8c564b", linestyle="--", linewidth=1.8, label="Mean Packets / Motif")
ax_ppm.set_ylabel("Mean Packets per Motif (P / M)", color="#8c564b")
ax_ppm.tick_params(axis="y", labelcolor="#8c564b")
lines_p = [ax_p.lines[1], ax_ppm.lines[1]]
labels_p = [@sprintf("Total Packets: %d", packets[end]), @sprintf("Mean Density: %.1f pkts/motif", ppm[end])]
ax_p.legend(lines_p, labels_p, loc="lower right", framealpha=0.9)

# Panel C: Foreground Assigned Spikes & Foreground Mass %
ax_s = axs[2, 1]
spk_k = spikes ./ 1000.0
pct_mass = (spikes ./ 2476213.0) .* 100.0
line_s = ax_s.plot(sweeps, spk_k, color="#2ca02c", linewidth=2.5, label="Assigned Spikes")
if N_sweeps >= 50
    ax_s.scatter([50], [spk_k[50]], color="#d62728", s=60, zorder=5)
end
ax_s.scatter([sweeps[end]], [spk_k[end]], color="#2ca02c", s=70, zorder=5)
ax_s.set_xlabel("MCMC Sweep")
ax_s.set_ylabel("Foreground Assigned Spikes (×10³)")
ax_s.set_title("C. Continuous Foreground Spike Recruitment", fontweight="bold")
ax_s.grid(true, linestyle="--", alpha=0.5)

# Secondary axis for percentage
ax_pct = ax_s.twinx()
ax_pct.plot(sweeps, pct_mass, color="#17becf", linestyle=":", linewidth=2.0, label="Foreground Mass (%)")
ax_pct.set_ylabel("Foreground Mass (%) of 2.47M Total Spikes", color="#17becf")
ax_pct.tick_params(axis="y", labelcolor="#17becf")
ax_pct.set_ylim(0, 35)

lines_c = [line_s[1], ax_pct.lines[1]]
labels_c = [@sprintf("Spikes: %d", spikes[end]), @sprintf("Foreground Mass: %.1f%%", pct_mass[end])]
ax_s.legend(lines_c, labels_c, loc="lower right", framealpha=0.9)

# Panel D: Total Occurrence Number (E) & Birth/Prune Equilibrium
ax_e = axs[2, 2]
line_e = ax_e.plot(sweeps, events, color="#e377c2", linewidth=2.5, label="Active Occurrences (E)")
if N_sweeps >= 50
    ax_e.scatter([50], [events[50]], color="#d62728", s=60, zorder=5, label=@sprintf("Sweep 50: E = %d", events[50]))
end
ax_e.scatter([sweeps[end]], [events[end]], color="#2ca02c", s=70, zorder=5, label=@sprintf("Sweep %d: E = %d", sweeps[end], events[end]))
ax_e.set_xlabel("MCMC Sweep")
ax_e.set_ylabel("Active Replay Occurrences (E)", color="#c51b8a")
ax_e.tick_params(axis="y", labelcolor="#c51b8a")
ax_e.set_title("D. Active Occurrence Population Dynamics (E)", fontweight="bold")
ax_e.grid(true, linestyle="--", alpha=0.5)

# Secondary axis for Occurrence Turnover Flux (Birth vs Prune proposals)
ax_flux = ax_e.twinx()
net_flux = birth_occs .- prune_occs
line_b = ax_flux.plot(sweeps, birth_occs, color="#2ca02c", linestyle=":", alpha=0.7, linewidth=1.4, label="Proposed Births")
line_pr = ax_flux.plot(sweeps, prune_occs, color="#d62728", linestyle=":", alpha=0.7, linewidth=1.4, label="Pruned Occurrences")
ax_flux.set_ylabel("Per-Sweep Occurrence Proposals (Birth / Prune)", color="#555555")
ax_flux.tick_params(axis="y", labelcolor="#555555")

lines_d = [line_e[1], line_b[1], line_pr[1]]
event_rate = events[end] / 2114.288
labels_d = [
    @sprintf("Active Occurrences: %d (%.2f Hz)", events[end], event_rate),
    @sprintf("Births: %d/sw", birth_occs[end]),
    @sprintf("Pruned: %d/sw (Net: %+d)", prune_occs[end], net_flux[end])
]
ax_e.legend(lines_d, labels_d, loc="lower right", framealpha=0.9)

PyPlot.tight_layout()

out_path = "/home/foaleng/.gemini/antigravity/brain/326fd4bb-683a-4e18-90e7-47edba633672/fig_progression_with_occurrences_v2.png"
PyPlot.savefig(out_path, dpi=300, bbox_inches="tight")
PyPlot.close()

println("Updated figure successfully saved to: $out_path")
