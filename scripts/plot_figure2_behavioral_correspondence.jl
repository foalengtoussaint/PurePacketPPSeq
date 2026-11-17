# plot_figure2_behavioral_correspondence.jl
# Publication-quality Figure 2:
# Behavioral Assembly, Learned Port Offsets, and Rigorous Surrogate Controls
#
# Panel A: Behavioral Arrival Waves (Anchored strictly to Motif 1, E_1 = 229)
#          P2 -> P1 -> P6 -> P3 -> P7 -> Reward (-0.2 s to +2.2 s)
#          Learned Gaussian arrival waves a_p * N(t; mu_p, sigma_p^2),
#          shaded in-port dwell bridges (110-180 ms), and empirical baseline ticks.
#
# Panel B: Out-of-Sample Validation & Surrogate Controls
#   B1: Time-Shift Null (0 ± 0 vs. 84 packets / 88.5% mass) with direct callouts
#   B2: Continuous Out-of-Sample Strip on Held-Out Second Half (25s window, same vertical port order)
#   B3: Split-Half CV Generalization (-2,020 nats vs. +4,556 nats / 4.0x focus) with direct callouts

using PurePacketPPSeq
using Serialization, Statistics, Printf, NPZ, DelimitedFiles, LinearAlgebra
using PyPlot

println("="^90)
println("GENERATING PUBLICATION FIGURE 2: BEHAVIORAL ASSEMBLY & SURROGATE CONTROLS")
println("="^90)

# Configure typography
plt.style.use("seaborn-v0_8-whitegrid" in plt.style.available ? "seaborn-v0_8-whitegrid" : "default")
PyPlot.rc("font", family="sans-serif", size=10.0)
PyPlot.rc("axes", titlesize=11.0, labelsize=10.0, linewidth=0.9, edgecolor="#1e293b")
PyPlot.rc("xtick", labelsize=8.5)
PyPlot.rc("ytick", labelsize=8.5)
PyPlot.rc("legend", fontsize=8.5, framealpha=0.92)

# 1. Load Behavioral Checkpoint and Raw Stream
ck_path = joinpath(@__DIR__, "checkpoint_behavioral_correspondence.jls")
println("Loading checkpoint: ", ck_path)
ck = open(deserialize, ck_path)

data_dir = "/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/results/top4_task_20261008/data/lgcy0288__20230519_114115__ephys"
raw_npz = npzread(joinpath(data_dir, "joint.npz"))
ts = Float64.(raw_npz["timestamp"])
ch = Int.(raw_npz["channel"])

m1 = ck.motifs[1]
occs = ck.occurrences
occs_m1 = [occ for (id, occ) in occs if occ.motif == 1]
N_M1 = length(occs_m1)
println(@sprintf("Loaded Motif 1 with %d occurrences.", N_M1))

# Extract learned behavioral packets in Motif 1
# Order of physical sequence: Port 2 -> Port 1 -> Port 6 -> Port 3 -> Port 7 -> Reward
port_configs = [
    (name = "Port 2",  c_en = 414, c_ex = 415, col = "#0284c7", fill = "#bae6fd"),
    (name = "Port 1",  c_en = 412, c_ex = 413, col = "#0d9488", fill = "#99f6e4"),
    (name = "Port 6",  c_en = 422, c_ex = 423, col = "#7c3aed", fill = "#ddd6fe"),
    (name = "Port 3",  c_en = 416, c_ex = 417, col = "#ea580c", fill = "#fed7aa"),
    (name = "Port 7 / Reward", c_en = 424, c_ex = 426, col = "#db2777", fill = "#fbcfe8")
]

# Extract dominant entry and exit packet parameters for each port in Motif 1
port_packet_params = Dict{Int, @NamedTuple{mu::Float64, var::Float64, mass::Float64}}()
for p in m1.packets
    if p.channel >= 412
        if !haskey(port_packet_params, p.channel) || p.mass > port_packet_params[p.channel].mass
            port_packet_params[p.channel] = (mu = p.mu, var = p.var, mass = p.mass)
        end
    end
end

println("Dominant learned port packet parameters in Motif 1:")
for cfg in port_configs
    p_en = port_packet_params[cfg.c_en]
    p_ex = port_packet_params[cfg.c_ex]
    dwell_ms = (p_ex.mu - p_en.mu) * 1000.0
    @printf("  %-16s: Entry mu = %6.1f ms, Exit mu = %6.1f ms (Dwell = %5.1f ms)\n",
            cfg.name, p_en.mu * 1000, p_ex.mu * 1000, dwell_ms)
end

# 2. Set Up Figure Layout
# 2-Tier Architecture: Panel A (Upper 52%), Panel B (Lower 48%)
println("Setting up publication figure layout (17.5 x 11.5 inches)...")
fig = PyPlot.figure(figsize=(17.5, 11.5), dpi=300, facecolor="#ffffff")
gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1.15, 0.95], hspace=0.38,
                                  left=0.06, right=0.97, top=0.91, bottom=0.06)

fig.suptitle("Figure 2: Behavioral Assembly, Learned Port Offsets, and Rigorous Surrogate Controls",
             fontsize=14.5, fontweight="bold", y=0.975, color="#0f172a")

# ==============================================================================
# PANEL A: BEHAVIORAL ARRIVAL WAVES (ANCHORED TO MOTIF 1 ONSET tau = 0)
# ==============================================================================
println("Rendering Panel A: Behavioral Arrival Waves & In-Port Dwell Bridges...")
gs_a = matplotlib.gridspec.GridSpecFromSubplotSpec(5, 1, subplot_spec=gs[1], hspace=0.20)

t_grid_a = range(-0.20, 2.20, length=600)
ax_a_first = nothing
ax_a_last = nothing

for (row_idx, cfg) in enumerate(port_configs)
    ax = fig.add_subplot(gs_a[row_idx])
    if row_idx == 1
        global ax_a_first = ax
        ax.text(-0.06, 1.22, "A   Behavioral Arrival Waves: Chronological Port Offsets & In-Port Dwells (Motif 1, E = 229)",
                transform=ax.transAxes, fontsize=12.0, fontweight="bold", color="#0f172a")
    end
    if row_idx == 5
        global ax_a_last = ax
    end
    
    p_en = port_packet_params[cfg.c_en]
    p_ex = port_packet_params[cfg.c_ex]
    
    sd_en = sqrt(p_en.var)
    sd_ex = sqrt(p_ex.var)
    
    # Learned Generative Gaussian Waves: a_p * N(t; mu_p, var_p)
    wave_en = p_en.mass .* exp.(-0.5 .* ((t_grid_a .- p_en.mu) ./ sd_en).^2) ./ (sd_en * sqrt(2 * pi))
    wave_ex = p_ex.mass .* exp.(-0.5 .* ((t_grid_a .- p_ex.mu) ./ sd_ex).^2) ./ (sd_ex * sqrt(2 * pi))
    
    # Normalize peak height visually per row for clean ridge presentation
    max_w = max(maximum(wave_en), maximum(wave_ex))
    wave_en_norm = wave_en ./ max_w
    wave_ex_norm = wave_ex ./ max_w
    
    # Shaded dwell bridge between entry and exit peaks
    dwell_mask = (t_grid_a .>= p_en.mu) .& (t_grid_a .<= p_ex.mu)
    dwell_ms = (p_ex.mu - p_en.mu) * 1000.0
    
    ax.fill_between(t_grid_a, 0, wave_en_norm, color=cfg.col, alpha=0.30)
    ax.plot(t_grid_a, wave_en_norm, color=cfg.col, linewidth=2.0, label=row_idx == 5 ? "Port 7 Entry" : "Entry Peak")
    
    ax.fill_between(t_grid_a, 0, wave_ex_norm, color="#64748b", alpha=0.20)
    ax.plot(t_grid_a, wave_ex_norm, color="#475569", linewidth=1.8, linestyle="--", label=row_idx == 5 ? "Reward Command" : "Exit Peak")
    
    # Shaded bridge
    ax.fill_between(t_grid_a[dwell_mask], 0, 0.45, color=cfg.col, alpha=0.18, hatch="//", edgecolor=cfg.col, linewidth=0.5)
    
    # Dwell annotation callout
    t_mid_dwell = (p_en.mu + p_ex.mu) / 2.0
    dwell_label = row_idx == 5 ? @sprintf("Rew Latency: %.0f ms", dwell_ms) : @sprintf("Dwell: %.0f ms", dwell_ms)
    ax.text(t_mid_dwell, 0.52, dwell_label,
            ha="center", va="bottom", fontsize=8.2, fontweight="bold", color=cfg.col,
            bbox=Dict("boxstyle"=>"round,pad=0.2", "fc"=>"#ffffff", "ec"=>cfg.col, "lw"=>0.8, "alpha"=>0.95))
    
    # Empirical baseline poke ticks from all 229 occurrences
    pokes_en_rel = Float64[]
    pokes_ex_rel = Float64[]
    for occ in occs_m1
        t0 = occ.tau
        i1 = searchsortedfirst(ts, t0 - 0.20)
        i2 = searchsortedlast(ts, t0 + 2.20)
        for idx in i1:i2
            if ch[idx] == cfg.c_en
                push!(pokes_en_rel, ts[idx] - t0)
            elseif ch[idx] == cfg.c_ex
                push!(pokes_ex_rel, ts[idx] - t0)
            end
        end
    end
    
    # Draw faint baseline ticks
    if !isempty(pokes_en_rel)
        ax.scatter(pokes_en_rel, fill(-0.06, length(pokes_en_rel)), marker="|", s=40, color=cfg.col, alpha=0.35, linewidth=1.0)
    end
    if !isempty(pokes_ex_rel)
        ax.scatter(pokes_ex_rel, fill(-0.14, length(pokes_ex_rel)), marker="|", s=30, color="#64748b", alpha=0.25, linewidth=0.8)
    end
    
    # Milestone guide line at sequence onset (tau = 0)
    ax.axvline(0.0, color="#94a3b8", linestyle=":", linewidth=0.9, zorder=1)
    
    ax.set_xlim(-0.20, 2.20)
    ax.set_ylim(-0.22, 1.25)
    ax.set_yticks([])
    ax.set_ylabel(cfg.name, fontsize=9.2, fontweight="bold", color="#1e293b", rotation=0, ha="right", va="center", labelpad=8)
    
    if row_idx < 5
        ax.tick_params(labelbottom=false)
    else
        ax.set_xlabel("Time from Neural Sequence Onset (s, Motif 1 tau = 0)", fontsize=9.5, fontweight="bold", labelpad=3)
        ax.set_xticks([-0.2, 0.0, 0.5, 1.0, 1.5, 2.0, 2.2])
    end
    
    if row_idx == 1
        ax.legend(loc="upper right", fontsize=8.0, framealpha=0.92)
    end
end

# ==============================================================================
# PANEL B: OUT-OF-SAMPLE VALIDATION & SURROGATE CONTROLS
# ==============================================================================
println("Rendering Panel B: Controls & Out-of-Sample Predictions...")
gs_b = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[2], width_ratios=[1.15, 3.20, 1.15], wspace=0.26)

# ── Subpanel B1: Time-Shift Null (Legend-Free with Direct Headroom Callouts) ──
ax_b1 = fig.add_subplot(gs_b[1])
ax_b1.text(-0.24, 1.18, "B1  Time-Shift Null", transform=ax_b1.transAxes, fontsize=11.5, fontweight="bold", color="#0f172a")

bars_b1 = ax_b1.bar([1, 2], [0.0, 84.0], width=0.55, color=["#94a3b8", "#ea580c"], edgecolor="#1e293b", linewidth=1.0)
ax_b1.set_xlim(0.3, 2.7)
ax_b1.set_ylim(0, 115) # +25% vertical headroom
ax_b1.set_xticks([1, 2])
ax_b1.set_xticklabels(["Shifted Null\n(20 Surrogates)", "Real Data\n(Native PPSeq)"], fontsize=8.5, fontweight="bold")
ax_b1.set_ylabel("Recovered Packets", fontsize=9.5, fontweight="bold")

# Direct numeric callouts
ax_b1.text(1, 4.0, "0 ± 0 pkts\n(0.0% mass)", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#64748b")
ax_b1.text(2, 87.0, "84 packets\n(88.5% mass)\np < 0.0476*", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#c2410c")

# ── Subpanel B2: Held-Out Second Half Strip (25s Window, Same Port Order) ──
ax_b2 = fig.add_subplot(gs_b[2])
ax_b2.text(-0.06, 1.18, "B2  Out-of-Sample Behavioral Decoding on Held-Out Second Half (T/2 to T)",
           transform=ax_b2.transAxes, fontsize=11.5, fontweight="bold", color="#0f172a")

# Window: t in [1108.0, 1133.0 s] containing ~8 active cycles
t_win_start = 1108.0
t_win_end   = 1133.0
t_grid_b2 = range(t_win_start, t_win_end, length=800)

# Extract occurrences in or near this window
occs_strip = [occ for (id, occ) in occs if t_win_start - 3.0 <= occ.tau <= t_win_end + 1.0]

# Render each port track matching Panel A order (top to bottom: Port 2, Port 1, Port 6, Port 3, Port 7/Rew)
for (p_idx, cfg) in enumerate(port_configs)
    y_base = 5 - p_idx + 1 # Stack: Row 5 at top, Row 1 at bottom
    
    # Continuous generative prediction lambda_pred(t, c)
    pred_wave = fill(0.05, length(t_grid_b2)) # baseline rate
    
    for occ in occs_strip
        m_id = occ.motif
        m_obj = ck.motifs[m_id]
        t0 = occ.tau
        for pkt in m_obj.packets
            if pkt.channel == cfg.c_en
                mu_eval = t0 + pkt.mu
                sd = sqrt(pkt.var)
                pred_wave .+= (pkt.mass ./ (sd * sqrt(2 * pi))) .* exp.(-0.5 .* ((t_grid_b2 .- mu_eval) ./ sd).^2)
            end
        end
    end
    
    # Normalize wave for visual stacking
    max_p = max(maximum(pred_wave), 0.5)
    wave_plot = (pred_wave ./ max_p) .* 0.65
    
    ax_b2.fill_between(t_grid_b2, y_base, y_base .+ wave_plot, color=cfg.fill, alpha=0.60)
    ax_b2.plot(t_grid_b2, y_base .+ wave_plot, color=cfg.col, linewidth=1.5)
    ax_b2.axhline(y_base, color="#cbd5e1", linewidth=0.6, linestyle=":")
    
    # Overlay actual second-half unseen pokes
    i1_strip = searchsortedfirst(ts, t_win_start)
    i2_strip = searchsortedlast(ts, t_win_end)
    pokes_strip = [ts[idx] for idx in i1_strip:i2_strip if ch[idx] == cfg.c_en]
    
    if !isempty(pokes_strip)
        ax_b2.scatter(pokes_strip, fill(y_base + 0.35, length(pokes_strip)),
                      marker="D", s=28, color=cfg.col, edgecolors="#1e293b", linewidth=0.8, zorder=5)
    end
end

ax_b2.set_xlim(t_win_start, t_win_end)
ax_b2.set_ylim(0.5, 5.8)
ax_b2.set_yticks([1.35, 2.35, 3.35, 4.35, 5.35])
ax_b2.set_yticklabels(["Port 7 / Rew", "Port 3", "Port 6", "Port 1", "Port 2"], fontsize=8.5, fontweight="bold")
ax_b2.set_xlabel("Session Physical Time on Held-Out Test Half (s)", fontsize=9.5, fontweight="bold", labelpad=3)

# ── Subpanel B3: Split-Half CV Generalization (Direct Headroom Callouts) ──
ax_b3 = fig.add_subplot(gs_b[3])
ax_b3.text(-0.24, 1.18, "B3  Split-Half Generalization", transform=ax_b3.transAxes, fontsize=11.5, fontweight="bold", color="#0f172a")

bars_b3 = ax_b3.bar([1, 2], [-2020.0, 4556.0], width=0.55, color=["#94a3b8", "#0284c7"], edgecolor="#1e293b", linewidth=1.0)
ax_b3.axhline(0.0, color="#1e293b", linewidth=0.8)
ax_b3.set_xlim(0.3, 2.7)
ax_b3.set_ylim(-3000, 6200) # +25% vertical headroom
ax_b3.set_xticks([1, 2])
ax_b3.set_xticklabels(["Shifted\nSurrogate", "Unseen\nTest Half"], fontsize=8.5, fontweight="bold")
ax_b3.set_ylabel("Out-of-Sample \$\\Delta \\ln L_{\\mathrm{test}}\$ (nats)", fontsize=9.2, fontweight="bold")

# Direct numeric callouts
ax_b3.text(1, -2200, "-2,020 nats\n(Null floor)", ha="center", va="top", fontsize=8.2, fontweight="bold", color="#475569")
ax_b3.text(2, 4800, "+4,556 nats\n(+1.39 nats/ev)\n4.0× focus", ha="center", va="bottom", fontsize=8.2, fontweight="bold", color="#0369a1")

# Save Outputs
out_png = joinpath(@__DIR__, "fig2_behavioral_correspondence.png")
println("Saving Figure 2 to: ", out_png)
PyPlot.savefig(out_png, dpi=300, bbox_inches="tight")
PyPlot.close(fig)

println("="^90)
println("FIGURE 2 GENERATION COMPLETE.")
println("="^90)
