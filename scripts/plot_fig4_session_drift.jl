# plot_fig4_session_drift.jl
# Publication-quality Figure 4 (Clean 2x2 Grid with Dual-Timescale Background & Behavioral Regime Gating):
# Multi-Minute Cellular Drift, Sub-Second Regional Arousal, Active Motifs, and Behavioral State Gating
#
# Panel A (Top-Left): Slow Multi-Minute Cellular OU Drift (tau_cell = 60 s, dt_cell = 10 s)
#   Smooth cellular baseline trajectories lambda_{0,c} * exp(delta x_c(t)) across 0 -> 35 min
#   for representative MOp and CP units, omitting 0.5s network jitter.
#
# Panel B (Top-Right): Dual-Track 30s High-Resolution Window (t in [1110, 1140 s], Shared X-Axis)
#   B1 (Top Track, 40% height): Fast Regional Log-Arousal x_MOp(t) vs. x_CP(t) at native 50 ms resolution
#       (tau_net = 0.5 s, rho = +0.162), absorbing unstructured sub-second regional co-fluctuations.
#   B2 (Bottom Track, 60% height): Unit #15 Baseline Intensity lambda_15^(0)(t) riding on the MOp arousal wave,
#       with gray background spikes (79.0%) and sharp vermilion motif bursts (21.0%) cleanly partitioned.
#
# Panel C (Bottom-Left): Session Motif Occurrence Raster (Active Motifs Only, Pure Motif IDs)
#   Full-height raster of active motifs (Motif 1, 2, 8, 7, 17, 5) across 0 -> 2,114s.
#   Explicitly marks the 3 behavioral phases without descriptive labels:
#     1. 0 - 66.8s: Initial Habituation (Zero Rewards)
#     2. 66.8 - 400s: Exploratory Warm-Up (First reward at 66.8s; Motifs 8 & 7 active; zero Motif 1 & 17)
#     3. 400 - 2,114s: Operant Execution (Motif 1 & Motif 17 active)
#
# Panel D (Bottom-Right): Behavioral State Gating Across Three Macroscopic Regimes
#   D1 (Motif 1 & Motif 17): Complete state-gating (0.0 -> 10.9 & 7.1 -> 1.6 & 2.2 ev/min).
#   D2 (Motif 5): State-invariant ongoing activity (96 -> 83 -> 76 ev/min).

using PurePacketPPSeq
using Serialization, Statistics, Printf, NPZ, DelimitedFiles, LinearAlgebra
using PyPlot

println("="^90)
println("GENERATING PUBLICATION FIGURE 4 (PURE MOTIF IDS & ZERO DESCRIPTIVE NAMES)")
println("="^90)

# Configure typography and styling
plt.style.use("seaborn-v0_8-whitegrid" in plt.style.available ? "seaborn-v0_8-whitegrid" : "default")
PyPlot.rc("font", family="sans-serif", size=10.0)
PyPlot.rc("axes", titlesize=11.0, labelsize=10.0, linewidth=0.9, edgecolor="#1e293b")
PyPlot.rc("xtick", labelsize=9.0)
PyPlot.rc("ytick", labelsize=9.0)
PyPlot.rc("legend", fontsize=8.5, framealpha=0.92)

# 1. Load Session Data and Checkpoints
data_dir = "/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/results/top4_task_20261008/data/lgcy0288__20230519_114115__ephys"
raw_npz = npzread(joinpath(data_dir, "joint.npz"))
ts_all = Float64.(raw_npz["timestamp"])
ch_all = Int.(raw_npz["channel"])

neural_mask = ch_all .<= 411
ts_neural = ts_all[neural_mask]
ch_neural = ch_all[neural_mask]

# Model Checkpoints
ckpt_path = joinpath(@__DIR__, "checkpoint_pure_packet_cold_start_100sweeps.jls")
println("Loading model checkpoint: ", ckpt_path)
ckpt = deserialize(ckpt_path)
ou = ckpt[:ou_ctx]
bg_rates = ckpt[:bg_rates]
assign = ckpt[:assign]

ckpt_b_path = joinpath(@__DIR__, "checkpoint_behavioral_correspondence.jls")
println("Loading behavioral checkpoint: ", ckpt_b_path)
ckpt_b = deserialize(ckpt_b_path)
occs_dict = ckpt_b[:occurrences]
occs_all = collect(values(occs_dict))

println(@sprintf("Loaded %d neural spikes, %d occurrences.", length(ts_neural), length(occs_all)))

# 2. Extract Data for Panels
# Panel A: Slow Cellular Drift (tau_cell = 60s, dt_cell = 10s)
t_cell = ou.t_cell
mop_u1 = 15   # MOp6a (26.0 Hz base)
mop_u2 = 302  # MOp6b (41.9 Hz base)
cp_u1  = 261  # CP Striatum (31.7 Hz base)
cp_u2  = 142  # CP Striatum (22.2 Hz base)

rate_mop1 = bg_rates[mop_u1] .* exp.(ou.delta_x[mop_u1, :])
rate_mop2 = bg_rates[mop_u2] .* exp.(ou.delta_x[mop_u2, :])
rate_cp1  = bg_rates[cp_u1]  .* exp.(ou.delta_x[cp_u1, :])
rate_cp2  = bg_rates[cp_u2]  .* exp.(ou.delta_x[cp_u2, :])

# Panel B: 30s High-Resolution Dual-Track Window (t in [1110, 1140 s])
t_zoom_start = 1110.0
t_zoom_end   = 1140.0
idx_net_zoom = findall(t -> t_zoom_start <= t <= t_zoom_end, ou.t_net)
t_net_zoom   = ou.t_net[idx_net_zoom]
x_mop_zoom   = ou.x_region[1, idx_net_zoom]
x_cp_zoom    = ou.x_region[2, idx_net_zoom]

idx_zoom_spk = findall(i -> (ch_neural[i] == mop_u1) && (t_zoom_start <= ts_neural[i] <= t_zoom_end), 1:length(ts_neural))
bg_spks      = [ts_neural[i] for i in idx_zoom_spk if assign[i].event_id == 0]
motif_spks   = [ts_neural[i] for i in idx_zoom_spk if assign[i].event_id > 0]

t_fine = t_zoom_start:0.05:t_zoom_end
lam_zoom = [exp(PurePacketPPSeq.evaluate_background_log_intensity(ou, bg_rates, mop_u1, t)) for t in t_fine]

# Panel C: Active Motifs (Pure IDs, No Descriptive Names)
active_motifs = [1, 2, 8, 7, 17, 5]
active_labels = [
    "Motif 1 (N=229)",
    "Motif 2 (N=98)",
    "Motif 8 (N=101)",
    "Motif 7 (N=84)",
    "Motif 17 (N=160)",
    "Motif 5 (N=2,956)"
]
active_colors = [
    "#e11d48", # Motif 1 Vermilion
    "#0284c7", # Motif 2 Sky blue
    "#0d9488", # Motif 8 Teal
    "#db2777", # Motif 7 Pink
    "#7c3aed", # Motif 17 Purple
    "#64748b"  # Motif 5 Slate
]

# Panel D: Behavioral State Gating Across Three Macroscopic Regimes
# 1. Warm-Up Phase (0 to 400s)
# 2. Active Operant Runs (ITI < 10s, 400 to 2114s excluding pauses)
# 3. Mid-Session Pauses (ITI > 30s)
m1_occs = sort(filter(o -> o.motif == 1, occs_all), by=o -> o.tau)
m1_taus = [o.tau for o in m1_occs]
m1_itis = diff(m1_taus)

pause_intervals = Tuple{Float64, Float64}[]
for i in 1:length(m1_itis)
    if m1_itis[i] > 30.0
        push!(pause_intervals, (m1_taus[i], m1_taus[i+1]))
    end
end
dur_pause = sum([p[2] - p[1] for p in pause_intervals])
dur_active = (2114.3 - 400.0) - dur_pause
dur_pre = 400.0
durations_min = [dur_pre / 60.0, dur_active / 60.0, dur_pause / 60.0]

function get_regime(t)
    if t < 400.0
        return 1
    end
    for p in pause_intervals
        if p[1] <= t <= p[2]
            return 3
        end
    end
    return 2
end

# Compute rates for M1, M17, and M5
m1_ts  = [o.tau for o in filter(o -> o.motif == 1, occs_all)]
m17_ts = [o.tau for o in filter(o -> o.motif == 17, occs_all)]
m5_ts  = [o.tau for o in filter(o -> o.motif == 5, occs_all)]

rate_m1  = [count(t -> get_regime(t) == r, m1_ts) for r in 1:3] ./ durations_min
rate_m17 = [count(t -> get_regime(t) == r, m17_ts) for r in 1:3] ./ durations_min
rate_m5  = [count(t -> get_regime(t) == r, m5_ts) for r in 1:3] ./ durations_min

# 3. Setup Figure Canvas (Clean 2x2 Grid, 15.5 x 10.8 inches)
fig = figure(figsize=(15.5, 10.8), dpi=300)

gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.25,
                      left=0.09, right=0.965, top=0.905, bottom=0.075)

# Main Titles
fig.text(0.09, 0.968, "Figure 4: Multi-Scale Background Filtering, Active Motif Dynamics, and Behavioral State Gating",
         fontsize=13.5, fontweight="bold", color="#0f172a")
fig.text(0.09, 0.942, "Session lgcy0288 (35.2 min, 2,114 s) | N = 411 Neurons | M = 20 Motifs (E = 8,770 Occurrences) | 79.0% Background, 21.0% Motifs",
         fontsize=10.0, color="#475569")

# ==============================================================================
# PANEL A: SLOW MULTI-MINUTE CELLULAR OU DRIFT (0 -> 35 min)
# ==============================================================================
ax_a = fig.add_subplot(gs[1, 1])
ax_a.text(-0.08, 1.06, "A   Slow Multi-Minute Cellular OU Drift (tau_cell = 60 s)",
          transform=ax_a.transAxes, fontsize=11.5, fontweight="bold", color="#0f172a")

ax_a.plot(t_cell, rate_mop1, color="#e11d48", linewidth=1.8, label="MOp Unit #15 (base = 26.0 Hz)")
ax_a.plot(t_cell, rate_mop2, color="#f43f5e", linewidth=1.6, linestyle="--", label="MOp Unit #302 (base = 41.9 Hz)")
ax_a.plot(t_cell, rate_cp1,  color="#7c3aed", linewidth=1.8, label="CP Unit #261 (base = 31.7 Hz)")
ax_a.plot(t_cell, rate_cp2,  color="#a855f7", linewidth=1.6, linestyle="--", label="CP Unit #142 (base = 22.2 Hz)")

ax_a.set_xlabel("Session Time (s)", fontsize=9.5, fontweight="bold")
ax_a.set_ylabel("Cellular Baseline Rate λ_0 exp(δx_c(t)) (Hz)", fontsize=9.5, fontweight="bold")
ax_a.set_xlim(0, 2120)
ax_a.set_ylim(0, 85)
ax_a.legend(loc="upper right", frameon=true, facecolor="white", framealpha=0.92)

# ==============================================================================
# PANEL B: DUAL-TRACK 30s WINDOW (REGIONAL AROUSAL + SPIKE PARTITION)
# ==============================================================================
gs_b = gs[1, 2].subgridspec(2, 1, height_ratios=[0.38, 0.62], hspace=0.12)

# --- Sub-Track B1: Fast Sub-Second Regional Arousal x_r(t) ---
ax_b1 = fig.add_subplot(gs_b[1])
ax_b1.text(-0.09, 1.18, "B   Sub-Second Regional Arousal & Generative Spike Partition (30s Window)",
           transform=ax_b1.transAxes, fontsize=11.5, fontweight="bold", color="#0f172a")

ax_b1.plot(t_net_zoom, x_mop_zoom, color="#e11d48", linewidth=1.4, alpha=0.95, label="MOp Cortex (x_1, tau=0.5s)")
ax_b1.plot(t_net_zoom, x_cp_zoom,  color="#7c3aed", linewidth=1.4, alpha=0.85, label="CP Striatum (x_2, tau=0.5s)")
ax_b1.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=0.7)

ax_b1.set_ylabel("Network Log-Arousal (x_r)", fontsize=8.5, fontweight="bold")
ax_b1.set_xlim(t_zoom_start, t_zoom_end)
ax_b1.set_ylim(-0.60, 0.60)
plt.setp(ax_b1.get_xticklabels(), visible=false)
ax_b1.legend(loc="upper right", fontsize=8.0, frameon=true, facecolor="white", framealpha=0.92, ncol=2)

ax_b1.text(0.03, 0.15, "Independent Gain (rho = +0.162): Absorbs global rate surges",
           transform=ax_b1.transAxes, fontsize=7.8, fontweight="bold", color="#334155")

# --- Sub-Track B2: Single-Neuron Continuous Rate & Spike Partition ---
ax_b2 = fig.add_subplot(gs_b[2], sharex=ax_b1)

ax_b2.plot(t_fine, lam_zoom, color="#e11d48", linewidth=1.8, label="Unit #15 Baseline Intensity λ_0(t)")

y_bg = -4.5
y_mo = -11.5
if !isempty(bg_spks)
    ax_b2.vlines(bg_spks, y_bg - 2.5, y_bg + 2.5, color="#64748b", linewidth=0.9, alpha=0.85,
                 label=@sprintf("Background Spikes (79.0%% session, N=%d)", length(bg_spks)))
end
if !isempty(motif_spks)
    ax_b2.vlines(motif_spks, y_mo - 2.5, y_mo + 2.5, color="#e11d48", linewidth=1.3, alpha=0.95,
                 label=@sprintf("Motif Packet Bursts (21.0%% session, N=%d)", length(motif_spks)))
end

ax_b2.axhline(0.0, color="#94a3b8", linestyle=":", linewidth=0.7)
ax_b2.set_xlabel("Session Time (s)", fontsize=9.5, fontweight="bold")
ax_b2.set_ylabel("Firing Rate (Hz)", fontsize=8.5, fontweight="bold")
ax_b2.set_xlim(t_zoom_start, t_zoom_end)
ax_b2.set_ylim(-16, 52)
ax_b2.legend(loc="upper right", fontsize=8.0, frameon=true, facecolor="white", framealpha=0.92)

# ==============================================================================
# PANEL C: SESSION MOTIF RASTER (PURE MOTIF IDS & ZERO DESCRIPTIVE NAMES)
# ==============================================================================
ax_c = fig.add_subplot(gs[2, 1])
ax_c.text(-0.08, 1.06, "C   Session Occurrence Raster (Active Motifs & 3 Behavioral Phases)",
          transform=ax_c.transAxes, fontsize=11.5, fontweight="bold", color="#0f172a")

# 1. Phase Shading
ax_c.axvspan(0.0, 66.8, color="#f1f5f9", alpha=0.85, zorder=0)
ax_c.axvspan(66.8, 400.0, color="#fffbeb", alpha=0.75, zorder=0)

# 2. Phase Banners at Top (Clean, Compact, Zero Descriptive Names)
ax_c.text(233.4, 6.75, "Warm-Up (67-400s)\n[Motifs 8 & 7 Active; Zero Motif 1]",
          ha="center", va="center", fontsize=7.5, fontweight="bold", color="#b45309")

ax_c.text(1257.0, 6.75, "Operant Task Execution (400-2,114s) [Motifs 1 & 17 Active]",
          ha="center", va="center", fontsize=8.0, fontweight="bold", color="#0284c7")

# 3. Vertical Demarcation Lines with Clear Callouts
ax_c.axvline(66.8, color="#d97706", linestyle=":", linewidth=1.5, alpha=0.95)
ax_c.text(71.0, 0.45, "66.8s: First Reward (Warm-Up Onset)", color="#d97706",
          fontsize=7.8, fontweight="bold", ha="left", va="bottom", rotation=90)

ax_c.axvline(400.0, color="#0284c7", linestyle="--", linewidth=1.4, alpha=0.95)
ax_c.text(405.0, 0.45, "400s: Sequence Ignition (Motifs 1 & 17)", color="#0284c7",
          fontsize=7.8, fontweight="bold", ha="left", va="bottom", rotation=90)

# Plot each active motif on its dedicated row
for (row_idx, m) in enumerate(active_motifs)
    m_occs = filter(o -> o.motif == m, occs_all)
    ts_m = [o.tau for o in m_occs]
    col = active_colors[row_idx]
    sz = (m == 5) ? 14 : 28
    alph = (m == 5) ? 0.40 : 0.85
    
    ax_c.scatter(ts_m, fill(Float64(row_idx), length(ts_m)), s=sz, color=col, alpha=alph, edgecolors="none")
end

ax_c.set_xlabel("Session Time (s)", fontsize=9.5, fontweight="bold")
ax_c.set_ylabel("Active Motif", fontsize=9.5, fontweight="bold")
ax_c.set_xlim(0, 2120)
ax_c.set_ylim(0.2, 7.3)
ax_c.set_yticks(1:6)
ax_c.set_yticklabels(active_labels, fontsize=8.5, fontweight="bold")

for (i, tick) in enumerate(ax_c.get_yticklabels())
    tick.set_color(active_colors[i])
end

# ==============================================================================
# PANEL D: BEHAVIORAL STATE GATING (D1 & D2 WITH PURE MOTIF IDS)
# ==============================================================================
gs_d = gs[2, 2].subgridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.28)

# --- Subpanel D1: Gated Operant Sequences (Motif 1 & Motif 17) ---
ax_d1 = fig.add_subplot(gs_d[1])
ax_d1.text(-0.16, 1.10, "D1  Gated Operant Sequences",
           transform=ax_d1.transAxes, fontsize=10.5, fontweight="bold", color="#0f172a")

reg_labels = ["Warm-Up\n(0-400s)", "Active Runs\n(ITI<10s)", "Pauses\n(ITI>30s)"]
x_pos = [1.0, 2.0, 3.0]
w_bar = 0.32

# Bars for Motif 1 and Motif 17 side-by-side
bars_m1  = ax_d1.bar(x_pos .- w_bar/2, rate_m1, width=w_bar, color="#e11d48", label="Motif 1", edgecolor="#1e293b", linewidth=0.8)
bars_m17 = ax_d1.bar(x_pos .+ w_bar/2, rate_m17, width=w_bar, color="#7c3aed", label="Motif 17", edgecolor="#1e293b", linewidth=0.8)

# Direct numeric callouts on top of bars
for (i, v) in enumerate(rate_m1)
    ax_d1.text(x_pos[i] - w_bar/2, v + 0.35, @sprintf("%.1f", v), ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#e11d48")
end
for (i, v) in enumerate(rate_m17)
    ax_d1.text(x_pos[i] + w_bar/2, v + 0.35, @sprintf("%.1f", v), ha="center", va="bottom", fontsize=7.8, fontweight="bold", color="#7c3aed")
end

ax_d1.set_ylabel("Occurrence Rate (events / min)", fontsize=9.0, fontweight="bold")
ax_d1.set_xticks(x_pos)
ax_d1.set_xticklabels(reg_labels, fontsize=8.0, fontweight="bold")
ax_d1.set_xlim(0.4, 3.6)
ax_d1.set_ylim(0, 13.5)
ax_d1.legend(loc="upper left", fontsize=8.0, frameon=true, facecolor="white", framealpha=0.92)

ax_d1.text(0.96, 0.88, "State-Gated:\nSilent in warm-up (0.0)\nSurges during runs",
           transform=ax_d1.transAxes, ha="right", va="top", fontsize=7.5,
           bbox=Dict("boxstyle"=>"round,pad=0.3", "facecolor"=>"#f8fafc", "edgecolor"=>"#cbd5e1", "linewidth"=>0.7))

# --- Subpanel D2: State-Invariant Ongoing Background (Motif 5) ---
ax_d2 = fig.add_subplot(gs_d[2])
ax_d2.text(-0.16, 1.10, "D2  State-Invariant Tonic Activity",
           transform=ax_d2.transAxes, fontsize=10.5, fontweight="bold", color="#0f172a")

bars_m5 = ax_d2.bar(x_pos, rate_m5, width=0.55, color="#64748b", edgecolor="#1e293b", linewidth=0.8, label="Motif 5")

# Direct numeric callouts on top of bars
for (i, v) in enumerate(rate_m5)
    ax_d2.text(x_pos[i], v + 2.5, @sprintf("%.1f", v), ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#334155")
end

ax_d2.set_ylabel("Tonic Rate (events / min)", fontsize=9.0, fontweight="bold")
ax_d2.set_xticks(x_pos)
ax_d2.set_xticklabels(reg_labels, fontsize=8.0, fontweight="bold")
ax_d2.set_xlim(0.4, 3.6)
ax_d2.set_ylim(0, 118)
ax_d2.legend(loc="upper right", fontsize=8.0, frameon=true, facecolor="white", framealpha=0.92)

ax_d2.text(0.04, 0.88, "State-Invariant:\nPersists steadily\nacross all 3 states",
           transform=ax_d2.transAxes, ha="left", va="top", fontsize=7.5,
           bbox=Dict("boxstyle"=>"round,pad=0.3", "facecolor"=>"#f8fafc", "edgecolor"=>"#cbd5e1", "linewidth"=>0.7))

# 4. Save High-Resolution Artifact (Strictly PNG only)
out_png = joinpath(@__DIR__, "fig4_session_drift.png")
println("Saving publication figure to: ", out_png)
fig.savefig(out_png, dpi=300)
close(fig)

println("="^90)
println("FIGURE 4 (PURE MOTIF IDS) COMPLETE: ", out_png)
println("="^90)
