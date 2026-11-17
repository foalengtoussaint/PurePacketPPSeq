#!/usr/bin/env python3
"""
scripts/plot_fig1_unsupervised_motifs.py
Publication-quality 3-row x 6-column Figure 1:
Unsupervised Corticostriatal Sequences Discovered Blind to Behavior

Columns:
  1. Motif 1  (Full Sequence Traversal, N = 229 occs)
  2. Motif 2  (Cortical-Led Initiation, N = 98 occs)
  3. Motif 8  (Striatal-Dense Sequence, N = 101 occs)
  4. Motif 7  (Reward Approach / Arrival, N = 84 occs)
  5. Motif 17 (Port Exit / Waypoint Transition, N = 160 occs)
  6. Motif 5  (Internal Non-Task Control, N = 2,956 occs)

Rows:
  Row A (Physical Space): True 2D spatial activation rate in Hz (Activity / Occupancy Time,
                          masking bins < 0.2 s) over 6-port arena, overlaid with faint gray
                          trajectories and directional arrows.
  Row B (Trial Timing): Behavior-aligned conditional spiking density in Hz/unit for MOp (vermilion)
                        and CP (purple), restricted to active trials (Active: X% of trials)
                        on unwarped physical clock (0.0 to 2.2 s from Port 2 entry) normalized
                        by observation seconds per bin.
  Row C (Intrinsic Firing): Motif-aligned intrinsic spiking density in Hz/unit (tau = 0)
                            showing sharp MOp -> CP causal sequencing (CP - MOp lag).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d

# Typography
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['axes.edgecolor'] = '#1e293b'
plt.rcParams['axes.linewidth'] = 1.0

print("=" * 80)
print("GENERATING REFINED 3-ROW FIGURE 1: UNSUPERVISED MOTIFS")
print("=" * 80)

# Paths
ROOT = Path("/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/results/top4_task_20261008")
SESSION = "lgcy0288__20230519_114115__ephys"
DATA_DIR = ROOT / "data" / SESSION
TRACK_FILE = Path("/home/foaleng/Downloads/cached_tracking_lgcy0288.npz")
MOTIFS_DATA_FILE = Path("/home/foaleng/Documents/PurePacketPPSeq/scripts/fig1_motifs_data.npz")
UNITS_FILE = DATA_DIR / "units.csv"
BEHAVIOR_FILE = DATA_DIR / "behavior.json"

OUT_PNG = Path("/home/foaleng/Documents/PurePacketPPSeq/scripts/fig1_unsupervised_motifs.png")
OUT_PDF = Path("/home/foaleng/Documents/PurePacketPPSeq/scripts/fig1_unsupervised_motifs.pdf")

# 1. Load Units and Regional Mask
units_df = pd.read_csv(UNITS_FILE)
regions = units_df["region"].values
is_mop = np.array(["MOp" in r for r in regions])
is_cp  = np.array(["CP" in r for r in regions])
n_mop = int(np.sum(is_mop))
n_cp  = int(np.sum(is_cp))
print(f"Loaded {len(units_df)} units: MOp = {n_mop}, CP = {n_cp}")

# 2. Load Behavior & Tracking
with open(BEHAVIOR_FILE) as f:
    df_beh = pd.DataFrame(json.load(f))

entries = df_beh[df_beh["name"].str.endswith("_entry")].sort_values("timestamp").to_dict("records")
exits = df_beh[df_beh["name"].str.endswith("_exit")].sort_values("timestamp").to_dict("records")

pokes = []
for en in entries:
    p_name = int(en["name"].split("_")[1])
    for ex in exits:
        if ex["name"] == f"port_{p_name}_exit" and ex["timestamp"] >= en["timestamp"]:
            dwell = ex["timestamp"] - en["timestamp"]
            if dwell < 20.0:
                pokes.append({"port": p_name, "entry": en["timestamp"], "exit": ex["timestamp"], "dwell": dwell, "trial": en["trial"]})
            break
df_p = pd.DataFrame(pokes).sort_values("entry").reset_index(drop=True)

track = np.load(TRACK_FILE)
t_bpod = track["t_bpod"]
imp_x = track["imp_x"].copy()
imp_y = track["imp_y"].copy()
imp_c = track["imp_c"]

bad = (imp_c < 0.6) | (imp_x < 50) | (imp_x > 1250) | (imp_y < 100) | (imp_y > 900)
imp_x[bad] = np.nan
imp_y[bad] = np.nan

# 3. Arena Geometry & Splines
corridors = [(2, 1), (1, 6), (6, 3), (3, 7), (7, 2)]
ports_list = [2, 1, 6, 3, 7]
N_BINS = 25

y_bot = 230
y_top = 470
ports_geom = {
    'P1': (220, y_bot),
    'P2': (500, y_bot),
    'P3': (780, y_bot),
    'P4/5': (220, y_top),
    'P6': (500, y_top),
    'P7': (780, y_top)
}

# Spline definitions
x_p2_p1 = np.linspace(ports_geom['P2'][0], ports_geom['P1'][0], N_BINS)
y_p2_p1 = np.full(N_BINS, y_bot)

hook_knots_x = [ports_geom['P1'][0], 170, 175, 235, 360, ports_geom['P6'][0]]
hook_knots_y = [y_bot, y_bot - 10, y_bot + 45, y_bot + 65, 380, ports_geom['P6'][1]]
tck_16, _ = splprep([hook_knots_x, hook_knots_y], s=0, k=3)
u_16 = np.linspace(0, 1, N_BINS)
x_p1_p6, y_p1_p6 = splev(u_16, tck_16)

desc_knots_x = [ports_geom['P6'][0], 560, 680, ports_geom['P3'][0]]
desc_knots_y = [ports_geom['P6'][1], 440, 320, y_bot]
tck_63, _ = splprep([desc_knots_x, desc_knots_y], s=0, k=3)
u_63 = np.linspace(0, 1, N_BINS)
x_p6_p3, y_p6_p3 = splev(u_63, tck_63)

x_p3_p7 = np.full(N_BINS, ports_geom['P3'][0])
y_p3_p7 = np.linspace(y_bot, y_top, N_BINS)

ret_knots_x = [ports_geom['P7'][0], 720, 610, ports_geom['P2'][0]]
ret_knots_y = [y_top, 420, 310, y_bot]
tck_72, _ = splprep([ret_knots_x, ret_knots_y], s=0, k=3)
u_72 = np.linspace(0, 1, N_BINS)
x_p7_p2, y_p7_p2 = splev(u_72, tck_72)

corridor_paths = {
    (2, 1): (x_p2_p1, y_p2_p1),
    (1, 6): (x_p1_p6, y_p1_p6),
    (6, 3): (x_p6_p3, y_p6_p3),
    (3, 7): (x_p3_p7, y_p3_p7),
    (7, 2): (x_p7_p2, y_p7_p2)
}

# 4. Compute True Spatial Activation Rate in Hz (Activity / Occupancy Time)
motifs_data = np.load(MOTIFS_DATA_FILE)
ev_ts = motifs_data["occ_taus"]
ev_motif = motifs_data["occ_motifs"]
motif_ids = [1, 2, 8, 7, 17, 5]

corridor_events = {m: {c: np.zeros(N_BINS) for c in corridors} for m in motif_ids}
corridor_occupancy = {c: np.zeros(N_BINS) for c in corridors}
port_events = {m: {p: 0 for p in ports_list} for m in motif_ids}
port_dwell_time = {p: 0.0 for p in ports_list}

# Measure continuous dwell time across ports
for p in ports_list:
    sub = df_p[df_p["port"] == p]
    port_dwell_time[p] = sub["dwell"].sum()
    for m in motif_ids:
        m_mask = ev_motif == m
        m_taus = ev_ts[m_mask]
        ev_cnt = sum(np.sum((m_taus >= r["entry"] - 0.2) & (m_taus <= r["exit"] + 0.2)) for _, r in sub.iterrows())
        port_events[m][p] = ev_cnt

# Measure spatial occupancy and event counts along each corridor
for pA, pB in corridors:
    c = (pA, pB)
    for i in range(len(df_p) - 1):
        r1 = df_p.iloc[i]
        r2 = df_p.iloc[i+1]
        if r1["port"] == pA and r2["port"] == pB:
            dt = r2["entry"] - r1["exit"]
            if 0.05 <= dt <= 3.5:
                mask = (t_bpod >= r1["exit"]) & (t_bpod <= r2["entry"])
                t_sub = t_bpod[mask]
                x_sub = imp_x[mask]
                y_sub = imp_y[mask]
                if len(t_sub) >= 4 and np.sum(np.isnan(x_sub)) == 0:
                    dx = np.diff(x_sub)
                    dy = np.diff(y_sub)
                    dist = np.sqrt(dx**2 + dy**2)
                    cum_dist = np.insert(np.cumsum(dist), 0, 0)
                    total_dist = cum_dist[-1]
                    if total_dist > 1.0:
                        s_norm = cum_dist / total_dist
                        dt_samples = np.diff(t_sub)
                        mid_s = (s_norm[:-1] + s_norm[1:]) / 2.0
                        for s_val, dt_s in zip(mid_s, dt_samples):
                            b = min(int(s_val * N_BINS), N_BINS - 1)
                            corridor_occupancy[c][b] += dt_s
                        
                        m_mask = (ev_ts >= r1["exit"] - 0.1) & (ev_ts <= r2["entry"] + 0.1)
                        for t_ev, m in zip(ev_ts[m_mask], ev_motif[m_mask]):
                            if m in motif_ids:
                                s_ev = np.interp(np.clip(t_ev, t_sub[0], t_sub[-1]), t_sub, s_norm)
                                b = min(int(s_ev * N_BINS), N_BINS - 1)
                                corridor_events[m][c][b] += 1

# Rate in Hz = Events / Occupancy Time (masking bins with Time < 0.2 s)
corridor_rate_hz = {m: {c: np.zeros(N_BINS) for c in corridors} for m in motif_ids}
dwell_rate_hz = {m: {} for m in motif_ids}

for m in motif_ids:
    for c in corridors:
        occ = corridor_occupancy[c]
        rates = np.where(occ >= 0.20, corridor_events[m][c] / occ, 0.0)
        corridor_rate_hz[m][c] = gaussian_filter1d(rates, sigma=1.0)
    for p in ports_list:
        tot_dwell = port_dwell_time[p]
        dwell_rate_hz[m][p] = port_events[m][p] / max(tot_dwell, 0.20)

# 5. Extract Full Behavioral Cycles (Port 2 Entry -> Next Port 2 Entry)
p2_entries = df_beh[df_beh["name"] == "port_2_entry"].sort_values("timestamp").reset_index(drop=True)
cycles = []
for i in range(len(p2_entries) - 1):
    t0 = p2_entries.iloc[i]["timestamp"]
    t1 = p2_entries.iloc[i+1]["timestamp"]
    dur = t1 - t0
    sub = df_beh[(df_beh["timestamp"] >= t0) & (df_beh["timestamp"] <= t1)]
    has_rew = np.any(sub["name"] == "reward_command")
    has_p7  = np.any(sub["name"] == "port_7_entry")
    if has_rew and has_p7 and 1.5 <= dur <= 10.0:
        cycles.append({"t0": t0, "t1": t1, "dur": dur, "trial": p2_entries.iloc[i]["trial"]})
df_cycles = pd.DataFrame(cycles)
N_CYCLES = len(df_cycles)
print(f"Extracted {N_CYCLES} valid full-loop P2 -> next P2 cycles (median dur = {df_cycles['dur'].median():.2f} s).")

# 6. Colormaps and Metadata
cmap_blue   = LinearSegmentedColormap.from_list('m_blue', [(0.0, '#f1f5f9'), (0.15, '#bae6fd'), (0.45, '#38bdf8'), (0.80, '#0284c7'), (1.0, '#082f49')])
cmap_teal   = LinearSegmentedColormap.from_list('m_teal', [(0.0, '#f1f5f9'), (0.15, '#99f6e4'), (0.45, '#14b8a6'), (0.80, '#0f766e'), (1.0, '#134e4a')])
cmap_purple = LinearSegmentedColormap.from_list('m_purp', [(0.0, '#f1f5f9'), (0.15, '#ddd6fe'), (0.45, '#a855f7'), (0.80, '#7c3aed'), (1.0, '#4c1d95')])
cmap_rose   = LinearSegmentedColormap.from_list('m_rose', [(0.0, '#f1f5f9'), (0.15, '#fbcfe8'), (0.45, '#f472b6'), (0.80, '#db2777'), (1.0, '#831843')])
cmap_amber  = LinearSegmentedColormap.from_list('m_amb',  [(0.0, '#f1f5f9'), (0.15, '#fde68a'), (0.45, '#fbbf24'), (0.80, '#d97706'), (1.0, '#78350f')])
cmap_slate  = LinearSegmentedColormap.from_list('m_slate',[(0.0, '#f1f5f9'), (0.15, '#cbd5e1'), (0.45, '#64748b'), (0.80, '#334155'), (1.0, '#0f172a')])

# Unified Regional Colors (Shared across Row B and Row C)
col_mop = "#E66101" # Vermilion
col_cp  = "#5E3C99" # Royal Purple

motif_meta = [
    {
        "id": 1,
        "name": "Motif 1: Full Sequence",
        "sub": "P2 → P1 → P6 → P3 → P7",
        "cmap": cmap_blue,
        "line_col": "#0284c7",
        "fill_col": "#bae6fd",
        "vmax": 1.2,
        "t_span": (0.0, 3.5)
    },
    {
        "id": 2,
        "name": "Motif 2: Cortical Initiation",
        "sub": "P2 → P1 Launch",
        "cmap": cmap_teal,
        "line_col": "#0d9488",
        "fill_col": "#99f6e4",
        "vmax": 1.8,
        "t_span": (-0.2, 1.5)
    },
    {
        "id": 8,
        "name": "Motif 8: Striatal-Dense",
        "sub": "P1 → P6 Mid-Run",
        "cmap": cmap_purple,
        "line_col": "#7c3aed",
        "fill_col": "#ddd6fe",
        "vmax": 1.5,
        "t_span": (-0.2, 1.5)
    },
    {
        "id": 7,
        "name": "Motif 7: Reward Approach",
        "sub": "P3 → P7 Goal",
        "cmap": cmap_rose,
        "line_col": "#db2777",
        "fill_col": "#fbcfe8",
        "vmax": 0.8,
        "t_span": (-0.2, 1.0)
    },
    {
        "id": 17,
        "name": "Motif 17: Port Exit / Waypoint",
        "sub": "P7 → P2 Transition",
        "cmap": cmap_amber,
        "line_col": "#d97706",
        "fill_col": "#fde68a",
        "vmax": 1.2,
        "t_span": (-0.2, 0.7)
    },
    {
        "id": 5,
        "name": "Motif 5: Non-Task Control",
        "sub": "Internal Baseline",
        "cmap": cmap_slate,
        "line_col": "#475569",
        "fill_col": "#cbd5e1",
        "vmax": 2.5,
        "t_span": (-0.1, 0.15)
    }
]

# Helper: Render Row 1 Arena Schema Panel with True 2D Spatial Rates in Hz
def render_schema_panel(ax, corr_rate_m, port_rate_m, title, sub_title, cmap, vmax):
    ax.set_facecolor('#ffffff')
    ax.set_xlim(80, 920)
    ax.set_ylim(130, 580)
    ax.set_aspect('equal')
    ax.axis('off')
    
    wall = patches.FancyBboxPatch((110, 145), 780, 415,
                                 boxstyle="round,pad=6,rounding_size=14",
                                 facecolor='#fafafa', edgecolor='#e2e8f0', linewidth=1.2, zorder=1)
    ax.add_patch(wall)
    
    ax.plot([140, 860], [(y_bot + y_top) / 2, (y_bot + y_top) / 2], color='#e2e8f0', linestyle='--', linewidth=0.9, zorder=2)
    ax.text(125, y_top, 'TOP TIER', ha='center', va='center', fontsize=6.5, fontweight='bold', color='#94a3b8', rotation=90, zorder=2)
    ax.text(125, y_bot, 'FLOOR', ha='center', va='center', fontsize=6.5, fontweight='bold', color='#94a3b8', rotation=90, zorder=2)
    
    norm = Normalize(vmin=0, vmax=max(vmax, 0.1))
    
    # 1. Base corridors (Faint gray trajectory tracks)
    for (pA, pB), (px, py) in corridor_paths.items():
        ax.plot(px, py, color='#cbd5e1', linewidth=2.4, alpha=0.45, zorder=2)
        
    # 2. Continuous Spatial Rate LineCollection (in Hz)
    for (pA, pB), (px, py) in corridor_paths.items():
        rate_arr = corr_rate_m[(pA, pB)]
        points = np.array([px, py]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        seg_rates = (rate_arr[:-1] + rate_arr[1:]) / 2.0
        lwidths = 2.0 + 5.5 * np.clip(seg_rates / max(vmax, 0.1), 0, 1)
        
        lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=lwidths, capstyle='round', zorder=4)
        lc.set_array(seg_rates)
        ax.add_collection(lc)
        
        # Direction arrow
        mid_idx = int(N_BINS * 0.55)
        dx = px[mid_idx+1] - px[mid_idx]
        dy = py[mid_idx+1] - py[mid_idx]
        arr_norm = np.sqrt(dx**2 + dy**2)
        if arr_norm > 0:
            dx_u = dx / arr_norm * 8.5
            dy_u = dy / arr_norm * 8.5
            arr_col = cmap(norm(seg_rates[mid_idx]))
            ax.annotate('', xy=(px[mid_idx] + dx_u, py[mid_idx] + dy_u), xytext=(px[mid_idx], py[mid_idx]),
                        arrowprops=dict(arrowstyle="-|>", color=arr_col, lw=2.0, mutation_scale=10), zorder=5)

    # 3. Ports (In-port dwell rates in Hz)
    for pname, pxy in ports_geom.items():
        if pname == 'P4/5':
            ax.add_patch(patches.Circle(pxy, 32, facecolor='#ffffff', edgecolor='#cbd5e1', linestyle=':', linewidth=1.2, zorder=6))
            ax.text(pxy[0], pxy[1], 'P4/5', ha='center', va='center', fontsize=8.0, color='#cbd5e1', zorder=7)
            continue
            
        p_num = int(pname[1])
        d_rate = port_rate_m.get(p_num, 0.0)
        
        ax.add_patch(patches.Circle(pxy, 32, facecolor='#ffffff', edgecolor='#334155', linewidth=1.6, zorder=6))
        
        dwell_col = cmap(norm(d_rate))
        fill_alpha = 0.08 + 0.85 * np.clip(d_rate / max(vmax, 0.1), 0, 1)
        ax.add_patch(patches.Circle(pxy, 27, facecolor=dwell_col, alpha=fill_alpha, edgecolor='none', zorder=7))
        ax.add_patch(patches.Circle(pxy, 27, facecolor='none', edgecolor=dwell_col, linewidth=1.3, zorder=8))
        
        lbl_col = '#ffffff' if (d_rate / max(vmax, 0.1)) > 0.50 else '#1e293b'
        ax.text(pxy[0], pxy[1], pname, ha='center', va='center', fontsize=9.0, fontweight='bold', color=lbl_col, zorder=9)

    ax.set_title(f"{title}\n{sub_title}", fontsize=9.2, fontweight='bold', color='#0f172a', linespacing=1.2, pad=10)

    # Inset Colorbar (labeled in Hz)
    cbar_ax = ax.inset_axes([0.16, 0.03, 0.68, 0.045])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cb.ax.tick_params(labelsize=6.8, pad=1)
    if vmax < 1.0:
        cbar_lbl = f"Rate: 0 – {vmax:.2f} Hz"
    elif vmax < 2.0:
        cbar_lbl = f"Rate: 0 – {vmax:.1f} Hz"
    else:
        cbar_lbl = f"Rate: 0 – {vmax:.0f} Hz"
    cb.set_label(cbar_lbl, fontsize=7.2, fontweight='bold', color='#334155', labelpad=2)


# ==============================================================================
# SET UP 3-ROW x 6-COLUMN FIGURE LAYOUT
# ==============================================================================
print("Setting up publication figure layout (21.0 x 11.5 inches)...")
fig = plt.figure(figsize=(21.0, 11.5), dpi=300, facecolor="#ffffff")
gs = GridSpec(3, 6, height_ratios=[1.25, 0.95, 1.05], wspace=0.25, hspace=0.48,
              left=0.05, right=0.98, top=0.91, bottom=0.06)

fig.suptitle("Figure 1: Unsupervised Corticostriatal Sequences Discovered Blind to Behavior",
             fontsize=15.0, fontweight="bold", y=0.97, color="#0f172a")

# Behavioral Landmarks along closed operant loop (P2 -> next P2) normalized phase theta in [0, 1]
port_landmarks = [
    (0.000, "P2"),
    (0.126, "P1"),
    (0.275, "P6"),
    (0.427, "P3"),
    (0.572, "Rew"),
    (0.849, "P7ex"),
    (1.000, "Next P2")
]

# Normalized Phase Grid for Row B: 0.0 to 1.0 (100 bins)
theta_grid = np.linspace(0.0, 1.0, 100)

# Render Columns
for col_idx, meta in enumerate(motif_meta):
    m = meta["id"]
    print(f"Rendering Column {col_idx+1}: Motif {m}...")
    
    # ──────────────────────────────────────────────────────────────────────────
    # ROW A: 2D PHYSICAL SPACE (TRUE RATE MAP IN HZ OVER 6-PORT ARENA)
    # ──────────────────────────────────────────────────────────────────────────
    ax_top = fig.add_subplot(gs[0, col_idx])
    render_schema_panel(ax_top, corridor_rate_hz[m], dwell_rate_hz[m],
                        meta["name"], meta["sub"], meta["cmap"], meta["vmax"])
    
    if col_idx == 0:
        # Generous top padding to prevent title collision
        ax_top.text(-0.16, 1.35, "A   Physical Space: True 2D Spatial Rate Map (Activity / Occupancy Time, Hz)",
                    transform=ax_top.transAxes, fontsize=12.0, fontweight="bold", color="#0f172a")

    # ──────────────────────────────────────────────────────────────────────────
    # ROW B: BEHAVIOR-ALIGNED CONDITIONAL SPIKING DENSITY (HZ / UNIT, RESTRICTED TO ACTIVE TRIALS)
    # ──────────────────────────────────────────────────────────────────────────
    ax_mid = fig.add_subplot(gs[1, col_idx])
    
    chs = motifs_data[f"m{m}_channels"] - 1
    mus = motifs_data[f"m{m}_mus"]
    vars_ = motifs_data[f"m{m}_vars"]
    masses = motifs_data[f"m{m}_masses"]
    mop_mask = is_mop[chs]
    cp_mask  = is_cp[chs]
    
    # Restrict to active cycles (>= 1 occurrence in [-0.2, t_end + 0.2] s)
    m_taus = ev_ts[ev_motif == m]
    active_cycles = []
    for _, cyc in df_cycles.iterrows():
        t0, t1 = cyc["t0"], cyc["t1"]
        occs = m_taus[(m_taus >= t0 - 0.2) & (m_taus <= t1 + 0.2)]
        if len(occs) > 0:
            active_cycles.append({"t0": t0, "t1": t1, "dur": cyc["dur"], "occs_rel": occs - t0})
            
    pct_active = (len(active_cycles) / N_CYCLES) * 100.0
    
    # Compute conditional total population spiking density across normalized cycle phase theta in [0, 1]
    rate_tot_b = np.zeros_like(theta_grid)
    n_total = n_mop + n_cp
    
    for cyc in active_cycles:
        dur = cyc["dur"]
        t_eval = theta_grid * dur
        cyc_rate = np.zeros_like(theta_grid)
        for occ_t in cyc["occs_rel"]:
            for mu, v, mass in zip(mus, vars_, masses):
                sd = np.sqrt(v)
                cyc_rate += (mass / (n_total * sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((t_eval - (occ_t + mu)) / sd)**2)
        rate_tot_b += cyc_rate
        
    rate_tot_b /= len(active_cycles)
    
    # Plot single total conditional spiking density curve using motif's theme color
    ax_mid.fill_between(theta_grid, 0, rate_tot_b, color=meta["fill_col"], alpha=0.55)
    ax_mid.plot(theta_grid, rate_tot_b, color=meta["line_col"], linewidth=2.2)
    
    # Mark behavioral landmarks
    for pt, plbl in port_landmarks:
        ax_mid.axvline(pt, color="#94a3b8", linestyle=":", linewidth=0.8, zorder=1)
        
    ax_mid.set_xlim(0.0, 1.0)
    ax_mid.set_xticks([0.000, 0.126, 0.275, 0.427, 0.572, 0.849, 1.000])
    ax_mid.set_xticklabels(["P2", "P1", "P6", "P3", "Rew", "P7ex", "Next P2"], fontsize=6.8, fontweight="bold")
    ax_mid.set_xlabel("Normalized Cycle Progression (0 to 1, P2 -> Next P2)", fontsize=8.2, fontweight="bold", labelpad=2)
    
    y_max_b = max(rate_tot_b.max() * 1.25, 0.20)
    ax_mid.set_ylim(0, y_max_b)
    
    ax_mid.set_title(f"Active: {pct_active:.1f}% of cycles (N = {len(active_cycles)})",
                     fontsize=8.5, fontweight="bold", color="#1e293b", pad=5)
    
    if col_idx == 0:
        ax_mid.set_ylabel("Total Spiking Density\n(Hz / unit)", fontsize=8.8, fontweight="bold")
        ax_mid.text(-0.16, 1.22, "B   Cycle Progression: Total Spiking Density along Normalized Operant Loop (0 to 1, Hz / unit)",
                    transform=ax_mid.transAxes, fontsize=12.0, fontweight="bold", color="#0f172a")

    # ──────────────────────────────────────────────────────────────────────────
    # ROW C: MOTIF-ALIGNED INTRINSIC SPIKING DENSITY (TAU = 0, HZ / UNIT)
    # ──────────────────────────────────────────────────────────────────────────
    ax_bot = fig.add_subplot(gs[2, col_idx])
    
    t_span = meta["t_span"]
    t_grid_c = np.linspace(t_span[0], t_span[1], 400)
    
    rate_mop_c = np.zeros_like(t_grid_c)
    for mu, v, mass in zip(mus[mop_mask], vars_[mop_mask], masses[mop_mask]):
        sd = np.sqrt(v)
        rate_mop_c += (mass / (n_mop * sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((t_grid_c - mu) / sd)**2)
        
    rate_cp_c = np.zeros_like(t_grid_c)
    for mu, v, mass in zip(mus[cp_mask], vars_[cp_mask], masses[cp_mask]):
        sd = np.sqrt(v)
        rate_cp_c += (mass / (n_cp * sd * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((t_grid_c - mu) / sd)**2)
        
    ax_bot.fill_between(t_grid_c * 1000.0, 0, rate_mop_c, color=col_mop, alpha=0.35, label="MOp Cortex")
    ax_bot.plot(t_grid_c * 1000.0, rate_mop_c, color=col_mop, linewidth=2.0)
    
    ax_bot.fill_between(t_grid_c * 1000.0, 0, rate_cp_c, color=col_cp, alpha=0.35, label="CP Striatum")
    ax_bot.plot(t_grid_c * 1000.0, rate_cp_c, color=col_cp, linewidth=2.0)
    
    ax_bot.axvline(0.0, color="#94a3b8", linestyle=":", linewidth=0.9, zorder=1)
    
    ax_bot.set_xlim(t_span[0] * 1000.0, t_span[1] * 1000.0)
    y_max_c = max(rate_mop_c.max(), rate_cp_c.max()) * 1.25
    ax_bot.set_ylim(0, max(y_max_c, 0.5))
    ax_bot.set_xlabel(r"Motif Time from Onset $\tau = 0$ (ms)", fontsize=8.2, fontweight="bold", labelpad=2)
    
    if col_idx == 0:
        ax_bot.set_ylabel("Spiking Density\n(Hz / unit)", fontsize=8.8, fontweight="bold")
        ax_bot.text(-0.16, 1.22, "C   Intrinsic Firing Envelopes: Corticostriatal Dynamics (tau = 0, Hz / unit)",
                    transform=ax_bot.transAxes, fontsize=12.0, fontweight="bold", color="#0f172a")
        ax_bot.legend(loc="upper right", fontsize=7.2, framealpha=0.90)
        
    span_ms = int((t_span[1] - t_span[0]) * 1000.0)
    ax_bot.set_title(f"Span: {span_ms} ms", fontsize=8.5, fontweight="bold", color="#475569", pad=5)

# Save Outputs
print(f"Saving publication-quality Figure 1 to: {OUT_PNG}")
fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
plt.close(fig)

print("=" * 80)
print("FIGURE 1 GENERATION COMPLETE.")
print("=" * 80)
