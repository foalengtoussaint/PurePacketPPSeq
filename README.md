# PurePacketPPSeq.jl

**Exact, Continuous-Time Marked Point-Process Modeling for Unsupervised Neural Sequence Discovery**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia: 1.10+](https://img.shields.io/badge/Julia-1.10+-blue.svg)](https://julialang.org)

`PurePacketPPSeq.jl` is a high-performance Julia package implementing a continuous-time marked point-process framework for discovering coordinated, multi-neuronal spatiotemporal firing motifs from large-scale extracellular recordings (e.g., Neuropixels) **without temporal binning, template matching, or external statistical readouts**.

Instead of slicing continuous spike trains into discrete time bins or relying on external regression classifiers, `PurePacketPPSeq.jl` models each individual spike as an event generated from a continuous conditional intensity function governed by latent sequence occurrences, multi-scale background excitability, motor vigor, and heavy-tailed continuous temporal warping:

$$\lambda_c(t) = \lambda_c^{(0)}(t) + \sum_{e=1}^E c_e \sum_{p \in \mathcal{P}_{z_e}: \mathrm{chan}(p)=c} a_p \, \mathcal{N}\left(t \;\middle|\; \tau_e + \mu_p + \sum_{r=1}^R W_{p,r}^{(z_e)} u_{e,r}, \; \sigma_p^2\right)$$

All log-likelihood evaluations and continuous compensator integrals are computed in closed form using analytical error functions ($\operatorname{erf}$).

---

## Quickstart

### Installation

```bash
git clone https://github.com/foalengtoussaint/PurePacketPPSeq.git
cd PurePacketPPSeq
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Self-Contained Runnable Example

Generate a synthetic multi-neuron spike train with an embedded sequence motif and fit the model from complete cold start without external data:

```julia
using PurePacketPPSeq
using Random

# 1. Generate synthetic spike train (50 units, 100 seconds)
Random.seed!(42)
N_units = 50
T_total = 100.0

timestamps = Float64[]
channels = Int[]

# Background Poisson spikes (1.5 Hz per unit)
for c in 1:N_units
    t = rand() * 0.5
    while t < T_total
        push!(timestamps, t)
        push!(channels, c)
        t += -log(rand()) / 1.5
    end
end

# Embed 20 occurrences of a 500 ms sequential cascade across units 1 -> 25
for occ_time in range(5.0, 95.0, length=20)
    for (step, c) in enumerate(1:25)
        spike_time = occ_time + (step - 1) * 0.020 + randn() * 0.003
        push!(timestamps, spike_time)
        push!(channels, c)
    end
end

perm = sortperm(timestamps)
dataset = SpikeDataset(timestamps[perm], channels[perm]; N_channels=N_units, T_total=T_total)

# 2. Initialize from cold start (0 initial motifs)
model = initialize_scratch_model(dataset, Dict{Int, PurePacketMotif}(); R_init=1)

# 3. Fit via Reversible-Jump MCMC
fit!(model, dataset; sweeps=25, allow_motif_birth=true)

# 4. Out-of-sample evaluation on held-out data with frozen motifs
test_model = initialize_scratch_model(
    dataset, deepcopy(model.motifs);
    bg_rates=model.bg_rates, ard_state=model.ard_state, R_max=length(model.ard_state.active_mask)
)
fit!(test_model, dataset; sweeps=10, allow_motif_birth=false, frozen_motif_ids=Set(keys(test_model.motifs)))

println("Discovered $(length(model.motifs)) motifs across $(length(model.occurrences)) occurrences.")
```

---

## Generative Tier Architecture ($m = 0 \to 5$)

The package is organized into a compositional hierarchy of generative tiers ($m = 0 \to 5$), where each level introduces an explicit biophysical mechanism with closed-form conjugate updates:

| Tier | Generative Mechanism | Description | Source Module |
| :--- | :--- | :--- | :--- |
| **$m = 0$** | Multi-Scale Background Floor | Baseline firing rate $\bar{\lambda}_{0,c}$ and continuous excitability floor | `src/scales/`, `m0_floor.jl` |
| **$m = 1$** | Biophysical Spike Packets | Localized Gaussian packet archetypes $(a_p, \mu_p, \sigma_p^2)$ with $\text{SNR}_p > 1.0$ | `src/mechanisms/m1_packet.jl` |
| **$m = 2$** | Sequence Carrier & Occurrences | Continuous onset discovery $\tau_e$, dynamic bounding envelopes $[\delta_L, \delta_R]$, and competitive occurrence life cycles | `src/mechanisms/m2_carrier.jl` |
| **$m = 3$** | Motif Dictionary & Vigor Dynamics | Nonparametric birth/death ($M \to M \pm 1$), Two-Part MDL complexity tax, gauge symmetry, and motor vigor $c_e$ | `src/mechanisms/m3_motif.jl` |
| **$m = 4$** | Continuous Warping Manifold | Heavy-tailed Student-$t$ kinematics $\mathbf{u}_e$ with ARD precision pruning | `src/mechanisms/m4_manifold.jl` |
| **$m = 5$** | Fully Coupled Continuous Engine | Dual-timescale OU network arousal $x_r(t)$ and cellular micro-drift $\delta x_c(t)$ | `src/mechanisms/m5_ou.jl` |

### Core Mathematical Components

- **Multi-Scale Background Filtering ($m=0, 5$):**
  $$\ln \lambda_c^{(0)}(t) = \ln \bar{\lambda}_{0,c} + \mu_{r(c)} + x_{r(c)}(t) + \delta x_c(t)$$
  Fast regional network OU ($x_r, \tau_{\mathrm{net}} \approx 0.5\text{ s}$) absorbs sub-second arousal surges ($\rho_{\mathrm{MOp},\mathrm{CP}} = +0.162$), while slow cellular drift ($\delta x_c, \tau_{\mathrm{cell}} \approx 60\text{ s}$) tracks multi-minute baseline shifts. Exact gauge invariance is enforced via orthogonal projection.

- **Single-Trial Motor Vigor Scaling ($m=3$):**
  $$c_e \sim \operatorname{Gamma}(\alpha_c, \beta_c), \quad \text{with } \beta_c = \alpha_c \implies \mathbb{E}[c_e] = 1.0$$
  Decouples trial-by-trial burst intensity from sequence duration, preventing duplicate motif allocation for high-rate vs. low-rate trial executions.

- **Heavy-Tailed Continuous Warping & ARD ($m=4$):**
  $$\mathbf{u}_e \mid \eta_e \sim \mathcal{N}\left(\mathbf{0}, \; \left(\eta_e \operatorname{diag}(\boldsymbol{\alpha})\right)^{-1}\right), \quad \eta_e \sim \operatorname{Gamma}\left(\frac{\nu}{2}, \frac{\nu}{2}\right) \implies \mathbf{u}_e \sim \mathcal{T}_\nu(\mathbf{0}, \boldsymbol{\alpha}^{-1})$$
  Student-$t$ robustness ($\nu = 4.0$) prevents behavioral timing outliers from distorting motif archetypes, while Automatic Relevance Determination (ARD) hyper-priors on $\alpha_r$ prune inactive warping axes automatically.

---

## Empirical Benchmarks & Validation Suite

Validated on dual-structure Neuropixels recordings from awake mice performing an operant behavioral task ($\text{MOp}$: $N = 208$ units; $\text{CP}$: $N = 203$ units; $411$ units total; $2.48\text{M}$ spikes across $2,114.3\text{ s}$):

| Validation Benchmark | Evaluation Protocol | Primary Result | Key Finding |
| :--- | :--- | :--- | :--- |
| **Out-of-Sample Split-Half CV** | Chronologically separated test blocks with frozen motifs | $\mathbf{+4,556\text{ nats}}$ ($+1.39\text{ nats/event}$) | **$4.0\times$ temporal concentration** above shifted null floor ($-2,020\text{ nats}$). |
| **Continuous Kinematic Coupling** | Single-trial latent warping $\mathbf{u}_e$ vs. kinematics | $\mathbf{R^2_{\mathrm{kin}} = 41.4\%}$ ($p = 1.0 \times 10^{-4}$) | Latent warping tracks trial duration with $r = +0.57$. |
| **Leak-Free Joint vs. Independent CV** | Trained cold ($400\text{--}1200\text{ s}$), tested on unseen $500\text{ s}$ ($593\text{k}$ spks) | $\mathbf{+17,027.50\text{ nats}}$ ($+0.0414\text{ bits/spk}$) | Joint model explains unseen data better with **fewer motifs** ($M=16$ vs. $18$). |
| **Prospective Causal Lookahead** | Forecast target in $[t_k, t_k + 250\text{ ms}]$ from past source ($t < t_k$) | $\text{MOp}\to\text{CP}: \mathbf{+0.0276\text{ nats/spk}}$<br>$\text{CP}\to\text{MOp}: \mathbf{+0.0196\text{ nats/spk}}$ | **$1.41\times$ stronger forward drive**, confirming cortical temporal lead ($+298.6\text{ ms}$). |
| **Regional Temporal Ordering** | Cross-structure motif packet peak latencies | $\text{MOp}$ leads $\text{CP}$ by $\mathbf{+298.6\text{ ms}}$ | Stereotyped sequence: early cortical initiation followed by striatal burst. |

### Visualizations

<p align="center">
  <img src="scripts/fig1_unsupervised_motifs.png" width="48%" alt="Unsupervised Motifs" />
  <img src="scripts/fig3_continuous_warping.png" width="48%" alt="Continuous Warping" />
</p>
<p align="center">
  <em>Left: Unsupervised spatiotemporal motif discovery across 411 units. Right: Continuous latent kinematic warping manifold.</em>
</p>

---

## Interactive Model Explorer

PurePacketPPSeq includes a zero-dependency, GPU-accelerated 60-FPS browser-based Model Explorer for interactive inspection of point-process motif occurrences, continuous data streams ($x_r(t)$, behavioral variables), and multi-unit spike rasters.

```bash
# 1. Launch the explorer (opens http://localhost:8050)
./run_explorer.sh

# 2. Export any custom model run into the explorer format
julia --project=. scripts/export_run_for_explorer.jl
```

### Standardized Run Bundle (`runs/<run_id>/`)
Any model fit can be inspected by producing a standardized 4-file bundle:
* `manifest.json`: Session metadata, channel anatomical mappings, motif dictionary, and stream keys.
* `events.json`: Motif occurrences $(\tau_e, z_e, c_e, \mathbf{u}_e, \text{observed spikes})$.
* `streams.json`: Continuous time series (e.g. cortical/striatal OU excitability, velocity, LFP).
* `spikes.json`: Multi-unit spike raster overview.

*Offline Mode:* Users can also open any run directory directly from disk without starting the Python server by clicking **📂 Open Folder** in the explorer header.

---

## Project Architecture

```
PurePacketPPSeq/
├── src/                          # Core mathematical engine (domain-agnostic)
│   ├── PurePacketPPSeq.jl        # Module definition and public API exports
│   ├── types.jl                  # Structs (SpikeDataset, Packet, PurePacketMotif, PureOccurrence)
│   ├── sampler.jl                # MCMC proposal and Gibbs transition kernels
│   ├── inference.jl              # fit! orchestration, M-step, and checkpointing
│   ├── data_loader.jl            # Spike stream formatting utilities
│   ├── metrics.jl                # Kolmogorov-Smirnov D_KS and time-rescaling tests
│   ├── mechanisms/               # Generative tier modules (m = 0 -> 5)
│   │   ├── m0_floor.jl           # m=0: Continuous background rate floor
│   │   ├── m1_packet.jl          # m=1: Biophysical SNR-truncated packet representation
│   │   ├── m2_carrier.jl         # m=2: Sequence carrier & occurrence discovery
│   │   ├── m3_motif.jl           # m=3: Motif dictionary prior & motor vigor scaling
│   │   ├── m4_manifold.jl        # m=4: Heavy-tailed Student-t continuous warping & ARD
│   │   └── m5_ou.jl              # m=5: Fully coupled continuous engine
│   └── scales/                   # Multi-scale background OU models
├── explorer/                     # Web-based Model Explorer
│   ├── index.html                # High-performance Canvas/SVG interactive visualizer
│   └── server.py                 # Lightweight API server for dynamic run discovery
├── runs/                         # Standardized store for exported model runs
│   └── lgcy0288_awake_100sweeps/ # Demo 100-sweep run (411 units, 20 motifs, 8,770 events)
├── run_explorer.sh               # One-click launcher script
├── scripts/                      # Experiment orchestration & figure generation
│   ├── export_run_for_explorer.jl # Standardized run export script
│   ├── plot_fig1_unsupervised_motifs.py
│   ├── plot_figure2_behavioral_correspondence.jl
│   ├── plot_fig3_continuous_warping.jl
│   ├── plot_fig4_session_drift.jl
│   ├── run_experiment1_joint_vs_indep.jl
│   └── run_leak_free_experiments.jl
├── Project.toml                  # Julia environment dependencies
└── README.md                     # Package documentation
```

---

## Citation

```bibtex
@software{PurePacketPPSeq2026,
  author = {Foaleng, Toussaint},
  title = {PurePacketPPSeq.jl: Exact Continuous-Time Marked Point-Process Modeling for Unsupervised Neural Sequence Discovery},
  year = {2026},
  url = {https://github.com/foalengtoussaint/PurePacketPPSeq}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
