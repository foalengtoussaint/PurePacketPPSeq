# PurePacketPPSeq.jl

**Exact, Continuous-Time Marked Point-Process Modeling for Unsupervised Neural Sequence Discovery**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Julia: 1.10+](https://img.shields.io/badge/Julia-1.10+-blue.svg)](https://julialang.org)

`PurePacketPPSeq.jl` is a high-performance Julia implementation of a continuous-time marked point-process framework for discovering coordinated, multi-neuronal spatiotemporal firing motifs from large-scale extracellular recordings (e.g., Neuropixels) **without temporal binning, template matching, or external statistical readouts**.

Traditional sequence detection algorithms either discretize continuous spike trains into arbitrary time bins (destroying millisecond-scale latency jitter and warping information) or rely on external regression classifiers and neural decoders that overfit and lack generative consistency. `PurePacketPPSeq.jl` models each individual spike as an event generated from a continuous intensity function governed by latent sequence occurrences, multi-timescale background excitability, motor vigor, and heavy-tailed continuous temporal warping.

---

## 1. Generative Point-Process Formulation

Let $c \in \{1, \dots, N\}$ denote the recorded neural units, and let $\mathcal{S} = \{(t_i, c_i)\}_{i=1}^{N_{\mathrm{spikes}}}$ denote the continuous event stream over session duration $T$. The instantaneous conditional intensity function for unit $c$ at time $t$ is defined as:

$$\lambda_c(t) = \lambda_c^{(0)}(t) + \sum_{e=1}^E c_e \sum_{p \in \mathcal{P}_{m_e}: \mathrm{chan}(p)=c} a_p \, \mathcal{N}\left(t \;\middle|\; \tau_e + \mu_p + \sum_{r=1}^R W_{p,r} u_{e,r}, \; \sigma_p^2\right)$$

where:
- $\lambda_c^{(0)}(t)$ is the multi-scale, continuous-time background intensity.
- $m_e \in \{1, \dots, M\}$ is the discrete motif archetype identity of occurrence $e$.
- $\tau_e \in [0, T]$ is the continuous onset time of occurrence $e$.
- $c_e > 0$ is the single-trial motor vigor amplitude scaling factor.
- $\mathbf{u}_e \in \mathbb{R}^R$ is the latent continuous temporal warping coordinate vector.
- $\mathcal{P}_m$ is the set of Gaussian spike packets comprising motif $m$.
- $a_p, \mu_p, \sigma_p^2$ are the packet amplitude mass, mean temporal latency, and timing variance on unit $\mathrm{chan}(p)$.
- $\mathbf{W} \in \mathbb{R}^{R \times |\mathcal{P}_m|}$ is the learned linear kinematic warping projection.

The exact point-process log-likelihood over the entire recording is evaluated analytically without numerical integration:

$$\ln \mathcal{L}(\mathcal{S} \mid \boldsymbol{\Theta}) = \sum_{i=1}^{N_{\mathrm{spikes}}} \ln \lambda_{c_i}(t_i) - \sum_{c=1}^N \int_0^T \lambda_c(t) \, dt$$

The compensator integral for any Gaussian packet on channel $c$ evaluates in closed form via the error function $\operatorname{erf}(\cdot)$:

$$\int_{T_1}^{T_2} \mathcal{N}(t \mid \tilde{\mu}, \sigma^2) \, dt = \frac{1}{2} \left[ \operatorname{erf}\left(\frac{T_2 - \tilde{\mu}}{\sqrt{2\sigma^2}}\right) - \operatorname{erf}\left(\frac{T_1 - \tilde{\mu}}{\sqrt{2\sigma^2}}\right) \right]$$

---

## 2. The Level $m=5$ Conjugate Triad Architecture

Biological neural populations operate across widely separated temporal and behavioral scales. `PurePacketPPSeq.jl` integrates three conjugate structural mechanisms to isolate reproducible millisecond sequence structure from non-stationary background noise:

```
                      +---------------------------------------+
                      |         Level m=5 Conjugate Triad      |
                      +---------------------------------------+
                                          |
        +---------------------------------+---------------------------------+
        |                                 |                                 |
        v                                 v                                 v
+-----------------------+     +-----------------------+     +-----------------------+
|        Leg 1          |     |        Leg 2          |     |        Leg 3          |
| Multi-Scale Excitability |     |   Single-Trial Vigor   |     | Heavy-Tailed Warping  |
|                       |     |                       |     |                       |
| - Fast Regional OU    |     | - Gamma(alpha, alpha) |     | - Student-t (nu=4.0)  |
|   (tau_net = 0.5 s)   |     | - Mean E[c_e] = 1.0   |     | - ARD alpha_r Pruning |
| - Slow Cellular Drift |     | - Decoupled from rate |     | - Closed-form linear  |
|   (tau_cell = 60 s)   |     |   and tempo           |     |   temporal shifts     |
+-----------------------+     +-----------------------+     +-----------------------+
```

### Leg 1: Multi-Scale Background Filtering
Unstructured multi-unit co-fluctuations and multi-minute recording drift frequently fool standard sequence detectors into hallucinating motifs. `PurePacketPPSeq.jl` decomposes baseline excitability into two continuous Ornstein-Uhlenbeck (OU) processes:

$$\ln \lambda_c^{(0)}(t) = \ln \bar{\lambda}_{0,c} + \mu_{r(c)} + x_{r(c)}(t) + \delta x_c(t)$$

- **Fast Regional Arousal $x_r(t)$ ($\tau_{\mathrm{net}} \approx 0.5\text{ s}$):** Absorbs sub-second waves of regional gain (e.g. MOp vs. CP, $\rho \approx +0.162$), preventing brief, unstructured rate surges from forming false sequence occurrences.
- **Slow Cellular Drift $\delta x_c(t)$ ($\tau_{\mathrm{cell}} \approx 60\text{ s}$):** Tracks unit-specific multi-minute excitability non-stationarity across long behavioral sessions.
- **Gauge Invariance:** Zero-mean constraints are strictly enforced via projection operators to guarantee statistical identifiability between background gain and motif amplitudes.

### Leg 2: Single-Trial Motor Vigor Pinning
Burst intensities naturally vary with animal engagement, arousal, and fatigue. Rather than allocating redundant motifs for high-rate vs. low-rate executions, each occurrence carries a latent vigor scalar:

$$c_e \sim \operatorname{Gamma}(\alpha_c, \beta_c), \quad \text{with } \beta_c = \alpha_c \implies \mathbb{E}[c_e] = 1.0$$

The shape parameter $\alpha_c$ is updated dynamically across MCMC sweeps. This decouples trial-by-trial burst vigor from internal sequence tempo.

### Leg 3: Heavy-Tailed Student-$t$ Continuous Warping & ARD
Motor sequences compress and expand non-linearly across trials. Standard Gaussian priors are fragile to behavioral pauses, outliers, and heavy-tailed timing shifts. `PurePacketPPSeq.jl` formulates continuous warping as a scale mixture:

$$\mathbf{u}_e \mid \tau_e^{\mathrm{prec}} \sim \mathcal{N}\left(\mathbf{0}, \; \left(\tau_e^{\mathrm{prec}} \operatorname{diag}(\boldsymbol{\alpha})\right)^{-1}\right), \quad \tau_e^{\mathrm{prec}} \sim \operatorname{Gamma}\left(\frac{\nu}{2}, \frac{\nu}{2}\right) \implies \mathbf{u}_e \sim \mathcal{T}_\nu(\mathbf{0}, \boldsymbol{\alpha}^{-1})$$

- Heavy-tailed degrees of freedom ($\nu = 4.0$) prevent timing outliers from destabilizing motif archetypes.
- **Automatic Relevance Determination (ARD):** Hyper-priors on $\alpha_r \sim \operatorname{Gamma}(a_\alpha, b_\alpha)$ drive precision $\alpha_r \to \infty$ for uninformative latent dimensions, automatically pruning inactive warping degrees of freedom without manual tuning.

---

## 3. Reversible-Jump MCMC (RJMCMC) Engine

`PurePacketPPSeq.jl` implements trans-dimensional RJMCMC moves to infer the number of motifs $M$ and packets $P_m$ directly from the data:

1. **Motif Birth & Death:**
   - **Dimension-Matching Birth:** Proposes new motif archetypes seeded from unassigned high-density event clusters.
   - **Two-Part Minimum Description Length (MDL) Prior:** Every proposed motif must overcome a strict dictionary code-length penalty:
     $$\Delta \ln \pi(\Theta) = -\left(\lambda_{\mathrm{dict}} + k_{\mathrm{packet}} P_m + k_{\mathrm{channel}} \sum_{c} \mathbb{I}[c \in \mathcal{P}_m]\right)$$
     preventing runaway dictionary bloat and ensuring parsimonious sequence recovery.
2. **Packet Gibbs Resampling:**
   - Packet latencies $\mu_p$ and temporal variances $\sigma_p^2$ are updated conditionally via conjugate normal-inverse-gamma transitions.
   - Packet masses $a_p$ are sampled conditionally subject to a biophysical SNR threshold ($\text{SNR}_p > 1.0$).
3. **Continuous Warping M-Step:**
   - Ridge-regularized projection updates for $\mathbf{W}$ combined with analytical Gibbs conditional updates for latent coordinates $\mathbf{u}_e$.

---

## 4. Empirical Validation & Benchmarks

The framework was benchmarked on dual-structure Neuropixels recordings from awake mice performing an operant behavioral task (motor cortex $\text{MOp}$: $N = 208$ units; dorsal striatum $\text{CP}$: $N = 203$ units; $411$ units total; $2.48\text{M}$ events across $2,114.3\text{ s}$):

### A. Split-Half Out-of-Sample Cross-Validation
- Evaluated on chronologically separated out-of-sample data with frozen motif archetypes:
  $$\Delta LL_{\mathrm{CV}} = +4,556\text{ nats} \quad (+1.39\text{ nats / event})$$
- Demonstrates a **$4.0\times$ temporal concentration** above the shifted null floor ($-2,020\text{ nats}$).

### B. Continuous Kinematic Coupling
- Inferred latent warping coordinates $\mathbf{u}_e$ correlate with single-trial movement kinematics:
  $$\text{Mean } R^2_{\mathrm{kin}} = 41.4\% \quad (p = 1.0 \times 10^{-4}, \text{ predictive Student-}t)$$
- Tracks single-trial behavioral execution duration with $r = +0.57$.

### C. Leak-Free Joint vs. Independent Cross-Validation (Held-Out Data)
Models were trained cold ($M_{\mathrm{init}} = 0$) on $t \in [400, 1200\text{ s}]$ and scored on an unseen $500\text{ s}$ held-out block ($t \in [1200, 1700\text{ s}]$, $593,118$ test spikes):

$$\Delta LL_1 = LL_{\mathrm{joint}} - \left(LL_{\mathrm{MOp}} + LL_{\mathrm{CP}}\right) = \mathbf{+17,027.50\text{ nats}} \quad (\mathbf{+0.0414\text{ bits / spike}})$$

- Cortical held-out gain: $+4,877.62\text{ nats}$ ($+0.0191\text{ nats/spk}$)
- Striatal held-out gain: $+12,149.88\text{ nats}$ ($+0.0360\text{ nats/spk}$)
- Model C achieved higher test likelihood using **fewer total motifs** ($M=16$) than the two independent regional models combined ($M = 8 + 10 = 18$).

### D. Prospective Causal Lookahead Forecasting ($\Delta = 250\text{ ms}$)
Predicting target spikes in the forward horizon $[t_k, t_k + 250\text{ ms}]$ conditioned strictly on past source activity ($t < t_k$):
- $\text{MOp}(t < t_k) \to \text{CP}([t_k, t_k + 250\text{ ms}])$: **$+0.0276\text{ nats / spike}$**
- $\text{CP}(t < t_k) \to \text{MOp}([t_k, t_k + 250\text{ ms}])$: **$+0.0196\text{ nats / spike}$**
- Forward corticostriatal forecasting is **$1.41\times$ stronger** than reverse striatocortical feedback, providing quantitative confirmation of feedforward cortical drive matching the $+298.6\text{ ms}$ MOp onset lead.

---

## 5. Installation & Quickstart

### Prerequisites
- Julia 1.10 or higher.

### Installation
Clone the repository and instantiate the Julia environment:

```bash
git clone https://github.com/foalengtoussaint/PurePacketPPSeq.jl.git
cd PurePacketPPSeq.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Basic Usage

```julia
using PurePacketPPSeq

# 1. Load spike train data (timestamps in seconds, channel indices 1:N)
# timestamps: Vector{Float64}, channels: Vector{Int}
dataset = SpikeDataset(timestamps, channels; N_channels=411, T_total=2114.28)

# 2. Initialize model from complete cold start (0 initial motifs)
empty_motifs = Dict{Int, PurePacketMotif}()
model = initialize_scratch_model(dataset, empty_motifs; R_init=1)

# 3. Fit via Reversible-Jump MCMC
fit!(
    model, dataset;
    sweeps=50,
    allow_motif_birth=true,        # Enables RJMCMC archetype birth/death
    checkpoint_path="checkpoint.jls",
    checkpoint_interval=10
)

# 4. Inspect discovered motifs and occurrences
println("Discovered $(length(model.motifs)) motifs across $(length(model.occurrences)) occurrences.")
```

### Out-of-Sample Evaluation with Frozen Motifs

```julia
# Test-time occurrence inference with frozen motifs
test_model = initialize_scratch_model(
    test_dataset, deepcopy(model.motifs);
    bg_rates=model.bg_rates,
    ard_state=model.ard_state,
    R_max=length(model.ard_state.active_mask)
)

fit!(
    test_model, test_dataset;
    sweeps=15,
    allow_motif_birth=false,
    frozen_motif_ids=Set(keys(test_model.motifs))
)
```

---

## 6. Project Architecture

```
PurePacketPPSeq/
├── src/                          # Core mathematical engine (strictly domain-agnostic)
│   ├── PurePacketPPSeq.jl        # Module definition and public API exports
│   ├── types.jl                  # Structs (SpikeDataset, Packet, PurePacketMotif, PureOccurrence)
│   ├── sampler.jl                # MCMC proposal and Gibbs transition kernels
│   ├── inference.jl              # fit! orchestration, M-step, and checkpointing
│   ├── data_loader.jl            # Spike stream formatting utilities
│   ├── metrics.jl                # Kolmogorov-Smirnov D_KS and time-rescaling tests
│   ├── mechanisms/               # Level m=5 Conjugate Triad modules
│   │   ├── m1_packet.jl          # Biophysical SNR-truncated packet representation
│   │   ├── m2_vigor.jl           # Gamma motor vigor scaling
│   │   └── m4_manifold.jl        # Heavy-tailed Student-t continuous warping & ARD
│   └── scales/                   # Multi-scale background OU models
├── scripts/                      # Experiment-specific orchestration & paper figures
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

## 7. Citation

If you use `PurePacketPPSeq.jl` in your research, please cite:

```bibtex
@software{PurePacketPPSeq2026,
  author = {Foaleng, Toussaint},
  title = {PurePacketPPSeq.jl: Exact Continuous-Time Marked Point-Process Modeling for Unsupervised Neural Sequence Discovery},
  year = {2026},
  url = {https://github.com/foalengtoussaint/PurePacketPPSeq.jl}
}
```

---

## 8. License

This project is licensed under the [MIT License](LICENSE).
