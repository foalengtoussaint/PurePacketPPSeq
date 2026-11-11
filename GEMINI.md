# Project Core Principles & Constraints

## 1. Generative Self-Sufficiency (No External Models)
- **Zero External Decoders:** Never use external statistical models, regression classifiers, linear readouts (e.g. ridge, LASSO, logistic regression), neural networks, or external HMM toolboxes to perform sequence decoding, behavioral prediction, or latent state inference.
- **Strictly Native Model Specification:** All scientific inference—including behavioral decoding, replay detection, cross-regional transfer, and trajectory tracking—must be formulated and executed strictly through **our own point-process model** (e.g., freezing the unsupervised neural motif backbone, augmenting the raster with behavioral channels as point-process events/pseudo-spikes, and reading out predictions via the generative model's own event occurrences and emission likelihood).
- **Metric Computation Exception:** External algorithms are permitted **only** to compute standard, objective validation metrics on existing outputs (e.g., Levenshtein string edit distance, Kolmogorov-Smirnov $D_{\text{KS}}$ statistics, time-rescaling tests, information bounds). They must never generate, transform, or decode the predictions themselves.

## 2. Interaction & Output Integrity
- **No Unprompted Markdown Artifacts (.md):** Do not generate or edit markdown report artifacts (`.md` files) in the artifact directory unless explicitly requested by the user. Deliver all math, explanations, code snippets, and benchmark tables directly in chat.
- **Markdown Table Formatting:** Ensure all markdown tables have strict column alignment and matching divider counts, avoiding unescaped pipe `|` characters inside LaTeX math or text cells.
