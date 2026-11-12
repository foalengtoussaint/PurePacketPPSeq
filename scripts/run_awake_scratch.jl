# run_awake_scratch.jl
# High-Performance Runner for Pure Packet Marked Point Process Engine
# Scratch Start (0 Initial Occurrences) on Pure-Neural N=411 Units (Awake Task: T=2114.28s)

# Load legacy struct definitions for initial seed dictionary checkpoint
include("/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/regime_triad_sampler/run_regime_sampler.jl")
using PurePacketPPSeq

# 1. Load domain-hygienic data and initial seed motifs
data = load_awake_spikes()
motifs = load_seed_motifs()

# 2. Initialize cold-scratch model with memoryless ISI tail baseline (0 active occurrences)
model = initialize_scratch_model(data, motifs; R_init=1)

# 3. Fit MCMC with automated diagnostics, KS monitoring & checkpoints
# Enable Level m=3 Dynamic Motif Birth and Death from scratch
N_sweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 50
checkpoint = joinpath(@__DIR__, "checkpoint_pure_packet_scratch_$(N_sweeps)sweeps.jls")
fit!(
    model, data;
    sweeps=N_sweeps,
    checkpoint_path=checkpoint,
    checkpoint_interval=5,
    allow_motif_birth=true,
    frozen_motif_ids=Set{Int}() # All motifs fully plastic with birth and death enabled
)
