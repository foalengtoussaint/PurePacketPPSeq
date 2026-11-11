# run_awake_scratch.jl
# High-Performance Runner for Pure Packet Marked Point Process Engine
# Scratch Start (0 Initial Occurrences) on Pure-Neural N=411 Units (Awake Task: T=2114.28s)

# Load legacy struct definitions for initial seed dictionary checkpoint
include("/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/regime_triad_sampler/run_regime_sampler.jl")
using PurePacketPPSeq

# 1. Load domain-hygienic data and initial seed motifs
data = load_awake_spikes()
motifs = load_seed_motifs()

# 2. Initialize cold-scratch model (0 active occurrences)
model = initialize_scratch_model(data, motifs; R_max=4)

# 3. Fit MCMC with automated diagnostics, KS monitoring & checkpoints
N_sweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 50
checkpoint = joinpath(@__DIR__, "checkpoint_pure_packet_scratch_$(N_sweeps)sweeps.jls")
fit!(model, data; sweeps=N_sweeps, checkpoint_path=checkpoint)
