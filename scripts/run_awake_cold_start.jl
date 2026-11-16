# run_awake_cold_start.jl
# True Cold-Start Discovery (M_init = 0, E_init = 0) on Pure-Neural N=411 Units (Awake Task: T=2114.28s)

include("/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/regime_triad_sampler/run_regime_sampler.jl")
using PurePacketPPSeq

# 1. Load domain-hygienic data
data = load_awake_spikes()

# 2. Initialize true cold-start model (M_init = 0, E_init = 0, 100% background spikes)
empty_motifs = Dict{Int, PurePacketMotif}()
model = initialize_scratch_model(data, empty_motifs; R_init=1)

# 3. Fit MCMC with automated diagnostics, KS monitoring & checkpoints
N_sweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 50
checkpoint = joinpath(@__DIR__, "checkpoint_pure_packet_cold_start_$(N_sweeps)sweeps.jls")
log_path = joinpath(@__DIR__, "run_awake_cold_start_$(N_sweeps)sweeps.log")

fit!(
    model, data;
    sweeps=N_sweeps,
    checkpoint_path=checkpoint,
    checkpoint_interval=5,
    allow_motif_birth=true,
    frozen_motif_ids=Set{Int}() # All motifs fully plastic with birth and death enabled
)
