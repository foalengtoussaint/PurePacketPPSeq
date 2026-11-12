# run_awake_to_200.jl
# Continues on-task MCMC fitting from Sweep 51 to 200 (Total 200 Sweeps)
# Same exact settings: Dynamic Motif Birth & Occam Death fully enabled, all motifs plastic

include("/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/regime_triad_sampler/run_regime_sampler.jl")
using PurePacketPPSeq

data = load_awake_spikes()
ck_path = joinpath(@__DIR__, "checkpoint_pure_packet_scratch_50sweeps.jls")

# Resume model exactly at its converged 50-sweep state
model = resume_model_from_checkpoint(ck_path, data; R_max=4)

target_sweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 200
checkpoint = joinpath(@__DIR__, "checkpoint_pure_packet_scratch_$(target_sweeps)sweeps.jls")

println("Resuming fitting from Sweep 51 to $(target_sweeps) with full Motif Birth and Death...")
fit!(
    model, data;
    sweeps=target_sweeps,
    start_sweep=51,
    checkpoint_path=checkpoint,
    checkpoint_interval=10,
    allow_motif_birth=true,
    frozen_motif_ids=Set{Int}() # All motifs fully plastic with birth and death enabled
)
