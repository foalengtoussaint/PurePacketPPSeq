# run_awake_cold_start_to_100.jl
# Continues true cold-start fitting from Sweep 51 to 100 (Total 100 Sweeps)
# All motifs fully plastic with birth and death enabled

include("/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/regime_triad_sampler/run_regime_sampler.jl")
using PurePacketPPSeq

data = load_awake_spikes()
ck_path_52 = joinpath(@__DIR__, "checkpoint_pure_packet_cold_start_52sweeps.jls")
ck_path_50 = joinpath(@__DIR__, "checkpoint_pure_packet_cold_start_50sweeps.jls")
ck_path = isfile(ck_path_52) ? ck_path_52 : ck_path_50
start_sw = isfile(ck_path_52) ? 53 : 51

# Resume model exactly at its converged cold-start state
model = resume_model_from_checkpoint(ck_path, data; R_max=4)

target_sweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
checkpoint = joinpath(@__DIR__, "checkpoint_pure_packet_cold_start_$(target_sweeps)sweeps.jls")

println("Resuming true cold-start fitting from Sweep $(start_sw) to $(target_sweeps) with full Motif Birth and Death...")
fit!(
    model, data;
    sweeps=target_sweeps,
    start_sweep=start_sw,
    checkpoint_path=checkpoint,
    checkpoint_interval=10,
    allow_motif_birth=true,
    frozen_motif_ids=Set{Int}()
)
