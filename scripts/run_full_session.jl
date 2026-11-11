# run_full_session.jl
# High-Performance Full Session Pipeline (3.84 hrs / 13,833s)
# Level m=0..5 Pure Packet Engine across Pre-Task, Awake Task, Off-Task & Sleep Replay

using PurePacketPPSeq

println("=================================================================")
println("Level m=0..5 Pure Packet Engine: FULL SESSION (13,833s / 3.84 hrs)")
println("Pure-Neural N=411 & Level m=5 Conjugate Triad Continuous Engine")
println("Spanning Pre-Task, Awake Task, Post-Task Off-Task & Sleep Replay")
println("=================================================================")

# 1. Load full session spikes & completed Phase 1 scratch motifs
data = load_full_session_spikes()
ck_path = isfile(joinpath(@__DIR__, "checkpoint_pure_packet_scratch_50sweeps.jls")) ?
    joinpath(@__DIR__, "checkpoint_pure_packet_scratch_50sweeps.jls") :
    (isfile(joinpath(@__DIR__, "..", "checkpoints", "checkpoint_pure_packet_scratch_50sweeps.jls")) ?
        joinpath(@__DIR__, "..", "checkpoints", "checkpoint_pure_packet_scratch_50sweeps.jls") :
        "/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/manifold_sampler/checkpoint_pure_packet_scratch_50sweeps.jls")

loaded = load_checkpoint_motifs(ck_path; N_channels=411)

# 2. Initialize model across full session using learned Phase 1 background rates & dynamic manifold
model = initialize_scratch_model(data, loaded.motifs; bg_rates=loaded.bg_rates, R_init=length(loaded.ard_state.active_mask))

# 3. Fit MCMC across full session
N_sweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 200
checkpoint = joinpath(@__DIR__, "checkpoint_pure_packet_full_session_$(N_sweeps)sweeps.jls")
fit!(model, data; sweeps=N_sweeps, checkpoint_path=checkpoint, checkpoint_interval=10)
