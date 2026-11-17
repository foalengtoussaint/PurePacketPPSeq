# run_100sweeps_actual_full_session.jl
# True Cold-Start Discovery (100 Sweeps) on Actual Full Session (T = 13,833.14s / 3.84 hours)
# All 411 Pure Neural Units (208 MOp + 203 CP, 12,464,150 Spikes)

using PurePacketPPSeq
using NPZ
using Printf
using Serialization

println("="^80)
println("PUREPACKET PPSEQ: 100-SWEEP RUN — ACTUAL FULL SESSION (411 UNITS, 3.84 HRS)")
println("="^80)

spikes_path = "/home/foaleng/Downloads/mop_awake_ppseq/data/mop_basal_ganglia_all_clusters/spikes_full.npz"
raw = npzread(spikes_path)
ts = Float64.(raw["timestamp"])
ch = Int.(raw["neuron"])

data = SpikeDataset(ts, ch; N_channels=411)
println(@sprintf("Full Recording Domain Hygiene: %d Spikes strictly across %d Pure Neural Units | Duration T = %.2f s (%.2f hours)",
        length(data.t), data.N_channels, data.T_total, data.T_total / 3600.0))

# Initialize true cold-start model (M_init = 0, E_init = 0)
empty_motifs = Dict{Int, PurePacketMotif}()
model = initialize_scratch_model(data, empty_motifs; R_init=1)

N_sweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
checkpoint = joinpath(@__DIR__, "checkpoint_pure_packet_actual_full_session_$(N_sweeps)sweeps.jls")

t0 = time()
fit!(
    model, data;
    sweeps=N_sweeps,
    checkpoint_path=checkpoint,
    checkpoint_interval=5,
    allow_motif_birth=true,
    frozen_motif_ids=Set{Int}()
)
dt = time() - t0

println("="^80)
@printf("Full Session 100-Sweep Fit Completed in %.1f s (%.2f hours)\n", dt, dt / 3600.0)
@printf("Discovered Motifs: %d | Total Occurrences: %d\n", length(model.motifs), length(model.occurrences))
@printf("Final Checkpoint Saved: %s\n", checkpoint)
println("="^80)
