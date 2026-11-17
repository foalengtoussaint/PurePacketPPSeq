# run_100sweeps_cp_only.jl
# True Cold-Start Discovery (100 Sweeps) on CP-Only Units (Awake Task: T = 2114.28s)
# 203 CP Neural Units (channels 99..301)

using PurePacketPPSeq
using NPZ
using DelimitedFiles
using Printf
using Serialization

println("="^80)
println("PUREPACKET PPSEQ: 100-SWEEP RUN — CP-ONLY (203 UNITS)")
println("="^80)

data_dir = "/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/results/top4_task_20261008/data/lgcy0288__20230519_114115__ephys"
units_raw, _ = readdlm(joinpath(data_dir, "units.csv"), Char(44), header=true)
raw_regions = String.(units_raw[:, 3])
cp_channels = Set(findall(r -> !startswith(r, "MOp"), raw_regions))

raw_npz = npzread(joinpath(data_dir, "joint.npz"))
ts_all = Float64.(raw_npz["timestamp"])
ch_all = Int.(raw_npz["channel"])

# Filter spikes strictly for CP units
idx_cp = findall(i -> ch_all[i] in cp_channels, 1:length(ts_all))
ts_cp = ts_all[idx_cp]
ch_cp = ch_all[idx_cp]

T_total = 2114.2852511102637
data = SpikeDataset(ts_cp, ch_cp; N_channels=411, T_total=T_total)

println(@sprintf("Domain Hygiene: Strictly %d CP Neural Units | %d Spikes | Duration T = %.2f s",
        length(cp_channels), length(data.t), data.T_total))

# Initialize true cold-start model (M_init = 0, E_init = 0)
empty_motifs = Dict{Int, PurePacketMotif}()
model = initialize_scratch_model(data, empty_motifs; R_init=1)

N_sweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100
checkpoint = joinpath(@__DIR__, "checkpoint_pure_packet_cp_only_$(N_sweeps)sweeps.jls")

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
@printf("CP-Only 100-Sweep Fit Completed in %.1f s (%.2f min)\n", dt, dt / 60.0)
@printf("Discovered Motifs: %d | Total Occurrences: %d\n", length(model.motifs), length(model.occurrences))
@printf("Final Checkpoint Saved: %s\n", checkpoint)
println("="^80)
