# test_behavioral_correspondence_setup.jl
# Loads joint.npz and verifies behavioral channel alignment

using NPZ, JSON, Printf

data_dir = "/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/results/top4_task_20261008/data/lgcy0288__20230519_114115__ephys"
meta = JSON.parsefile(joinpath(data_dir, "joint.json"))
raw = npzread(joinpath(data_dir, "joint.npz"))

ts = Float64.(raw["timestamp"])
ch = Int.(raw["channel"])

println("Loaded joint.npz:")
@printf("  Total events: %d\n", length(ts))
@printf("  Time range:   %.3f s to %.3f s\n", minimum(ts), maximum(ts))
@printf("  Channel range: %d to %d\n", minimum(ch), maximum(ch))

n_neural = count(<=(411), ch)
n_beh = count(>=(412), ch)
@printf("  Neural spikes (1..411):     %d\n", n_neural)
@printf("  Behavioral events (412..426): %d\n", n_beh)

println("\nBehavioral Event Breakdown by Channel:")
for c in 412:426
    c_idx = c - 411
    ch_info = meta["channels"][c_idx]
    cnt = count(==(c), ch)
    rate = cnt / meta["duration_s"]
    @printf("  Ch %3d: %-16s | %5d events | %6.4f Hz\n", c, ch_info["name"], cnt, rate)
end
