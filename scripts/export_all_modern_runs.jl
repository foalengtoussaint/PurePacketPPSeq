using PurePacketPPSeq
using Serialization
using DelimitedFiles

include("export_run_for_explorer.jl")

data_dir = "/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/results/top4_task_20261008/data/lgcy0288__20230519_114115__ephys"
runs_dir = "/home/foaleng/Documents/PurePacketPPSeq/runs"

# Load region channels
units_file = joinpath(data_dir, "units.csv")
units_raw, _ = readdlm(units_file, Char(44), header=true)
raw_regions = String.(units_raw[:, 3])
mop_channels = Set(findall(r -> startswith(r, "MOp"), raw_regions))
cp_channels  = Set(findall(r -> !startswith(r, "MOp"), raw_regions))

println("="^80)
println("EXPORTING ALL MODERN ARCHITECTURE RUNS TO: $runs_dir")
println("="^80)

# 1. Full-Session 100-Sweep Checkpoint
p_100 = "/home/foaleng/Documents/PurePacketPPSeq/scripts/checkpoint_pure_packet_cold_start_100sweeps.jls"
if isfile(p_100)
    export_run(
        p_100, data_dir, joinpath(runs_dir, "lgcy0288_awake_100sweeps");
        run_id="lgcy0288_awake_100sweeps",
        run_name="Session lgcy0288 · Full-Session 100 Sweeps (411 Units, 20 Motifs)"
    )
end

# 2. Behavioral Correspondence Checkpoint (Native Port Alignment)
p_beh = "/home/foaleng/Documents/PurePacketPPSeq/scripts/checkpoint_behavioral_correspondence.jls"
if isfile(p_beh)
    export_run(
        p_beh, data_dir, joinpath(runs_dir, "lgcy0288_behavioral_aligned");
        run_id="lgcy0288_behavioral_aligned",
        run_name="Session lgcy0288 · Native Behavioral Port Alignment (426 Channels)"
    )
end

# 3. Model A (MOp only, 400-1200 s)
p_ma = "/home/foaleng/Documents/PurePacketPPSeq/scripts/checkpoint_leak_free_model_a_mop.jls"
if isfile(p_ma)
    export_run(
        p_ma, data_dir, joinpath(runs_dir, "exp1_leak_free_model_a_mop");
        run_id="exp1_leak_free_model_a_mop",
        run_name="Exp 1 · Model A (MOp-only 208 Units, 400–1200 s)",
        t_window=(400.0, 1200.0),
        channel_subset=mop_channels
    )
end

# 4. Model B (CP only, 400-1200 s)
p_mb = "/home/foaleng/Documents/PurePacketPPSeq/scripts/checkpoint_leak_free_model_b_cp.jls"
if isfile(p_mb)
    export_run(
        p_mb, data_dir, joinpath(runs_dir, "exp1_leak_free_model_b_cp");
        run_id="exp1_leak_free_model_b_cp",
        run_name="Exp 1 · Model B (CP-only 203 Units, 400–1200 s)",
        t_window=(400.0, 1200.0),
        channel_subset=cp_channels
    )
end

# 5. Model C (Joint MOp + CP, 400-1200 s)
p_mc = "/home/foaleng/Documents/PurePacketPPSeq/scripts/checkpoint_leak_free_model_c_joint.jls"
if isfile(p_mc)
    export_run(
        p_mc, data_dir, joinpath(runs_dir, "exp1_leak_free_model_c_joint");
        run_id="exp1_leak_free_model_c_joint",
        run_name="Exp 1 · Model C (Joint MOp + CP 411 Units, 400–1200 s)",
        t_window=(400.0, 1200.0)
    )
end

# 6. Model C Held-Out Test Evaluation (1200-1700 s)
p_te = "/home/foaleng/Documents/PurePacketPPSeq/scripts/checkpoint_leak_free_model_c_test.jls"
if isfile(p_te)
    export_run(
        p_te, data_dir, joinpath(runs_dir, "exp1_leak_free_held_out_test");
        run_id="exp1_leak_free_held_out_test",
        run_name="Exp 1 · Held-Out Test Evaluation (1200–1700 s, +17,027 nats)",
        t_window=(1200.0, 1700.0)
    )
end

println("="^80)
println("ALL MODERN ARCHITECTURE RUNS SUCCESSFULLY EXPORTED TO: $runs_dir")
println("="^80)
