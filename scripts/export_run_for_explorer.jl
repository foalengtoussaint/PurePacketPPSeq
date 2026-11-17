using PurePacketPPSeq
using Serialization
using JSON
using Statistics
using LinearAlgebra
using NPZ
using DelimitedFiles
using Printf

function export_run(
    ckpt_path::String,
    data_dir::String,
    output_dir::String;
    run_id::String="run",
    run_name::String="PPSeq Run"
)
    mkpath(output_dir)
    println("="^75)
    println("Exporting Run for Model Explorer: $run_name")
    println("="^75)

    # 1. Load Checkpoint
    println("Loading checkpoint: $ckpt_path ...")
    ckpt = deserialize(ckpt_path)
    motifs = ckpt[:motifs]
    occs = ckpt[:occurrences]
    assign = ckpt[:assign]
    ou_ctx = ckpt[:ou_ctx]
    bg_rates = ckpt[:bg_rates]
    ard_state = ckpt[:ard_state]

    # 2. Load Unit & Region metadata
    units_file = joinpath(data_dir, "units.csv")
    units_raw, _ = readdlm(units_file, Char(44), header=true)
    raw_regions = String.(units_raw[:, 3])
    raw_depths = [coalesce(tryparse(Float64, string(x)), 0.0) for x in units_raw[:, 4]]
    N_units = length(raw_regions)

    mop_units = findall(r -> startswith(r, "MOp"), raw_regions)
    cp_units  = findall(r -> !startswith(r, "MOp"), raw_regions)

    # 3. Load Spikes
    raw_npz = npzread(joinpath(data_dir, "joint.npz"))
    ts_all = Float64.(raw_npz["timestamp"])
    ch_all = Int.(raw_npz["channel"])

    idx_valid = findall(i -> ch_all[i] <= N_units, 1:length(ts_all))
    ts_spk = ts_all[idx_valid]
    ch_spk = ch_all[idx_valid]
    T_total = maximum(ts_spk)

    println("Total Spikes: $(length(ts_spk)) across $N_units units (T = $(round(T_total, digits=2)) s)")

    # 4. Process Motifs
    # Palette of high-contrast colors
    palette = [
        "#e11d48", "#0284c7", "#0d9488", "#db2777", "#7c3aed",
        "#f59e0b", "#10b981", "#6366f1", "#ec4899", "#84cc16",
        "#14b8a6", "#f97316", "#8b5cf6", "#06b6d4", "#a855f7",
        "#ef4444", "#3b82f6", "#10b981", "#64748b", "#94a3b8"
    ]

    motifs_dict = Dict{String, Any}()
    for (m_idx, (m_id, m)) in enumerate(sort(collect(motifs), by=x->x[1]))
        P = length(m.packets)
        color = palette[mod1(m_idx, length(palette))]
        
        # Compute mean latency in MOp vs CP
        mop_mus = [p.mu for p in m.packets if p.channel in mop_units]
        cp_mus  = [p.mu for p in m.packets if p.channel in cp_units]
        mop_mean_mu = isempty(mop_mus) ? 0.0 : mean(mop_mus)
        cp_mean_mu  = isempty(cp_mus) ? 0.0 : mean(cp_mus)
        delta_lag_ms = (cp_mean_mu - mop_mean_mu) * 1000.0

        lead_dir = "Synchronous"
        if delta_lag_ms > 15.0
            lead_dir = "MOp leads CP"
        elseif delta_lag_ms < -15.0
            lead_dir = "CP leads MOp"
        end

        occ_count = count(o -> o.motif == m_id, values(occs))

        motifs_dict[string(m_id)] = Dict(
            "id" => m_id,
            "name" => "Motif $m_id",
            "color" => color,
            "n_packets" => P,
            "count" => occ_count,
            "delta_lag_ms" => round(delta_lag_ms, digits=1),
            "lead_direction" => lead_dir,
            "packets" => [
                Dict(
                    "channel" => p.channel,
                    "mu" => round(p.mu, digits=4),
                    "sigma" => round(sqrt(max(p.var, 1e-6)), digits=4),
                    "mass" => round(p.mass, digits=3)
                ) for p in m.packets
            ]
        )
    end

    # 5. Process Occurrences
    println("Processing $(length(occs)) occurrences...")
    events_list = []
    for occ in sort(collect(values(occs)), by=o->o.tau)
        m_id = occ.motif
        # Observed spikes assigned to this occurrence
        n_obs = sum(occ.counts)
        warp_mag = norm(occ.u)
        push!(events_list, Dict(
            "id" => occ.id,
            "motif" => m_id,
            "time" => round(occ.tau, digits=4),
            "amp" => round(occ.amplitude, digits=3),
            "warp" => round(warp_mag, digits=3),
            "u" => [round(x, digits=3) for x in occ.u],
            "obs" => n_obs
        ))
    end

    # 6. Sample Continuous Data Streams (OU Arousal at native grid)
    println("Sampling continuous streams (OU network arousal & drift)...")
    t_stream = ou_ctx.t_net
    B = min(length(t_stream), size(ou_ctx.x_region, 2))
    t_stream_sub = t_stream[1:B]
    x_mop_path = [round(ou_ctx.x_region[1, j], digits=4) for j in 1:B]
    x_cp_path  = [round(ou_ctx.x_region[2, j], digits=4) for j in 1:B]

    streams_dict = Dict(
        "time" => [round(t, digits=2) for t in t_stream_sub],
        "streams" => [
            Dict(
                "id" => "x_MOp",
                "label" => "MOp Network Arousal (τ = 0.5s)",
                "color" => "#388bfd",
                "unit" => "a.u.",
                "data" => x_mop_path
            ),
            Dict(
                "id" => "x_CP",
                "label" => "CP Network Arousal (τ = 0.5s)",
                "color" => "#f0883e",
                "unit" => "a.u.",
                "data" => x_cp_path
            )
        ]
    )

    # 7. Downsampled Overview Spikes (for instant browser render)
    # Full spike density LOD: bin into 100ms bins per channel for macroscopic overview
    println("Preparing multi-scale spike data...")
    # Save raw downsampled spikes (up to 150k representative spikes for standalone offline view)
    step_sub = max(1, Int(floor(length(ts_spk) / 120000)))
    sub_idx = 1:step_sub:length(ts_spk)
    
    spikes_overview = Dict(
        "t" => [round(t, digits=3) for t in ts_spk[sub_idx]],
        "c" => ch_spk[sub_idx],
        "total_count" => length(ts_spk)
    )

    # 8. Manifest Metadata
    manifest = Dict(
        "run_id" => run_id,
        "run_name" => run_name,
        "session" => basename(data_dir),
        "duration" => round(T_total, digits=2),
        "n_units" => N_units,
        "n_spikes" => length(ts_spk),
        "n_motifs" => length(motifs),
        "n_events" => length(occs),
        "regions" => Dict(
            "MOp" => Dict("name" => "Motor Cortex (MOp)", "count" => length(mop_units), "channels" => mop_units, "color" => "#388bfd"),
            "CP"  => Dict("name" => "Dorsal Striatum (CP)", "count" => length(cp_units),  "channels" => cp_units,  "color" => "#f0883e")
        ),
        "motifs" => motifs_dict,
        "active_streams" => ["x_MOp", "x_CP"]
    )

    # Write JSON files
    println("Saving bundle to $output_dir ...")
    open(joinpath(output_dir, "manifest.json"), "w") do f; JSON.print(f, manifest, 2); end
    open(joinpath(output_dir, "events.json"), "w") do f; JSON.print(f, events_list); end
    open(joinpath(output_dir, "streams.json"), "w") do f; JSON.print(f, streams_dict); end
    open(joinpath(output_dir, "spikes.json"), "w") do f; JSON.print(f, spikes_overview); end

    println("✓ Run successfully exported to: $output_dir")
    println("="^75)
end

# Export default run from 100-sweep checkpoint
ckpt_path = "/home/foaleng/Documents/PurePacketPPSeq/scripts/checkpoint_pure_packet_cold_start_100sweeps.jls"
data_dir  = "/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/results/top4_task_20261008/data/lgcy0288__20230519_114115__ephys"
out_dir   = "/home/foaleng/Documents/PurePacketPPSeq/runs/lgcy0288_awake_100sweeps"

export_run(ckpt_path, data_dir, out_dir;
    run_id="lgcy0288_awake_100sweeps",
    run_name="Session lgcy0288 · Cold-Start 100 Sweeps (411 Units, 20 Motifs)"
)
