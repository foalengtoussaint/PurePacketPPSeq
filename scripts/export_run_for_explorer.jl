using PurePacketPPSeq
using Serialization
using JSON
using Statistics
using LinearAlgebra
using NPZ
using DelimitedFiles
using Printf

"""
    export_run(ckpt_input, data_dir, output_dir; ...)

Exports a model checkpoint into a standardized browser bundle:
  output_dir/
    ├── manifest.json
    ├── events.json
    ├── streams.json
    └── spikes.json
"""
function export_run(
    ckpt_input::Union{String, NamedTuple, Dict},
    data_dir::String,
    output_dir::String;
    run_id::String="run",
    run_name::String="PPSeq Run",
    t_window::Union{Nothing, Tuple{Float64, Float64}}=nothing,
    channel_subset::Union{Nothing, Vector{Int}, Set{Int}}=nothing
)
    mkpath(output_dir)
    println("="^75)
    println("Exporting Run for Model Explorer: $run_name")
    println("="^75)

    # 1. Load Checkpoint
    ckpt = isa(ckpt_input, String) ? deserialize(ckpt_input) : ckpt_input
    motifs = ckpt[:motifs]
    occs = ckpt[:occurrences]
    assign = haskey(ckpt, :assign) ? ckpt[:assign] : nothing
    ou_ctx = haskey(ckpt, :ou_ctx) ? ckpt[:ou_ctx] : nothing
    beh_names = haskey(ckpt, :beh_names) ? ckpt[:beh_names] : nothing

    # 2. Load Unit & Region metadata
    units_file = joinpath(data_dir, "units.csv")
    units_raw, _ = readdlm(units_file, Char(44), header=true)
    raw_regions = String.(units_raw[:, 3])
    raw_depths = [coalesce(tryparse(Int, string(x)), 0) for x in units_raw[:, 7]]
    N_neural = length(raw_regions)
    N_units = N_neural
    if beh_names !== nothing
        N_units = max(N_neural, maximum(keys(beh_names)))
    end

    all_mop_idx = findall(r -> startswith(r, "MOp"), raw_regions)
    all_cp_idx  = findall(r -> !startswith(r, "MOp"), raw_regions)

    # Sort MOp units descending by cortical depth (superficial -> deep)
    mop_sorted = all_mop_idx[sortperm(raw_depths[all_mop_idx], rev=true)]
    # Sort CP units descending by striatal depth (dorsal -> ventral)
    cp_sorted  = all_cp_idx[sortperm(raw_depths[all_cp_idx], rev=true)]

    if channel_subset !== nothing
        mop_sorted = filter(c -> c in channel_subset, mop_sorted)
        cp_sorted  = filter(c -> c in channel_subset, cp_sorted)
    end

    # Build contiguous anatomical rank map:
    # Ranks 0 .. length(mop_sorted)-1 -> MOp
    # Ranks length(mop_sorted) .. length(mop_sorted)+length(cp_sorted)-1 -> CP
    # Ranks length(neural) .. N_units-1 -> Behavioral channels
    rank_map = Dict{Int, Int}()
    curr_rank = 0
    for c in mop_sorted
        rank_map[c] = curr_rank
        curr_rank += 1
    end
    divider_rank = curr_rank

    for c in cp_sorted
        rank_map[c] = curr_rank
        curr_rank += 1
    end
    neural_divider_rank = curr_rank

    if beh_names !== nothing
        for c in sort(collect(keys(beh_names)))
            rank_map[c] = curr_rank
            curr_rank += 1
        end
    end

    # 3. Load Spikes
    raw_npz = npzread(joinpath(data_dir, "joint.npz"))
    ts_all = Float64.(raw_npz["timestamp"])
    ch_all = Int.(raw_npz["channel"])

    if t_window !== nothing
        t0, t1 = t_window
        valid_idx = findall(i -> (t0 <= ts_all[i] <= t1) && (ch_all[i] <= N_units), 1:length(ts_all))
        if channel_subset !== nothing
            valid_idx = filter(i -> ch_all[i] in channel_subset, valid_idx)
        end
        ts_spk = ts_all[valid_idx] .- t0
        ch_spk = ch_all[valid_idx]
        T_total = t1 - t0
    else
        valid_idx = findall(i -> ch_all[i] <= N_units, 1:length(ts_all))
        if channel_subset !== nothing
            valid_idx = filter(i -> ch_all[i] in channel_subset, valid_idx)
        end
        ts_spk = ts_all[valid_idx]
        ch_spk = ch_all[valid_idx]
        T_total = isempty(ts_spk) ? 100.0 : maximum(ts_spk)
    end

    N_spk = length(ts_spk)
    println("Total Spikes: $N_spk across $curr_rank active mapped channels (T = $(round(T_total, digits=2)) s)")

    # Align assignments to spikes
    ev_spk = zeros(Int32, N_spk)
    m_spk  = zeros(Int32, N_spk)

    if assign !== nothing && length(assign) == length(ts_all)
        # Sliced or full from global index
        for (out_i, glob_i) in enumerate(valid_idx)
            a = assign[glob_i]
            if a.event_id > 0 && haskey(occs, a.event_id)
                ev_spk[out_i] = Int32(a.event_id)
                m_spk[out_i]  = Int32(occs[a.event_id].motif)
            end
        end
    elseif assign !== nothing && length(assign) == N_spk
        # Directly matching sliced spikes
        for out_i in 1:N_spk
            a = assign[out_i]
            if a.event_id > 0 && haskey(occs, a.event_id)
                ev_spk[out_i] = Int32(a.event_id)
                m_spk[out_i]  = Int32(occs[a.event_id].motif)
            end
        end
    end

    n_fg = count(>(0), ev_spk)
    println("Foreground Spikes Identified: $n_fg ($(round(100.0 * n_fg / max(1, N_spk), digits=1))%)")

    # 4. Process Motifs
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
        
        mop_mus = [p.mu for p in m.packets if p.channel in mop_sorted]
        cp_mus  = [p.mu for p in m.packets if p.channel in cp_sorted]
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
                    "rank" => get(rank_map, p.channel, 0),
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

    # 6. Sample Continuous Data Streams
    println("Sampling continuous streams...")
    active_stream_ids = String[]
    streams_list = []

    if ou_ctx !== nothing
        t_stream = ou_ctx.t_net
        if t_window !== nothing
            t0, t1 = t_window
            mask = findall(t -> t0 <= t <= t1, t_stream)
            if !isempty(mask)
                t_out = [round(t_stream[j] - t0, digits=2) for j in mask]
                x_mop = [round(ou_ctx.x_region[1, j], digits=4) for j in mask]
                x_cp  = [round(ou_ctx.x_region[2, j], digits=4) for j in mask]
            else
                t_out = [0.0, T_total]
                x_mop = [0.0, 0.0]
                x_cp  = [0.0, 0.0]
            end
        else
            B = min(length(t_stream), size(ou_ctx.x_region, 2))
            t_out = [round(t, digits=2) for t in t_stream[1:B]]
            x_mop = [round(ou_ctx.x_region[1, j], digits=4) for j in 1:B]
            x_cp  = [round(ou_ctx.x_region[2, j], digits=4) for j in 1:B]
        end

        push!(streams_list, Dict(
            "id" => "x_MOp",
            "label" => "MOp Network Arousal (τ = 0.5s)",
            "color" => "#388bfd",
            "unit" => "a.u.",
            "data" => x_mop
        ))
        push!(streams_list, Dict(
            "id" => "x_CP",
            "label" => "CP Network Arousal (τ = 0.5s)",
            "color" => "#f0883e",
            "unit" => "a.u.",
            "data" => x_cp
        ))
        active_stream_ids = ["x_MOp", "x_CP"]
        streams_dict = Dict("time" => t_out, "streams" => streams_list)
    else
        streams_dict = Dict("time" => [0.0, T_total], "streams" => [])
    end

    # 7. Multi-scale overview: Keep 100% of foreground sequence spikes, subsample Poisson background
    println("Preparing multi-scale raster with foreground preservation...")
    fg_indices = findall(>(0), ev_spk)
    bg_indices = findall(==(0), ev_spk)

    target_bg = max(10000, 120000 - length(fg_indices))
    bg_step = max(1, Int(floor(length(bg_indices) / target_bg)))
    selected_bg = bg_indices[1:bg_step:end]

    combined_indices = sort(vcat(fg_indices, selected_bg))

    ts_sub = [round(ts_spk[i], digits=4) for i in combined_indices]
    ch_sub = ch_spk[combined_indices]
    rank_sub = [get(rank_map, c, 0) for c in ch_sub]
    ev_sub = ev_spk[combined_indices]
    m_sub  = m_spk[combined_indices]

    spikes_overview = Dict(
        "t" => ts_sub,
        "c" => ch_sub,
        "r" => rank_sub,
        "ev" => ev_sub,
        "m" => m_sub,
        "total_count" => N_spk,
        "fg_count" => n_fg
    )

    # 8. Manifest Metadata
    regions_dict = Dict{String, Any}(
        "MOp" => Dict(
            "name" => "Motor Cortex (MOp)",
            "count" => length(mop_sorted),
            "channels" => mop_sorted,
            "rank_start" => 0,
            "rank_end" => divider_rank - 1,
            "color" => "#388bfd"
        ),
        "CP"  => Dict(
            "name" => "Dorsal Striatum (CP)",
            "count" => length(cp_sorted),
            "channels" => cp_sorted,
            "rank_start" => divider_rank,
            "rank_end" => neural_divider_rank - 1,
            "color" => "#f0883e"
        )
    )
    if beh_names !== nothing
        beh_chans = sort(collect(keys(beh_names)))
        regions_dict["Behavior"] = Dict(
            "name" => "Task Pokes & Reward",
            "count" => length(beh_chans),
            "channels" => beh_chans,
            "rank_start" => neural_divider_rank,
            "rank_end" => curr_rank - 1,
            "color" => "#10b981",
            "labels" => [beh_names[c] for c in beh_chans]
        )
    end

    manifest = Dict(
        "run_id" => run_id,
        "run_name" => run_name,
        "session" => basename(data_dir),
        "duration" => round(T_total, digits=2),
        "n_units" => curr_rank,
        "n_spikes" => N_spk,
        "n_fg_spikes" => n_fg,
        "n_motifs" => length(motifs),
        "n_events" => length(occs),
        "divider_rank" => divider_rank,
        "neural_divider_rank" => neural_divider_rank,
        "rank_map" => [get(rank_map, c, 0) for c in 1:N_units],
        "regions" => regions_dict,
        "motifs" => motifs_dict,
        "active_streams" => active_stream_ids
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
