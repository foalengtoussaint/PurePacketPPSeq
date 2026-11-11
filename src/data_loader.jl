# data_loader.jl - Point-Process Data Ingestion & Domain Hygiene
#
# Generative Foundation:
# Strictly enforces pure-neural domain boundaries (C = 411 units: MOp 1..98, 302..411; CP 99..301).
# Completely filters out behavioral pseudo-channels and non-neural clicks.

using NPZ
using Serialization
using Printf

"""
    SpikeDataset

Encapsulates marked point-process event streams with domain hygiene:
- `t`: Timestamps in seconds
- `neuron`: Physical neuron index in 1:N_channels
- `N_channels`: Total physical units (strictly 411)
- `T_total`: Recording span in seconds
- `chan_to_spikes`: Inverted index mapping channel k -> list of spike indices
- `total_spikes_per_chan`: Precomputed spike counts per physical channel
"""
struct SpikeDataset
    t::Vector{Float64}
    neuron::Vector{Int}
    N_channels::Int
    T_total::Float64
    chan_to_spikes::Vector{Vector{Int}}
    total_spikes_per_chan::Vector{Int}
end

function SpikeDataset(t_raw::AbstractVector{Float64}, neuron_raw::AbstractVector{Int}; N_channels::Int=411, T_total::Union{Nothing, Float64}=nothing)
    # 1. Enforce strict pure-neural domain hygiene (1 <= c <= N_channels)
    valid_idx = findall(c -> 1 <= c <= N_channels, neuron_raw)
    t = Float64.(t_raw[valid_idx])
    neuron = Int.(neuron_raw[valid_idx])

    # 2. Enforce monotonic timestamp ordering
    perm = sortperm(t)
    t = t[perm]
    neuron = neuron[perm]

    # 3. Compute temporal boundary and inverted index
    T = T_total !== nothing ? T_total : (isempty(t) ? 0.0 : maximum(t))
    c2s = [Int[] for _ in 1:N_channels]
    for (idx, c) in enumerate(neuron)
        push!(c2s[c], idx)
    end
    spikes_per_chan = [length(c2s[c]) for c in 1:N_channels]

    return SpikeDataset(t, neuron, N_channels, T, c2s, spikes_per_chan)
end

"""
    load_awake_spikes(; ck_path=nothing, N_channels::Int=411) -> SpikeDataset

Loads on-task awake recording data (T = 2114.28s) across pure neural units.
"""
function load_awake_spikes(; ck_path::Union{Nothing, String}=nothing, N_channels::Int=411)::SpikeDataset
    default_ck = "/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/results/top4_task_20261008/mh_screening_experiment/regime_triad_reproduction_100s/checkpoint.jls"
    target_path = ck_path !== nothing ? ck_path : default_ck

    if !isfile(target_path)
        error("Awake spike checkpoint not found at: $(target_path)")
    end

    ck = open(deserialize, target_path)
    b = ck.state.base

    dataset = SpikeDataset(b.t, b.neuron; N_channels=N_channels, T_total=b.T)
    println(@sprintf("Domain Hygiene Enforced: Strictly N=%d pure neural units (MOp: 208, CP: 203)", N_channels))
    println(@sprintf("Spike stream stripped of behavioral channels: %d / %d spikes retained (%.2f%%)",
            length(dataset.t), length(b.t), (length(dataset.t) / length(b.t)) * 100.0))
    return dataset
end

"""
    load_full_session_spikes(; data_path=nothing, N_channels::Int=411) -> SpikeDataset

Loads full session recording data (13,833s / 3.84 hrs) across pure neural units.
"""
function load_full_session_spikes(; data_path::Union{Nothing, String}=nothing, N_channels::Int=411)::SpikeDataset
    default_path = "/home/foaleng/Downloads/mop_awake_ppseq/data/mop_basal_ganglia_all_clusters/spikes_full.npz"
    target_path = data_path !== nothing ? data_path : default_path

    if !isfile(target_path)
        # Fallback to local data symlink
        target_path = joinpath(@__DIR__, "..", "data", "mop_basal_ganglia_all_clusters", "spikes_full.npz")
    end
    if !isfile(target_path)
        error("Full session spike dataset not found at: $(target_path)")
    end

    raw = npzread(target_path)
    ts = Float64.(raw["timestamp"])
    ch = Int.(raw["neuron"])

    dataset = SpikeDataset(ts, ch; N_channels=N_channels)
    println(@sprintf("Full Recording Domain Hygiene: %d Spikes strictly across %d Pure Neural Units | Duration T = %.2f s (%.2f hours)",
            length(dataset.t), N_channels, dataset.T_total, dataset.T_total / 3600.0))
    return dataset
end

"""
    load_seed_motifs(; ck_path=nothing, N_channels::Int=411, R_max::Int=4) -> Dict{Int, PurePacketMotif}

Extracts initial carrier motifs from seed checkpoint, converting legacy formats into PurePacketMotifs
with origin anchoring mu_1 == 0 and pure neural channel bounds.
"""
function load_seed_motifs(; ck_path::Union{Nothing, String}=nothing, N_channels::Int=411, R_max::Int=4)::Dict{Int, PurePacketMotif}
    default_ck = "/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/results/top4_task_20261008/mh_screening_experiment/regime_triad_reproduction_100s/checkpoint.jls"
    target_path = ck_path !== nothing ? ck_path : default_ck

    ck = open(deserialize, target_path)
    b = ck.state.base

    motifs = Dict{Int, PurePacketMotif}()
    for (mid, m) in b.motifs
        active_chans = filter(k -> m.pi[k] >= 0.001, 1:N_channels)
        sort!(active_chans, by=k -> m.mu[k])
        orig_shift = !isempty(active_chans) ? m.mu[active_chans[1]] : 0.0

        pkts = Packet[
            Packet(
                p, k,
                m.mu[k] - orig_shift,
                m.var[k],
                max(m.pi[k] * (b.amp_a[b.channel_group[k]] / b.amp_b[b.channel_group[k]]), 0.01)
            )
            for (p, k) in enumerate(active_chans)
        ]
        motifs[mid] = PurePacketMotif(mid, pkts, R_max, N_channels)
    end
    return motifs
end

"""
    load_checkpoint_motifs(ck_path::String; N_channels::Int=411) -> (motifs, ard_state)

Loads motifs and ARD state directly from a native PurePacketPPSeq .jls checkpoint.
"""
function load_checkpoint_motifs(ck_path::String; N_channels::Int=411)
    if !isfile(ck_path)
        error("Checkpoint file not found: $(ck_path)")
    end
    ck_data = open(deserialize, ck_path)
    motifs = ck_data.motifs
    ard_state = ck_data.ard_state

    for pm in values(motifs)
        filter!(p -> p.channel <= N_channels, pm.packets)
        rebuild_inverted_index!(pm, N_channels)
    end
    return (motifs = motifs, ard_state = ard_state)
end
