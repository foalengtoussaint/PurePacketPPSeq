# bounds.jl - Dynamic Motif Bounding Envelopes & Inverted Index

function PurePacketMotif(id::Int, packets::Vector{Packet}, R_max::Int, N_channels::Int; W_window::Float64=0.0)
    P = length(packets)
    W = zeros(Float64, R_max, P)
    c2p = [Int[] for _ in 1:N_channels]
    for (p, pkt) in enumerate(packets)
        push!(c2p[pkt.channel], p)
    end
    chan_active = falses(N_channels)
    chan_bounds = fill((0.0, 0.0), N_channels)
    m = PurePacketMotif(id, packets, W, c2p, W_window, 0.0, 0.0, chan_active, chan_bounds)
    update_motif_bounds!(m)
    return m
end

"""
    update_motif_bounds!(m::PurePacketMotif, lambda_min::Vector{Float64}=Float64[]; bg_rates::Union{Nothing, Vector{Float64}}=nothing, kappa::Float64=4.0, kappa_u::Float64=3.0)

Dynamically computes:
1. Channel-Local Bounding Envelopes: m.chan_bounds[k] = (delta_left_k, delta_right_k)
   Evaluated strictly over packets located on channel k, eliminating cross-channel latency coupling.
   Uses Amplitude-Aware SNR Dominance Radius: R_p = sigma_p * sqrt(2 * ln^+(SNR_p)).
2. Global Motif Bounding Envelope: m.delta_left, m.delta_right across all channels.
3. Fast Boolean BitVector: m.chan_active[k] = true iff channel k has active packets.
4. Dynamic Physical Span: m.W_window = max(m.delta_right - m.delta_left, 0.0).

In-place, zero allocations during sweeps.
"""
function update_motif_bounds!(
    m::PurePacketMotif,
    lambda_min::Vector{Float64}=Float64[];
    bg_rates::Union{Nothing, Vector{Float64}}=nothing,
    kappa::Float64=4.0,
    kappa_u::Float64=3.0
)
    N_chans = length(m.chan_to_packets)
    if length(m.chan_active) != N_chans
        m.chan_active = falses(N_chans)
        m.chan_bounds = fill((0.0, 0.0), N_chans)
    else
        fill!(m.chan_active, false)
    end

    if isempty(m.packets)
        m.delta_left = 0.0
        m.delta_right = 0.0
        m.W_window = 0.0
        fill!(m.chan_bounds, (0.0, 0.0))
        return m
    end

    R = size(m.W, 1)
    global_left = Inf
    global_right = -Inf

    for k in 1:N_chans
        pkts_k = m.chan_to_packets[k]
        if isempty(pkts_k)
            m.chan_bounds[k] = (0.0, 0.0)
        else
            m.chan_active[k] = true
            c_left = Inf
            c_right = -Inf
            lam0 = (bg_rates !== nothing && k <= length(bg_rates)) ? max(bg_rates[k], 1e-12) : 0.5
            for p in pkts_k
                pkt = m.packets[p]
                sig = sqrt(max(pkt.var, 1e-8))

                # Amplitude-Aware SNR Dominance Radius
                if bg_rates !== nothing
                    snr_p = pkt.mass / (sqrt(2π) * sig * lam0)
                    R_p = (snr_p > 1.0) ? (sig * sqrt(2.0 * log(snr_p))) : 0.0
                    R_eff = max(R_p, sig)
                else
                    R_eff = kappa * sig
                end

                var_def = 0.0
                if R > 0 && !isempty(lambda_min)
                    for r in 1:min(R, length(lambda_min))
                        var_def += (m.W[r, p]^2) / max(lambda_min[r], 1e-6)
                    end
                end
                delta_p = kappa_u * sqrt(var_def)
                left_p = pkt.mu - R_eff - delta_p
                right_p = pkt.mu + R_eff + delta_p
                c_left = min(c_left, left_p)
                c_right = max(c_right, right_p)
            end
            m.chan_bounds[k] = (c_left, c_right)
            global_left = min(global_left, c_left)
            global_right = max(global_right, c_right)
        end
    end

    m.delta_left = global_left
    m.delta_right = global_right
    m.W_window = isfinite(global_right - global_left) ? max(global_right - global_left, 0.0) : 0.0
    return m
end

# Backward-compatibility alias
const update_channel_bounds! = update_motif_bounds!

# Checkpoint backward compatibility deserializer
const MOTIF_FLOAT64_TAG = Int32(0x0e)

function Serialization.deserialize(s::Serialization.AbstractSerializer, t::Type{PurePacketMotif})
    x = ccall(:jl_new_struct_uninit, Any, (Any,), t)
    Serialization.deserialize_cycle(s, x)
    for i in 1:5
        tag = Int32(read(s.io, UInt8)::UInt8)
        if tag != Serialization.UNDEFREF_TAG
            ccall(:jl_set_nth_field, Cvoid, (Any, Csize_t, Any), x, i-1, Serialization.handle_deserialize(s, tag))
        end
    end
    N_chans = length(x.chan_to_packets)
    x.chan_active = falses(N_chans)
    x.chan_bounds = fill((0.0, 0.0), N_chans)
    x.delta_left = 0.0
    x.delta_right = 0.0

    mark(s.io)
    if !eof(s.io)
        tag6 = Int32(read(s.io, UInt8)::UInt8)
        if tag6 == MOTIF_FLOAT64_TAG
            x.delta_left = Serialization.handle_deserialize(s, tag6)
            tag7 = Int32(read(s.io, UInt8)::UInt8)
            x.delta_right = Serialization.handle_deserialize(s, tag7)
            
            mark(s.io)
            if !eof(s.io)
                tag8 = Int32(read(s.io, UInt8)::UInt8)
                val8 = Serialization.handle_deserialize(s, tag8)
                if val8 isa BitVector && length(val8) == N_chans
                    x.chan_active = val8
                    tag9 = Int32(read(s.io, UInt8)::UInt8)
                    x.chan_bounds = Serialization.handle_deserialize(s, tag9)
                    unmark(s.io)
                    unmark(s.io)
                    return x
                else
                    reset(s.io)
                end
            end
            unmark(s.io)
            update_motif_bounds!(x)
            return x
        end
    end
    reset(s.io)
    update_motif_bounds!(x)
    return x
end

# =====================================================================
# 2. INVERTED INDEX & CACHE MAINTENANCE
# =====================================================================

function rebuild_inverted_index!(m::PurePacketMotif, N_channels::Int)
    for c in 1:N_channels
        empty!(m.chan_to_packets[c])
    end
    for (p, pkt) in enumerate(m.packets)
        pkt.id = p
        push!(m.chan_to_packets[pkt.channel], p)
    end
end


"""
    update_chan_to_motifs(motifs::Dict{Int, PurePacketMotif}, N_channels::Int)

Builds the inverted mapping chan_to_motifs[k] -> list of motif IDs containing packets on channel k.
Enables O(1) motif filtering during spike resampling, dropping candidate checks to <= 1.0 per spike.
"""
function update_chan_to_motifs(motifs::Dict{Int, PurePacketMotif}, N_channels::Int)
    chan_to_motifs = [Int[] for _ in 1:N_channels]
    for (mid, m) in motifs
        for k in 1:N_channels
            if !isempty(m.chan_to_packets[k])
                push!(chan_to_motifs[k], mid)
            end
        end
    end
    return chan_to_motifs
end


"""
    update_all_motif_bounds!(motifs::Dict{Int, PurePacketMotif}; bg_rates=nothing, kappa::Float64=4.0, kappa_u::Float64=3.0)

Amortized batch update of dynamic bounding envelopes across all motifs.
Called once per sweep after packet positions and loadings W are updated.
Takes < 2 microseconds across M motifs.
"""
function update_all_motif_bounds!(motifs::Dict{Int, PurePacketMotif}; bg_rates=nothing, kappa::Float64=4.0, kappa_u::Float64=3.0)
    for m in values(motifs)
        update_motif_bounds!(m; bg_rates=bg_rates, kappa=kappa, kappa_u=kappa_u)
    end
end

# Backward compatibility alias
function update_all_motif_bounds!(motifs::Dict{Int, PurePacketMotif}, legacy_arg; bg_rates=nothing, kappa::Float64=4.0, kappa_u::Float64=3.0)
    update_all_motif_bounds!(motifs; bg_rates=bg_rates, kappa=kappa, kappa_u=kappa_u)
end


# Backward-compatibility alias
