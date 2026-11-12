# mechanisms/m3_motif.jl - Level m=3 Motif Archetype Dictionary, Loadings & Amplitude Dynamics
#
# Generative Role:
# - Governs the sequence motif archetype dictionary M = {m_1, ..., m_M}:
#   1. Nonparametric Data-Driven Motif Birth: proposes brand-new sequence motifs M -> M+1
#      from unassigned background spike bursts that exceed the Occam complexity threshold.
#   2. Occam Motif Death: prunes non-frozen motifs whose active occurrence count drops to 0 (M -> M-1).
#   3. Frozen Motif Preservation: protects baseline task motifs (e.g., motifs 1..9) from deletion.
# - Enforces exact translational gauge symmetry on manifold loadings W:
#     sum_{p=1}^{P_m} kappa_{m,p} W_{r,p} == 0
#   where kappa_{m,p} = 1.0 / var_p is the precision of emission packet p.
# - Coordinates occurrence-level amplitude / motor vigor dynamics c_e ~ Gamma(alpha_c, beta_c)
#   modulating the motif's overall emission intensity across continuous time.

"""
    project_translational_gauge!(m::PurePacketMotif, active_mask::BitVector)

Strictly projects manifold deformation loadings W onto the orthogonal subspace:
    sum_p kappa_{m,p} W_{r,p} == 0
for all active ARD dimensions r.
Zeroes out dormant dimensions.
"""
function project_translational_gauge!(m::PurePacketMotif, active_mask::BitVector)
    P = length(m.packets)
    P == 0 && return
    R_max = length(active_mask)
    tot_prec = sum(1.0 / m.packets[p].var for p in 1:P)
    tot_prec <= 0.0 && return

    for r in 1:R_max
        if active_mask[r]
            w_mean = sum((1.0 / m.packets[p].var) * m.W[r, p] for p in 1:P) / tot_prec
            @views m.W[r, :] .-= w_mean
        else
            @views m.W[r, :] .= 0.0
        end
    end
end

"""
    compute_gauge_error(m::PurePacketMotif, active_mask::BitVector) -> Float64

Evaluates maximum normalized gauge error across active ARD axes:
    max_r |sum_p kappa_p W_{r,p}| / sum_p kappa_p
Verifies exact machine-zero gauge invariance.
"""
function compute_gauge_error(m::PurePacketMotif, active_mask::BitVector)::Float64
    P = length(m.packets)
    P == 0 && return 0.0
    tot_prec = sum(1.0 / pkt.var for pkt in m.packets)
    tot_prec <= 0.0 && return 0.0

    max_err = 0.0
    for r in 1:length(active_mask)
        if active_mask[r]
            err = abs(sum((1.0 / m.packets[p].var) * m.W[r, p] for p in 1:P) / tot_prec)
            max_err = max(max_err, err)
        end
    end
    return max_err
end

"""
    detect_and_birth_new_motifs!(
        motifs::Dict{Int, PurePacketMotif},
        occurrences::Dict{Int, PureOccurrence},
        assign::Vector{SpikeAssignment},
        spike_t::Vector{Float64},
        spike_neuron::Vector{Int},
        bg_rates::Vector{Float64},
        T_total::Float64,
        R_current::Int;
        N_channels::Int=411,
        W_window::Float64=0.40,
        min_channels::Int=5,
        min_spikes::Int=8,
        K_candidates::Int=30,
        occam_motif::Float64=20.0,
        rng::AbstractRNG=Random.default_rng()
    ) -> Int

Proposes brand-new sequence motifs M -> M+1 from dense unassigned background spike bursts.
Evaluates the Marginal Log-Evidence Delta ln M of a candidate new motif against the Poisson
background null hypothesis:
    Delta ln M = sum_{p=1}^P ln( (a_p * N(0; 0, sigma_p^2)) / lambda_{0, c(p)} )
Accepts the birth if Delta ln M > occam_motif, initializing a new PurePacketMotif, its packets,
and its seed occurrence.
"""
function detect_and_birth_new_motifs!(
    motifs::Dict{Int, PurePacketMotif},
    occurrences::Dict{Int, PureOccurrence},
    assign::Vector{SpikeAssignment},
    spike_t::Vector{Float64},
    spike_neuron::Vector{Int},
    bg_rates::Vector{Float64},
    T_total::Float64,
    R_current::Int;
    N_channels::Int=411,
    W_window::Float64=0.40,
    min_channels::Int=5,
    min_spikes::Int=8,
    K_candidates::Int=30,
    occam_motif::Float64=20.0,
    rng::AbstractRNG=Random.default_rng()
)::Int
    N_spikes = length(spike_t)
    N_spikes < min_spikes && return 0

    bg_spike_indices = Int[]
    sizehint!(bg_spike_indices, 128)
    for _ in 1:K_candidates
        rand_idx = rand(rng, 1:N_spikes)
        if assign[rand_idx].event_id == 0
            push!(bg_spike_indices, rand_idx)
        end
    end
    isempty(bg_spike_indices) && return 0

    n_motifs_birthed = 0
    sigma_prior = 0.015
    var_prior = sigma_prior^2

    for anchor_idx in bg_spike_indices
        t_anchor = spike_t[anchor_idx]
        t_start = max(0.0, t_anchor - 0.05)
        t_end = min(t_start + W_window, T_total)
        (t_end - t_start) < 0.10 && continue

        i_start = searchsortedfirst(spike_t, t_start)
        i_stop = searchsortedlast(spike_t, t_end)

        cand_spikes = Int[]
        cand_chans = Set{Int}()
        for idx in i_start:i_stop
            if assign[idx].event_id == 0
                push!(cand_spikes, idx)
                push!(cand_chans, spike_neuron[idx])
            end
        end

        if length(cand_chans) < min_channels || length(cand_spikes) < min_spikes
            continue
        end

        chan_first_t = Dict{Int, Float64}()
        chan_spk_map = Dict{Int, Int}()
        for idx in cand_spikes
            c = spike_neuron[idx]
            dt = spike_t[idx] - t_start
            if !haskey(chan_first_t, c) || dt < chan_first_t[c]
                chan_first_t[c] = dt
                chan_spk_map[c] = idx
            end
        end

        sorted_chans = sort(collect(keys(chan_first_t)), by=c -> chan_first_t[c])
        origin_t = chan_first_t[sorted_chans[1]]

        pkts = Packet[
            Packet(
                p, c,
                chan_first_t[c] - origin_t,
                var_prior,
                1.0
            )
            for (p, c) in enumerate(sorted_chans)
        ]

        llr = 0.0
        for (p, c) in enumerate(sorted_chans)
            lam0 = (c <= length(bg_rates)) ? max(bg_rates[c], 1e-12) : 0.5
            N_j = 1.0 / (sqrt(2π) * sigma_prior)
            ratio = (1.0 * N_j) / lam0
            llr += log(max(ratio, 1.0))
        end

        if llr > occam_motif
            new_mid = maximum(keys(motifs); init=0) + 1
            new_motif = PurePacketMotif(new_mid, pkts, R_current, N_channels; W_window=W_window)
            project_translational_gauge!(new_motif, trues(R_current))
            update_motif_bounds!(new_motif; bg_rates=bg_rates)
            motifs[new_mid] = new_motif

            next_eid = isempty(occurrences) ? 1 : (maximum(keys(occurrences)) + 1)
            tau_new = t_start + origin_t
            new_occ = PureOccurrence(next_eid, new_mid, tau_new, R_current, length(pkts); amplitude=1.0)

            for (p, c) in enumerate(sorted_chans)
                s_idx = chan_spk_map[c]
                assign[s_idx] = SpikeAssignment(next_eid, p)
                push!(new_occ.members, s_idx)
                new_occ.counts[p] = 1
                new_occ.sum_t[p] = spike_t[s_idx]
                new_occ.sum_t2[p] = spike_t[s_idx]^2
            end
            occurrences[next_eid] = new_occ
            n_motifs_birthed += 1
            println(@sprintf("   --> [Level 3 Motif Birthed] Motif ID: %d | Packets: %d across %d units | Onset: %.2fs (LLR: %.1f > Tax: %.1f)",
                    new_mid, length(pkts), length(sorted_chans), tau_new, llr, occam_motif))
            break
        end
    end

    return n_motifs_birthed
end

"""
    prune_dead_motifs!(
        motifs::Dict{Int, PurePacketMotif},
        occurrences::Dict{Int, PureOccurrence},
        frozen_motif_ids::Set{Int}=Set{Int}()
    )::Vector{Int}

Identifies and deletes any non-frozen motif whose active occurrence count has dropped to 0 (M -> M-1).
Motifs in `frozen_motif_ids` (e.g. task motifs 1..9) are strictly protected and never deleted.
Returns the vector of pruned motif IDs.
"""
function prune_dead_motifs!(
    motifs::Dict{Int, PurePacketMotif},
    occurrences::Dict{Int, PureOccurrence},
    frozen_motif_ids::Set{Int}=Set{Int}()
)::Vector{Int}
    motif_occ_counts = Dict{Int, Int}()
    for occ in values(occurrences)
        motif_occ_counts[occ.motif] = get(motif_occ_counts, occ.motif, 0) + 1
    end

    dead_mids = Int[]
    for mid in collect(keys(motifs))
        if mid ∉ frozen_motif_ids && get(motif_occ_counts, mid, 0) == 0
            push!(dead_mids, mid)
            delete!(motifs, mid)
            println(@sprintf("   --> [Level 3 Motif Pruned] Motif ID: %d (Zero Active Occurrences)", mid))
        end
    end

    return dead_mids
end
