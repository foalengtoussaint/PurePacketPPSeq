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
    W_reach::Float64=0.75,
    min_packets::Int=2,
    min_spikes::Int=2,
    K_candidates::Int=30,
    occam_motif::Union{Float64, Nothing}=nothing,
    gamma_motif::Float64=100.0,
    alpha_a::Float64=1.0,
    beta_a::Float64=1.0,
    alpha_c::Float64=10.0,
    beta_c::Float64=10.0,
    alpha_m::Float64=0.05,
    beta_m::Float64=1.0,
    counts_bg::Union{Vector{Int}, Nothing}=nothing,
    ou_ctx::Union{Nothing, ContinuousOUContext}=nothing,
    rng::AbstractRNG=Random.default_rng(),
    W_window::Union{Nothing, Float64}=nothing,
    min_channels::Union{Nothing, Int}=nothing
)::Int
    N_spikes = length(spike_t)
    N_spikes < min_spikes && return 0

    reach = W_window !== nothing ? W_window : W_reach
    req_chans = min_channels !== nothing ? min_channels : 1
    req_pkts = max(min_packets, 2)

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
    delta_t_sep = 2.0 * sqrt(2.0) * sigma_prior # Rayleigh packet separation limit (~42.4 ms)

    for anchor_idx in bg_spike_indices
        t_seed = spike_t[anchor_idx]
        t_start = max(0.0, t_seed - reach)
        t_end = min(t_seed + reach, T_total)
        (t_end - t_start) < delta_t_sep && continue

        i_start = searchsortedfirst(spike_t, t_start)
        i_stop = searchsortedlast(spike_t, t_end)

        # Collect unassigned background spikes by channel
        chan_spk = Dict{Int, Vector{Int}}()
        for idx in i_start:i_stop
            if assign[idx].event_id == 0
                c = spike_neuron[idx]
                push!(get!(chan_spk, c, Int[]), idx)
            end
        end

        isempty(chan_spk) && continue

        # Rayleigh clustering per channel into candidate packets
        cand_pkts = @NamedTuple{chan::Int, spikes::Vector{Int}, mu::Float64, a_post::Float64, llr::Float64, net_evidence::Float64}[]

        for (c, spk_list) in chan_spk
            sort!(spk_list, by=s -> spike_t[s])
            
            clusters = Vector{Int}[]
            curr_cluster = Int[spk_list[1]]
            for j in 2:length(spk_list)
                if spike_t[spk_list[j]] - spike_t[spk_list[j-1]] <= delta_t_sep
                    push!(curr_cluster, spk_list[j])
                else
                    push!(clusters, curr_cluster)
                    curr_cluster = Int[spk_list[j]]
                end
            end
            push!(clusters, curr_cluster)

            for cl in clusters
                n_cl = length(cl)
                t_bar = sum(spike_t[s] for s in cl) / Float64(n_cl)
                mu_p = t_bar - t_seed

                # Negative Binomial log-evidence against instantaneous OU null
                sum_w = 0.0
                for s in cl
                    dt = spike_t[s] - t_bar
                    sum_w += exp(-0.5 * (dt * dt) / var_prior)
                end
                a_post = (alpha_a + sum_w) / (beta_a + 1.0)

                llr_p = 0.0
                for s in cl
                    t_s = spike_t[s]
                    dt = t_s - t_bar
                    lam_s = if ou_ctx !== nothing
                        exp(evaluate_background_log_intensity(ou_ctx, bg_rates, c, t_s))
                    else
                        (c <= length(bg_rates)) ? max(bg_rates[c], 1e-12) : 0.5
                    end
                    bump = a_post * exp(-0.5 * (dt * dt) / var_prior) / (sqrt(2π) * sigma_prior)
                    llr_p += log(max(bump / lam_s, 1.0))
                end
                # Exact Centroid Latency Marginalization Factor: -0.5 * ln(n_p)
                llr_p -= 0.5 * log(Float64(n_cl))

                net_ev = llr_p - log(beta_a + 1.0)
                push!(cand_pkts, (chan=c, spikes=cl, mu=mu_p, a_post=a_post, llr=llr_p, net_evidence=net_ev))
            end
        end

        length(cand_pkts) < req_pkts && continue
        K_cand = length(cand_pkts)

        # Sort candidate packets by descending net log-evidence
        sort!(cand_pkts, by=p -> p.net_evidence, rev=true)

        # Prefix search maximizing joint net log-evidence under dynamic envelope
        best_k = 0
        best_E_net = -Inf
        best_pkts = Packet[]
        best_tax_total = 0.0
        best_llr_sum = 0.0

        E_total = length(occurrences)
        tax_crp = log(max(1.0, Float64(E_total)) / max(1e-6, gamma_motif))

        max_prefix = min(K_cand, 50)
        for k in req_pkts:max_prefix
            prefix = cand_pkts[1:k]
            if req_chans > 1
                n_distinct = length(Set(p.chan for p in prefix))
                n_distinct < req_chans && continue
            end

            # Construct temporary packets to compute exact envelope via update_motif_bounds!
            pkts_cand = [
                Packet(p, prefix[p].chan, prefix[p].mu, var_prior, prefix[p].a_post)
                for p in 1:k
            ]
            temp_m = PurePacketMotif(0, pkts_cand, R_current, N_channels)
            update_motif_bounds!(temp_m; bg_rates=bg_rates)

            W_eff = max(temp_m.W_window, 6.0 * sigma_prior)
            tax_per_pkt = log(W_eff / (sqrt(2π) * sigma_prior)) + log(beta_a + 1.0)
            tax_chan = -compute_motif_channel_log_prior(temp_m.chan_to_packets, k, N_channels, W_eff; sigma_prior=sigma_prior)
            tax_dict = max(0.0, tax_crp) + k * tax_per_pkt + tax_chan
            tax_e0 = log((beta_m + T_total) / max(W_eff, 1e-6)) + log(1.0 / alpha_m)

            # Exact Combinatorial Model Selection Prior: ln \binom{K_cand}{k}
            log_comb = loggamma(K_cand + 1) - loggamma(k + 1) - loggamma(K_cand - k + 1)
            tax_total = occam_motif !== nothing ? occam_motif : (tax_dict + tax_e0 + log_comb)

            llr_sum = sum(prefix[p].llr for p in 1:k)
            if occam_motif === nothing
                A_m_cand = sum(prefix[p].a_post for p in 1:k)
                N_e_cand = sum(length(prefix[p].spikes) for p in 1:k)
                delta_motor = alpha_c * log(beta_c / (beta_c + A_m_cand)) +
                              loggamma(alpha_c + Float64(N_e_cand)) - loggamma(alpha_c) -
                              Float64(N_e_cand) * log(beta_c + A_m_cand)
                llr_sum += delta_motor
            end
            E_net = llr_sum - tax_total

            if E_net > best_E_net
                best_E_net = E_net
                best_k = k
                best_pkts = pkts_cand
                best_tax_total = tax_total
                best_llr_sum = llr_sum
            end
        end

        if best_k >= req_pkts && best_E_net > 0.0
            # Sort final chosen prefix packets chronologically by canonical latency mu_p
            sort!(best_pkts, by=p -> p.mu)
            for (new_p, pkt) in enumerate(best_pkts)
                pkt.id = new_p
            end

            # Accept birth of new motif
            new_mid = maximum(keys(motifs); init=0) + 1
            new_motif = PurePacketMotif(new_mid, best_pkts, R_current, N_channels)
            project_translational_gauge!(new_motif, trues(R_current))
            update_motif_bounds!(new_motif; bg_rates=bg_rates)
            motifs[new_mid] = new_motif

            next_eid = isempty(occurrences) ? 1 : (maximum(keys(occurrences)) + 1)
            tau_new = t_seed
            new_occ = PureOccurrence(next_eid, new_mid, tau_new, R_current, length(best_pkts); amplitude=1.0)

            # Map candidate clusters to their new canonical packet indices
            pkt_id_map = Dict{Int, Int}()
            for (new_p, pkt) in enumerate(best_pkts)
                # Find matching cluster in cand_pkts
                for orig_p in 1:best_k
                    if cand_pkts[orig_p].chan == pkt.channel && isapprox(cand_pkts[orig_p].mu, pkt.mu, atol=1e-6)
                        pkt_id_map[orig_p] = new_p
                        break
                    end
                end
            end

            for orig_p in 1:best_k
                new_p = get(pkt_id_map, orig_p, orig_p)
                for s_idx in cand_pkts[orig_p].spikes
                    assign[s_idx] = SpikeAssignment(next_eid, new_p)
                    push!(new_occ.members, s_idx)
                    new_occ.counts[new_p] += 1
                    new_occ.sum_t[new_p] += spike_t[s_idx]
                    new_occ.sum_t2[new_p] += spike_t[s_idx]^2
                    c = cand_pkts[orig_p].chan
                    if counts_bg !== nothing && 1 <= c <= length(counts_bg)
                        counts_bg[c] = max(0, counts_bg[c] - 1)
                    end
                end
            end
            occurrences[next_eid] = new_occ
            n_motifs_birthed += 1
            n_units = length(Set(p.channel for p in best_pkts))
            println(@sprintf("   --> [Level 3 Motif Birthed] Motif ID: %d | Packets: %d across %d units | Tau: %.2fs | Span: %.1fms (LLR: %.1f > Tax: %.1f, Net: +%.1f)",
                    new_mid, length(best_pkts), n_units, tau_new, new_motif.W_window * 1000.0, best_llr_sum, best_tax_total, best_E_net))
            break
        end
    end

    return n_motifs_birthed
end

"""
    count_candidate_clusters_in_window(spike_t, spike_neuron, assign, tau_seed, reach, T_total, sigma_prior, target_eid) -> Int

Reconstructs the candidate micro-cluster pool K_cand around tau_seed on D_bg ∪ S_{target_eid} using the exact Rayleigh separation limit.
"""
function count_candidate_clusters_in_window(
    spike_t::Vector{Float64},
    spike_neuron::Vector{Int},
    assign::Vector{SpikeAssignment},
    t_start::Float64,
    t_end::Float64,
    sigma_prior::Float64,
    target_eid::Int=0
)::Int
    delta_t_sep = 2.0 * sqrt(2.0) * sigma_prior

    i_start = searchsortedfirst(spike_t, t_start)
    i_stop = searchsortedlast(spike_t, t_end)

    chan_spk = Dict{Int, Vector{Float64}}()
    for idx in i_start:i_stop
        as = assign[idx]
        if as.event_id == 0 || as.event_id == target_eid
            c = spike_neuron[idx]
            push!(get!(chan_spk, c, Float64[]), spike_t[idx])
        end
    end

    total_clusters = 0
    for (c, times) in chan_spk
        sort!(times)
        n_cl = 1
        for j in 2:length(times)
            if times[j] - times[j-1] > delta_t_sep
                n_cl += 1
            end
        end
        total_clusters += n_cl
    end
    return total_clusters
end

function count_candidate_clusters_in_window(
    spike_t::Vector{Float64},
    spike_neuron::Vector{Int},
    assign::Vector{SpikeAssignment},
    tau_seed::Float64,
    reach::Float64,
    T_total::Float64,
    sigma_prior::Float64,
    target_eid::Int=0
)::Int
    t_start = max(0.0, tau_seed - reach)
    t_end = min(tau_seed + reach, T_total)
    return count_candidate_clusters_in_window(
        spike_t, spike_neuron, assign, t_start, t_end, sigma_prior, target_eid
    )
end

"""
    prune_dead_motifs!(
        motifs::Dict{Int, PurePacketMotif},
        occurrences::Dict{Int, PureOccurrence},
        frozen_motif_ids::Set{Int}=Set{Int}();
        spike_t::Union{Vector{Float64}, Nothing}=nothing,
        spike_neuron::Union{Vector{Int}, Nothing}=nothing,
        assign::Union{Vector{SpikeAssignment}, Nothing}=nothing,
        bg_rates::Union{Vector{Float64}, Nothing}=nothing,
        counts_bg::Union{Vector{Int}, Nothing}=nothing,
        active_mask::Union{BitVector, Nothing}=nothing,
        gamma_motif::Float64=100.0,
        alpha_c::Float64=10.0,
        beta_c::Float64=10.0,
        beta_a::Float64=1.0,
        sigma_prior::Float64=0.015,
        rng::AbstractRNG=Random.default_rng()
    )::Vector{Int}

Identifies and deletes non-frozen motifs using reversible-jump detailed balance:
1. Dead motifs with 0 active occurrences (E_m == 0) are unconditionally pruned.
2. Singleton motifs with 1 active occurrence (E_m == 1) undergo the symmetric joint
   motif death move (m, e) -> ∅. If the occurrence's marginal log-evidence cannot pay
   the P_m-packet continuous parameter Occam volume tax plus CRP motif prior:
       Δln Post = Δln M(e) - Tax_Occam(m)
   the joint jump is accepted with MH probability min(1, exp(-Δln Post) * (q_birth / q_death)),
   releasing all assigned spikes back to background.
Motifs in `frozen_motif_ids` are strictly protected and never deleted.
Returns the vector of pruned motif IDs.
"""
function prune_dead_motifs!(
    motifs::Dict{Int, PurePacketMotif},
    occurrences::Dict{Int, PureOccurrence},
    frozen_motif_ids::Set{Int}=Set{Int}();
    spike_t::Union{Vector{Float64}, Nothing}=nothing,
    spike_neuron::Union{Vector{Int}, Nothing}=nothing,
    assign::Union{Vector{SpikeAssignment}, Nothing}=nothing,
    bg_rates::Union{Vector{Float64}, Nothing}=nothing,
    counts_bg::Union{Vector{Int}, Nothing}=nothing,
    active_mask::Union{BitVector, Nothing}=nothing,
    ou_ctx::Union{Nothing, ContinuousOUContext}=nothing,
    T_total::Float64=25.0,
    gamma_motif::Float64=100.0,
    alpha_c::Float64=10.0,
    beta_c::Float64=10.0,
    alpha_a::Float64=1.0,
    beta_a::Float64=1.0,
    alpha_m::Float64=0.05,
    beta_m::Float64=1.0,
    sigma_prior::Float64=0.015,
    rng::AbstractRNG=Random.default_rng()
)::Vector{Int}
    # 1. Map each motif to its list of active occurrences
    motif_occs = Dict{Int, Vector{PureOccurrence}}()
    for occ in values(occurrences)
        push!(get!(motif_occs, occ.motif, PureOccurrence[]), occ)
    end

    dead_mids = Int[]

    # 2. Phase 1: Unconditional pruning of motifs with E_m == 0
    for mid in collect(keys(motifs))
        mid in frozen_motif_ids && continue
        occs = get(motif_occs, mid, PureOccurrence[])
        if isempty(occs)
            push!(dead_mids, mid)
            delete!(motifs, mid)
            println(@sprintf("   --> [Level 3 Motif Pruned] Motif ID: %d (Zero Active Occurrences)", mid))
        end
    end

    # 3. Phase 2: Unified Net Log-Evidence Joint Motif Death Jump (m, {e}) -> ∅ for E_m >= 1
    if assign !== nothing && spike_t !== nothing && bg_rates !== nothing
        n_bg_spikes = count(a -> a.event_id == 0, assign)
        candidate_mids = [
            mid for mid in keys(motifs)
            if mid ∉ frozen_motif_ids && !isempty(get(motif_occs, mid, PureOccurrence[]))
        ]
        n_candidates = length(candidate_mids)

        if n_candidates > 0 && n_bg_spikes > 0
            # Proposal ratio: q(birth) / q(death) = (1 / n_bg_spikes) / (1 / n_candidates)
            #                                     = n_candidates / n_bg_spikes
            log_q_ratio = log(Float64(n_candidates) / Float64(n_bg_spikes))
            E_total = length(occurrences)

            for mid in candidate_mids
                !haskey(motifs, mid) && continue
                m = motifs[mid]
                occs = get(motif_occs, mid, PureOccurrence[])
                isempty(occs) && continue
                E_m = length(occs)

                P_m = length(m.packets)
                update_motif_bounds!(m; bg_rates=bg_rates)
                W_eff = max(m.W_window, 6.0 * sigma_prior)
                tax_crp = log(max(1.0, Float64(E_total)) / max(1e-6, gamma_motif))
                tax_per_pkt = log(W_eff / (sqrt(2π) * sigma_prior)) + log(beta_a + 1.0)
                N_chans = (bg_rates !== nothing) ? length(bg_rates) : length(m.chan_to_packets)
                tax_chan = -compute_motif_channel_log_prior(m.chan_to_packets, P_m, N_chans, W_eff; sigma_prior=sigma_prior)
                tax_dict = max(0.0, tax_crp) + P_m * tax_per_pkt + tax_chan

                centroid_tax = 0.5 * sum(
                    begin
                        N_p = sum(occ.counts[p] for occ in occs; init=0)
                        log(Float64(max(1, N_p)))
                    end
                    for p in 1:P_m
                )

                if E_m == 1
                    # Guard 2 & 3: Evaluate exact combinatorial model prior and centroid
                    # marginalization factor across singleton E_m == 1 on reference pool D_bg ∪ S_{anchor}.
                    anchor_occ = occs[1]
                    reach = max(W_eff, 0.75)
                    K_cand = count_candidate_clusters_in_window(
                        spike_t, spike_neuron, assign, anchor_occ.tau, reach, T_total, sigma_prior, anchor_occ.id
                    )
                    if K_cand < P_m
                        continue # Reversibility guard: single-step birth unreachable (alpha_death = 0)
                    end
                    log_comb = loggamma(K_cand + 1) - loggamma(P_m + 1) - loggamma(K_cand - P_m + 1)
                    tax_dict += log_comb + centroid_tax
                else
                    # E_m >= 2: Exact generative dictionary prior matching Level m=1 packet discovery
                    tax_dict += centroid_tax
                end

                n_tot_occ_spk = sum(length(occ.members) for occ in occs)
                log_q_ratio = log(max(1.0, Float64(n_tot_occ_spk)) * Float64(n_candidates) / Float64(n_bg_spikes + n_tot_occ_spk))

                tau_min = max(0.0, -m.delta_left)
                tau_max = max(tau_min, T_total - m.delta_right)
                T_span = max(tau_max - tau_min, 1e-6)
                T_eff_m = compute_effective_motif_duration(ou_ctx, mid, tau_min, tau_max)
                tax_e0 = log((beta_m + T_eff_m) / max(W_eff, 1e-6)) + log(1.0 / alpha_m)

                mask = active_mask !== nothing ? active_mask : trues(size(m.W, 1))
                net_occ_evidence = 0.0

                for (idx_e, occ) in enumerate(occs)
                    delta_ln_M = calc_marginal_log_evidence(
                        m, occ.counts, occ.members, occ.tau, occ.u,
                        spike_t, assign, bg_rates, mask;
                        alpha_c=alpha_c, beta_c=beta_c, ou_ctx=ou_ctx
                    )
                    y_m = evaluate_motif_log_intensity(ou_ctx, mid, occ.tau)

                    if idx_e == 1
                        tax_occ_e = tax_e0 - y_m
                    else
                        ratio_E = Float64(idx_e) / (alpha_m + Float64(idx_e) - 1.0)
                        tax_occ_e = log((beta_m + T_eff_m) * ratio_E / max(T_span, 1e-6)) - y_m
                    end
                    net_occ_evidence += (delta_ln_M + y_m - tax_occ_e)
                end

                # Net log-evidence of keeping the motif:
                # E_net = \sum_e [ Δln M(e) + y_m(τ_e) - Tax_Occam(e) ] - Tax_dict(m)
                E_net = net_occ_evidence - tax_dict

                # Reverse jump (m, {e}) -> ∅ evaluated under exact potential reciprocity:
                # ln α_death = min(0, -E_net)
                log_alpha = min(0.0, -E_net)

                if log(rand(rng)) < log_alpha
                    # Accept symmetric joint death jump: release all assigned spikes to background and delete m
                    for occ in occs
                        for s_idx in occ.members
                            assign[s_idx] = SpikeAssignment(0, 0)
                            if counts_bg !== nothing && spike_neuron !== nothing
                                c = spike_neuron[s_idx]
                                if 1 <= c <= length(counts_bg)
                                    counts_bg[c] += 1
                                end
                            end
                        end
                        delete!(occurrences, occ.id)
                    end
                    delete!(motifs, mid)
                    push!(dead_mids, mid)
                    println(@sprintf("   --> [Level 3 Motif Pruned] Motif ID: %d (Joint Motif Death: E_net: %.1f < 0 | Ev: %.1f, DictTax: %.1f)",
                            mid, E_net, net_occ_evidence, tax_dict))
                end
            end
        end
    end

    return dead_mids
end
