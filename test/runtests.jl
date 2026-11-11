using Test, Random, Statistics, LinearAlgebra, StaticArrays, Distributions, SpecialFunctions, Serialization, Printf

include(joinpath(@__DIR__, "..", "src", "pure_packet_engine.jl"))

function check_coord_alloc(u_out, m, active_mask, res, prec, prior_p, rng)
    sample_occurrence_coordinate!(u_out, m, active_mask, res, prec; prior_precisions=prior_p, rng=rng)
    return @allocated sample_occurrence_coordinate!(u_out, m, active_mask, res, prec; prior_precisions=prior_p, rng=rng)
end

function check_bg_alloc(bg_rates_buf, pi_bg_buf, gamma_buf, counts_bg_buf, test_assign, test_neuron, chan_group, groups, bg_a, bg_b, beta_bg, T_tot, exposure, rng)
    resample_background_pure_packet!(bg_rates_buf, pi_bg_buf, gamma_buf, counts_bg_buf, test_assign, test_neuron, chan_group, groups, bg_a, bg_b, beta_bg, T_tot, exposure; rng=rng)
    return @allocated resample_background_pure_packet!(bg_rates_buf, pi_bg_buf, gamma_buf, counts_bg_buf, test_assign, test_neuron, chan_group, groups, bg_a, bg_b, beta_bg, T_tot, exposure; rng=rng)
end

@testset "Pure Packet Engine Verification" begin
    rng = MersenneTwister(42)
    N_chans = 50
    R_max = 4

    # -------------------------------------------------------------
    # TEST 1: ANCHOR HANDOVER ON PACKET 1 MASS COLLAPSE
    # -------------------------------------------------------------
    @testset "Anchor Handover When Packet 1 Dies" begin
        p1 = Packet(1, 10, 0.000, 0.0004, 0.0001)  # Low mass (will die)
        p2 = Packet(2, 25, 0.080, 0.0004, 1.2)     # Strong packet
        p3 = Packet(3, 30, 0.250, 0.0004, 0.9)     # Strong packet

        m = PurePacketMotif(1, [p1, p2, p3], R_max, N_chans; W_window=1.0)
        
        # 10 Occurrences around tau = 100.0s
        E = 10
        occurrences = [PureOccurrence(e, 1, 100.0 + (e - 1) * 2.0, R_max, 3) for e in 1:E]
        for occ in occurrences
            occ.counts[1] = 0        # 0 spikes on Packet 1
            occ.counts[2] = 15       # 15 spikes on Packet 2
            occ.counts[3] = 12       # 12 spikes on Packet 3
        end

        # Record physical expected arrival times before pruning: t_arr = tau + mu_p
        expected_t2_before = [occ.tau + m.packets[2].mu for occ in occurrences]
        expected_t3_before = [occ.tau + m.packets[3].mu for occ in occurrences]

        bg_rates = fill(0.5, N_chans)
        pruned = prune_dead_packets!(m, occurrences, bg_rates, N_chans; occam_tax=1.0)

        @test 1 in pruned
        @test length(m.packets) == 2
        # Former Packet 2 is now Packet 1 with mu == 0.0 identically
        @test m.packets[1].channel == 25
        @test m.packets[1].mu == 0.0
        # Former Packet 3 is now Packet 2 with mu == 0.250 - 0.080 = 0.170s
        @test m.packets[2].channel == 30
        @test isapprox(m.packets[2].mu, 0.170, atol=1e-12)

        # Inverted index updated
        @test m.chan_to_packets[10] == Int[]
        @test m.chan_to_packets[25] == [1]
        @test m.chan_to_packets[30] == [2]

        # Verify exact machine-precision arrival time preservation across anchor handover!
        expected_t2_after = [occ.tau + m.packets[1].mu for occ in occurrences]
        expected_t3_after = [occ.tau + m.packets[2].mu for occ in occurrences]
        max_err_t2 = maximum(abs.(expected_t2_after .- expected_t2_before))
        max_err_t3 = maximum(abs.(expected_t3_after .- expected_t3_before))

        @test max_err_t2 == 0.0
        @test max_err_t3 == 0.0
        println("Anchor handover maximum physical arrival shift: ", max(max_err_t2, max_err_t3), " s (Exact zero drift)")
    end

    # -------------------------------------------------------------
    # TEST 2: BIMODAL PACKET SPLIT ON A SINGLE NEURON
    # -------------------------------------------------------------
    @testset "Continuous Bayes Factor Bimodal Split" begin
        # Neuron 25 starts with a single packet at 50 ms
        p1 = Packet(1, 25, 0.050, 0.0004, 1.0)
        p2 = Packet(2, 10, 0.150, 0.0004, 1.0)
        m = PurePacketMotif(1, [p1, p2], R_max, N_chans; W_window=1.0)

        E = 30
        occurrences = [PureOccurrence(e, 1, 10.0 + (e - 1) * 1.5, R_max, 2) for e in 1:E]

        spike_t = Float64[]
        spike_chan = Int[]
        assign = SpikeAssignment[]

        # Generate true bimodal spikes on Neuron 25:
        # Peak A: around 50 ms (assigned to Packet 1)
        # Peak B: around 320 ms (residual > 1.5 sigma from Packet 1, triggering split)
        spk_idx = 0
        for (e, occ) in enumerate(occurrences)
            # Peak A
            for _ in 1:2
                spk_idx += 1
                t = occ.tau + 0.050 + 0.005 * randn(rng)
                push!(spike_t, t)
                push!(spike_chan, 25)
                push!(assign, SpikeAssignment(e, 1))
                push!(occ.members, spk_idx)
                occ.counts[1] += 1
            end
            # Peak B: second burst at 320 ms
            for _ in 1:3
                spk_idx += 1
                t = occ.tau + 0.320 + 0.005 * randn(rng)
                push!(spike_t, t)
                push!(spike_chan, 25)
                # Currently assigned to Packet 1 but with huge residual |dt| ~ 270 ms >> 1.5 sigma
                push!(assign, SpikeAssignment(e, 1))
                push!(occ.members, spk_idx)
                occ.counts[1] += 1
            end
        end

        bg_rates = fill(0.2, N_chans)
        birthed = detect_and_birth_or_split_packets!(
            m, occurrences, spike_t, spike_chan, assign, bg_rates, N_chans;
            sigma_prior=0.015, alpha_a=1.0, beta_a=1.0
        )

        @test birthed == true
        @test length(m.packets) == 3
        # Channel 25 now has two packets!
        @test m.chan_to_packets[25] == [1, 3]
        new_pkt = m.packets[3]
        @test new_pkt.channel == 25
        println("Birthed packet 3 latency (True 0.320 s): ", round(new_pkt.mu, digits=4), " s")
        @test isapprox(new_pkt.mu, 0.320, atol=0.015)
        # New packet loadings are identically zero
        @test all(m.W[:, 3] .== 0.0)
    end

    # -------------------------------------------------------------
    # TEST 3: TRANSLATIONAL GAUGE INVARIANCE UNDER MULTI-PACKET
    # -------------------------------------------------------------
    @testset "Translational Gauge Projection" begin
        # Neuron 25 has Packet 1 (50ms) and Packet 2 (320ms)
        p1 = Packet(1, 25, 0.000, 0.0004, 1.0)
        p2 = Packet(2, 25, 0.270, 0.0004, 1.0)
        p3 = Packet(3, 10, 0.100, 0.0004, 1.0)
        m = PurePacketMotif(1, [p1, p2, p3], R_max, N_chans; W_window=1.0)

        state_alpha = [500.0, 500.0, 1e6, 1e6]
        active_mask = trues(R_max)
        active_mask[3] = false
        active_mask[4] = false

        E = 50
        occ_u = [randn(rng, R_max) for _ in 1:E]
        residuals = [randn(rng, 3) * 0.01 for _ in 1:E]
        precisions = [fill(2500.0, 3) for _ in 1:E]

        sample_packet_loadings!(m, state_alpha, active_mask, occ_u, residuals, precisions; rng=rng)

        # Check translational gauge condition: sum_p (1/var_p) * W_{r,p} == 0
        tot_prec = sum(1.0 / m.packets[p].var for p in 1:3)
        for r in 1:2
            gauge_val = sum((1.0 / m.packets[p].var) * m.W[r, p] for p in 1:3)
            println("Gauge condition axis $r: ", gauge_val)
            @test abs(gauge_val) < 1e-12
        end
    end

    # -------------------------------------------------------------
    # TEST 4: ZERO HEAP ALLOCATIONS IN COORDINATE SOLVE
    # -------------------------------------------------------------
    @testset "Zero Heap Allocations" begin
        p1 = Packet(1, 25, 0.000, 0.0004, 1.0)
        p2 = Packet(2, 25, 0.270, 0.0004, 1.0)
        m = PurePacketMotif(1, [p1, p2], R_max, N_chans)
        active_mask = trues(R_max)
        active_mask[3] = false; active_mask[4] = false
        u_out = zeros(Float64, R_max)
        res = [0.005, -0.005]
        prec = [2500.0, 2500.0]
        prior_p = [1.0, 1.0, 1.0, 1.0]

        # Warm-up
        sample_occurrence_coordinate!(u_out, m, active_mask, res, prec; prior_precisions=prior_p, rng=rng)

        # Benchmark allocations
        allocs = check_coord_alloc(u_out, m, active_mask, res, prec, prior_p, rng)
        println("Allocations in sample_occurrence_coordinate!: ", allocs, " bytes")
        @test allocs == 0
    end

    # -------------------------------------------------------------
    # TEST 5: DUAL-INDEX SPIKE GIBBS RESAMPLER
    # -------------------------------------------------------------
    @testset "Dual-Index Spike Assignment" begin
        # Neuron 25 has Packet 1 (50ms) and Packet 2 (320ms)
        p1 = Packet(1, 25, 0.050, 0.0004, 2.0)
        p2 = Packet(2, 25, 0.320, 0.0004, 2.0)
        m = PurePacketMotif(1, [p1, p2], R_max, N_chans)
        motifs_dict = Dict(1 => m)

        occ = PureOccurrence(1, 1, 10.0, R_max, 2)
        occ_dict = Dict(1 => occ)

        bg_rates = fill(0.01, N_chans)
        active_mask = trues(R_max)

        assign = [SpikeAssignment(0, 0), SpikeAssignment(0, 0), SpikeAssignment(0, 0)]
        spikes_t = [10.051, 10.319, 10.850]
        spikes_c = [25, 25, 25]

        buf_occ = Int[]
        buf_pkt = Int[]
        buf_logw = Float64[]

        # Resample all three spikes
        for spk_i in 1:3
            resample_spike_pure_packet!(
                spk_i, spikes_t[spk_i], spikes_c[spk_i],
                assign, values(occ_dict), occ_dict, motifs_dict, bg_rates, active_mask,
                buf_occ, buf_pkt, buf_logw; rng=rng
            )
        end

        # Spike 1 -> (Event 1, Packet 1)
        @test assign[1].event_id == 1
        @test assign[1].packet_idx == 1

        # Spike 2 -> (Event 1, Packet 2)
        @test assign[2].event_id == 1
        @test assign[2].packet_idx == 2

        # Spike 3 -> Background (0, 0)
        @test assign[3].event_id == 0
        @test assign[3].packet_idx == 0

        # Occurrence moments updated correctly
        @test occ.counts == [1, 1]
        @test isapprox(occ.sum_t[1], 10.051)
        @test isapprox(occ.sum_t[2], 10.319)
        println("Dual-index assignments: Spk1 -> (", assign[1].event_id, ", ", assign[1].packet_idx,
                "), Spk2 -> (", assign[2].event_id, ", ", assign[2].packet_idx,
                "), Spk3 -> (", assign[3].event_id, ", ", assign[3].packet_idx, ")")

        # Test Channel-First Routed Resampler
        chan_to_motifs = update_chan_to_motifs(motifs_dict, N_chans)
        @test 1 in chan_to_motifs[25]
        motif_occs = Dict(1 => [occ])
        motif_taus = Dict(1 => [occ.tau])
        fill!(occ.counts, 0); fill!(occ.sum_t, 0.0); fill!(occ.sum_t2, 0.0)
        assign = [SpikeAssignment(0, 0), SpikeAssignment(0, 0), SpikeAssignment(0, 0)]
        for spk_i in 1:3
            resample_spike_pure_packet!(
                spk_i, spikes_t[spk_i], spikes_c[spk_i],
                assign, chan_to_motifs, motif_occs, motif_taus,
                occ_dict, motifs_dict, bg_rates, active_mask,
                buf_occ, buf_pkt, buf_logw; rng=rng
            )
        end
        @test assign[1].event_id == 1 && assign[1].packet_idx == 1
        @test assign[2].event_id == 1 && assign[2].packet_idx == 2
        @test assign[3].event_id == 0 && assign[3].packet_idx == 0
    end

    # -------------------------------------------------------------
    # TEST 6: LEVEL m=0 ZERO-ALLOCATION BACKGROUND RESAMPLER
    # -------------------------------------------------------------
    @testset "Level m=0 Zero-Allocation Background Resampling" begin
        N_test_chans = 10
        groups = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
        chan_group = [k <= 5 ? 1 : 2 for k in 1:N_test_chans]
        bg_a = [1.0, 1.0]
        bg_b = [0.01, 0.01]
        beta_bg = [0.5, 0.5]
        T_tot = 100.0
        exposure = [1.0, 1.0]

        bg_rates_buf = zeros(Float64, N_test_chans)
        pi_bg_buf = zeros(Float64, N_test_chans)
        gamma_buf = zeros(Float64, N_test_chans)
        counts_bg_buf = zeros(Int, N_test_chans)

        # 500 spikes, 300 background on channels 1-5, 200 on occurrences
        N_test_spk = 500
        test_assign = [i <= 300 ? SpikeAssignment(0, 0) : SpikeAssignment(1, 1) for i in 1:N_test_spk]
        test_neuron = [mod1(i, N_test_chans) for i in 1:N_test_spk]

        # Warm-up call
        resample_background_pure_packet!(
            bg_rates_buf, pi_bg_buf, gamma_buf, counts_bg_buf,
            test_assign, test_neuron, chan_group, groups,
            bg_a, bg_b, beta_bg, T_tot, exposure; rng=rng
        )

        # Check total rate matches approximate empirical rate
        tot_rate_g1 = sum(bg_rates_buf[1:5])
        # Group 1 has ~150 background spikes over T=100s -> rate ~ 1.5 Hz
        @test 1.0 < tot_rate_g1 < 2.5
        @test isapprox(sum(pi_bg_buf[1:5]), 1.0, atol=1e-12)
        @test isapprox(sum(pi_bg_buf[6:10]), 1.0, atol=1e-12)

        # Verify zero heap allocations inside compiled function
        allocs = check_bg_alloc(
            bg_rates_buf, pi_bg_buf, gamma_buf, counts_bg_buf,
            test_assign, test_neuron, chan_group, groups,
            bg_a, bg_b, beta_bg, T_tot, exposure, rng
        )
        println("Allocations in resample_background_pure_packet!: ", allocs, " bytes")
        @test allocs == 0
    end

    # -------------------------------------------------------------
    # TEST 7: OCCURRENCE AMPLITUDE RESAMPLING & COMPENSATOR COUPLING
    # -------------------------------------------------------------
    @testset "Occurrence Amplitude Resampling & Scale Anchoring" begin
        rng = MersenneTwister(42)
        # Create motif with 2 packets of total mass 1.0 + 1.0 = 2.0
        p1 = Packet(1, 1, 0.0, 0.0004, 1.0)
        p2 = Packet(2, 2, 0.1, 0.0004, 1.0)
        m = PurePacketMotif(1, [p1, p2], R_max, N_chans)
        motifs_dict = Dict(1 => m)

        # 100 occurrences expected to fire 2 spikes each (total mass 2.0)
        E = 100
        occs = [PureOccurrence(e, 1, 10.0 + e, R_max, 2) for e in 1:E]
        for occ in occs
            occ.counts[1] = 1
            occ.counts[2] = 1
        end

        sample_occurrence_amplitudes!(occs, motifs_dict; alpha_c=25.0, rng=rng)
        mean_c = mean(occ.amplitude for occ in occs)
        println("Mean occurrence amplitude across 100 occurrences: ", round(mean_c, digits=4))
        # Prior alpha_c = 25 clamps ensemble mean strictly around 1.00 +- 0.05
        @test isapprox(mean_c, 1.00, atol=0.05)

        # Test high-gain occurrence (fires 8 spikes instead of 2)
        high_gain_occ = PureOccurrence(999, 1, 50.0, R_max, 2)
        high_gain_occ.counts[1] = 4
        high_gain_occ.counts[2] = 4
        sample_occurrence_amplitudes!([high_gain_occ], motifs_dict; alpha_c=25.0, rng=rng)
        println("High-gain occurrence amplitude (8 spikes vs 2 expected): ", round(high_gain_occ.amplitude, digits=3))
        # (25 + 8) / (25 + 2) = 33 / 27 = 1.22
        @test high_gain_occ.amplitude > 1.1

        # Test bidirectional exposure in sample_packet_masses!
        sample_packet_masses!(m, occs; rng=rng)
        # Total mass should stay stable around 2.0 (1.0 each)
        @test isapprox(m.packets[1].mass, 1.0, atol=0.2)
        @test isapprox(m.packets[2].mass, 1.0, atol=0.2)
    end

    # -------------------------------------------------------------
    # TEST 8: SESSION BOUNDARY ONSET CENSORING
    # -------------------------------------------------------------
    @testset "Session Boundary Onset Censoring" begin
        p1 = Packet(1, 1, 0.0, 0.0004, 1.0)
        m = PurePacketMotif(1, [p1], R_max, N_chans; W_window=2.5)
        occ = PureOccurrence(1, 1, 98.0, R_max, 1)
        occ.counts[1] = 1
        # Place spike very close to T_max = 100.0 (e.g. at 99.5)
        occ.sum_t[1] = 99.5

        # Resample onset with T_max = 100.0. Since W_window = 2.5, max valid tau is 100.0 - 2.5 = 97.5s
        for _ in 1:20
            resample_occurrence_onset!(occ, m, falses(R_max); T_max=100.0, rng=rng)
            @test occ.tau <= 97.50000001
            @test occ.tau >= 0.0
        end
        println("Boundary onset successfully clamped <= 97.5s: final tau = ", round(occ.tau, digits=4))
    end

    # -------------------------------------------------------------
    # TEST 9: OCCURRENCE PRUNING OPERATOR
    # -------------------------------------------------------------
    @testset "Occurrence Pruning Operator" begin
        p1 = Packet(1, 1, 0.0, 0.0004, 1.0)
        p2 = Packet(2, 2, 0.1, 0.0004, 1.0)
        m = PurePacketMotif(1, [p1, p2], R_max, N_chans)
        motifs_dict = Dict(1 => m)
        bg_rates = fill(0.05, N_chans)

        occ1 = PureOccurrence(1, 1, 10.0, R_max, 2; amplitude=1.0)
        occ1.counts[1] = 8; occ1.counts[2] = 7; occ1.members = [1, 2, 3]

        occ2 = PureOccurrence(2, 1, 20.0, R_max, 2; amplitude=1.0) # 0 spikes
        occ2.members = Int[]

        occ3 = PureOccurrence(3, 1, 30.0, R_max, 2; amplitude=0.10) # collapsed amplitude
        occ3.counts[1] = 1; occ3.members = [4]

        occs_dict = Dict(1 => occ1, 2 => occ2, 3 => occ3)
        assign = [SpikeAssignment(1, 1), SpikeAssignment(1, 1), SpikeAssignment(1, 2), SpikeAssignment(3, 1)]

        pruned = prune_dead_occurrences!(occs_dict, motifs_dict, assign, bg_rates; min_spikes=2, min_amplitude=0.25)
        @test 2 in pruned
        @test 3 in pruned
        @test !(1 in pruned)
        @test haskey(occs_dict, 1)
        @test !haskey(occs_dict, 2)
        @test !haskey(occs_dict, 3)
        # Spike 4 from Occ 3 was released back to background
        @test assign[4] == SpikeAssignment(0, 0)
        # Spike 1 from Occ 1 remains assigned
        @test assign[1] == SpikeAssignment(1, 1)
        println("Pruned zombie occurrences 2 and 3; verified spike 4 released to background.")

        # Test Exact Marginal Negative Binomial Prune
        # Spike 4 is noise at 30.50s (dt = 0.50s >> sigma), so Occ 3 has negative evidence
        spike_times_test = [10.05, 10.06, 10.15, 30.50]
        active_mask_test = trues(R_max)
        occ3_reborn = PureOccurrence(3, 1, 30.0, R_max, 2; amplitude=0.10)
        occ3_reborn.counts[1] = 1; occ3_reborn.members = [4]
        occs_dict[3] = occ3_reborn
        assign[4] = SpikeAssignment(3, 1)
        
        pruned_nb = prune_dead_occurrences!(
            occs_dict, motifs_dict, spike_times_test, assign, bg_rates, 100.0, active_mask_test;
            alpha_c=10.0, beta_c=10.0, gamma_m=100.0
        )
        @test 3 in pruned_nb
        @test !haskey(occs_dict, 3)
        @test assign[4] == SpikeAssignment(0, 0)
    end

    # -------------------------------------------------------------
    # TEST 10: OCCURRENCE BIRTH & COMPETITIVE OVERWRITE
    # -------------------------------------------------------------
    @testset "Occurrence Birth & Competitive Overwrite" begin
        p1 = Packet(1, 5, 0.000, 0.0004, 2.0)
        p2 = Packet(2, 12, 0.150, 0.0004, 2.0)
        m = PurePacketMotif(1, [p1, p2], R_max, N_chans; W_window=1.0)
        motifs_dict = Dict(1 => m)
        bg_rates = fill(0.02, N_chans)

        # Build synthetic spikes: 10 background spikes + an unmodeled replay at tau = 42.0s
        spikes_t = Float64[]
        spikes_chan = Int[]
        
        # 10 tonic background spikes
        for i in 1:10
            push!(spikes_t, 1.0 + i * 2.0)
            push!(spikes_chan, 5)
        end

        # Replay at 42.0s: 6 spikes on chan 5 at ~42.000s, 6 spikes on chan 12 at ~42.150s
        for dt in [-0.015, -0.008, 0.0, 0.005, 0.012, 0.018]
            push!(spikes_t, 42.000 + dt)
            push!(spikes_chan, 5)
        end
        for dt in [-0.012, -0.005, 0.0, 0.006, 0.014, 0.019]
            push!(spikes_t, 42.150 + dt)
            push!(spikes_chan, 12)
        end

        # Sort all spikes by time
        p_order = sortperm(spikes_t)
        spikes_t = spikes_t[p_order]
        spikes_chan = spikes_chan[p_order]

        N_total_spk = length(spikes_t)
        chan_to_spk = [Int[] for _ in 1:N_chans]
        for (idx, c) in enumerate(spikes_chan)
            push!(chan_to_spk[c], idx)
        end
        assign = [SpikeAssignment(0, 0) for _ in 1:N_total_spk]
        occs_dict = Dict{Int, PureOccurrence}()

        # Step A: Detect and birth the unmodeled replay
        birthed, overwrote = detect_and_birth_occurrences!(
            occs_dict, motifs_dict, spikes_t, spikes_chan, chan_to_spk,
            assign, bg_rates, 100.0, R_max; gamma_m=50.0, delta_margin=3.0, rng=rng
        )

        @test birthed >= 1
        @test length(occs_dict) == 1
        birthed_occ = first(values(occs_dict))
        println("Birthed occurrence onset: ", round(birthed_occ.tau, digits=4), " s (True 42.000 s)")
        @test isapprox(birthed_occ.tau, 42.000, atol=0.015)
        @test sum(birthed_occ.counts) >= 10
        @test birthed_occ.amplitude > 0.8

        # Step B: Competitive Overwrite of a misaligned zombie event
        # Place a weak zombie occurrence at tau = 60.002s with non-zero u coordinate
        zombie_eid = 999
        zombie_occ = PureOccurrence(zombie_eid, 1, 60.002, R_max, 2; amplitude=0.4)
        zombie_occ.u .= [1.2, -0.9, 0.5, 0.0] # corrupted coordinate
        # Add 2 spikes to zombie
        push!(spikes_t, 60.002); push!(spikes_chan, 5)
        push!(spikes_t, 60.152); push!(spikes_chan, 12)
        
        # Add a true strong 14-spike burst at 60.12s
        for dt in [-0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015]
            push!(spikes_t, 60.120 + dt); push!(spikes_chan, 5)
        end
        for dt in [-0.015, -0.01, -0.005, 0.0, 0.005, 0.01, 0.015]
            push!(spikes_t, 60.270 + dt); push!(spikes_chan, 12)
        end

        # Re-sort
        p_order = sortperm(spikes_t)
        spikes_t = spikes_t[p_order]
        spikes_chan = spikes_chan[p_order]
        N_new_spk = length(spikes_t)
        chan_to_spk = [Int[] for _ in 1:N_chans]
        for (idx, c) in enumerate(spikes_chan)
            push!(chan_to_spk[c], idx)
        end
        assign = [SpikeAssignment(0, 0) for _ in 1:N_new_spk]
        
        # Zombie occurrence owns its 2 spikes after sorting
        z1 = findfirst(t -> isapprox(t, 60.002, atol=1e-5), spikes_t)
        z2 = findfirst(t -> isapprox(t, 60.152, atol=1e-5), spikes_t)
        zombie_occ.counts = [1, 1]
        zombie_occ.members = [z1, z2]
        assign[z1] = SpikeAssignment(zombie_eid, 1)
        assign[z2] = SpikeAssignment(zombie_eid, 2)
        occs_dict[zombie_eid] = zombie_occ

        # Re-map existing birthed_occ spikes after sorting
        for (eid, occ) in occs_dict
            eid == zombie_eid && continue
            empty!(occ.members)
            fill!(occ.counts, 0)
            for (idx, st) in enumerate(spikes_t)
                if abs(st - occ.tau) < 0.03
                    assign[idx] = SpikeAssignment(eid, 1)
                    push!(occ.members, idx)
                    occ.counts[1] += 1
                elseif abs(st - (occ.tau + 0.15)) < 0.03
                    assign[idx] = SpikeAssignment(eid, 2)
                    push!(occ.members, idx)
                    occ.counts[2] += 1
                end
            end
        end

        birthed_2, overwrote_2 = detect_and_birth_occurrences!(
            occs_dict, motifs_dict, spikes_t, spikes_chan, chan_to_spk,
            assign, bg_rates, 100.0, R_max; gamma_m=50.0, delta_margin=3.0, rng=rng
        )

        @test overwrote_2 >= 1
        @test haskey(occs_dict, zombie_eid)
        overwritten_occ = occs_dict[zombie_eid]
        println("Overwritten occurrence onset: ", round(overwritten_occ.tau, digits=4), " s (True 60.120 s)")
        @test isapprox(overwritten_occ.tau, 60.120, atol=0.015)
        # Verify latent coordinates cleanly reset to 0.0 (Safeguard 3)
        @test overwritten_occ.u == zeros(Float64, R_max)
        @test sum(overwritten_occ.counts) >= 12
        println("Latent coordinate cleanly reset to: ", overwritten_occ.u)
    end

    # -------------------------------------------------------------
    # TEST 11: LEVEL m=5 TASK REGIMES & KINEMATIC SUBSPACE
    # -------------------------------------------------------------
    @testset "Level m=5 Task Regimes & Kinematic Subspace" begin
        K = 2
        R = 2
        M = 2
        hyperstate = Level5Hyperstate(K, R, M; force_thresh=0.01)

        # Invariant 1: Scale Anchoring on Regime 1
        @test hyperstate.regimes[1].lambda_u == [1.0, 1.0]
        @test hyperstate.regimes[1].alpha_c == hyperstate.regimes[1].beta_c

        # Setup 2 motifs
        p1 = Packet(1, 1, 0.0, 0.0004, 2.0)
        p2 = Packet(2, 2, 0.1, 0.0004, 2.0)
        m1 = PurePacketMotif(1, [p1, p2], R, 2)
        m1.W[1, 1] = 0.02; m1.W[1, 2] = -0.02
        m1.W[2, 1] = 0.00; m1.W[2, 2] = 0.00 # Axis 2 has zero loading (dormant candidate)

        motifs = Dict(1 => m1)

        # Create occurrences:
        # Group 1 (10 occs): baseline vigor (c_e ~ 1.0), tight u_e near 0
        occs = Dict{Int, PureOccurrence}()
        for e in 1:10
            occ = PureOccurrence(e, 1, 10.0 + e, R, 2; amplitude=1.0)
            occ.counts = [2, 2]
            occ.sum_t = [occ.tau, occ.tau + 0.1]
            occ.u = [0.01 * randn(rng), 0.0]
            occs[e] = occ
        end

        # Group 2 (10 occs): high vigor (c_e ~ 2.5), wide u_e along Axis 1
        for e in 11:20
            occ = PureOccurrence(e, 1, 30.0 + e, R, 2; amplitude=2.5)
            occ.regime_id = 2
            occ.counts = [5, 5]
            # Residual pull along Axis 1
            occ.sum_t = [occ.tau + 0.01, occ.tau + 0.09] # pull on W[1, :]
            occ.u = [0.8 * randn(rng), 0.0]
            occs[e] = occ
        end

        # Set Regime 2 prior: higher vigor (beta_c = 4.0 vs alpha_c = 10.0 => E[c] = 2.5)
        hyperstate.regimes[2].beta_c = 4.0
        hyperstate.pi_m[1, :] = [0.5, 0.5]

        # Test Step 5A: resample occurrence regimes
        resample_occurrence_regimes!(hyperstate, occs, motifs; rng=rng)

        # High-vigor occurrences should predominantly choose Regime 2
        high_vigor_reg2 = count(e -> occs[e].regime_id == 2, 11:20)
        low_vigor_reg1 = count(e -> occs[e].regime_id == 1, 1:10)
        @test high_vigor_reg2 >= 7
        @test low_vigor_reg1 >= 7

        # Test Step 5B, 5C, 5D: sample task regimes
        sample_task_regimes!(hyperstate, occs, motifs; T_total=100.0, rng=rng)

        # Verify Regime 1 scale invariants preserved strictly
        @test hyperstate.regimes[1].lambda_u == [1.0, 1.0]
        @test hyperstate.regimes[1].beta_c == hyperstate.regimes[1].alpha_c

        # Verify Axis 2 is flagged as dormant in Regime 2 (zero loading -> zero force)
        @test hyperstate.regimes[2].active_dims[2] == false
        @test hyperstate.regimes[2].lambda_u[2] == hyperstate.lambda_dormant

        # Test Zero-Allocation in inner solve using regime precisions
        u_out = zeros(Float64, R)
        active_mask = trues(R)
        res = [0.005, -0.005]
        prec = [2500.0, 2500.0]
        allocs = check_coord_alloc(
            u_out, m1, active_mask, res, prec,
            hyperstate.regimes[2].lambda_u, rng
        )
        @test allocs == 0
        println("Zero-allocation solve with regime prior: ", allocs, " bytes")

        # Test Dynamic Dimensional Growth Hook R -> R+1
        m1.W = vcat(m1.W, [0.015 -0.015]) # New dimension 3 has active loadings
        extend_task_regimes_dimension!(hyperstate, motifs, occs, 3)
        @test hyperstate.R == 3
        @test length(hyperstate.regimes[1].lambda_u) == 3
        @test hyperstate.regimes[1].lambda_u[3] == 1.0
        @test length(hyperstate.regimes[2].lambda_u) == 3
        println("Dynamic growth R -> 3 successful; Regime 1 lambda_u: ", hyperstate.regimes[1].lambda_u)
    end

    # -------------------------------------------------------------
    # TEST 12: LEVEL m=5 DYNAMIC BIRTH, DEATH & SCALE GAUGE HANDOVER
    # -------------------------------------------------------------
    @testset "Level m=5 Dynamic Birth, Death & Scale Gauge Handover" begin
        rng = MersenneTwister(42)
        R = 2
        M = 2
        p1 = Packet(1, 1, 0.0, 0.0004, 2.0)
        p2 = Packet(2, 2, 0.1, 0.0004, 2.0)
        m1 = PurePacketMotif(1, [p1, p2], R, 2)
        m1.W[1, 1] = 0.02; m1.W[1, 2] = -0.02
        m1.W[2, 1] = 0.00; m1.W[2, 2] = 0.00
        motifs = Dict(1 => m1)

        # 1. Test Sequential Dynamic Birth (K -> K+1) from scratch (K=1)
        hyperstate = Level5Hyperstate(1, R, M; alpha_dp=1.0, N_min=2, force_thresh=0.05)
        @test hyperstate.K == 1

        occs_birth = Dict{Int, PureOccurrence}()
        # 10 baseline occurrences (N_e = 4 spikes vs 4 expected, c_e = 1.0)
        for e in 1:10
            occ = PureOccurrence(e, 1, 10.0 + e, R, 2; amplitude=1.0)
            occ.counts = [2, 2]
            occ.sum_t = [occ.tau, occ.tau + 0.1]
            occ.u = [0.01 * randn(rng), 0.0]
            occs_birth[e] = occ
        end
        # 10 massive high-vigor burst occurrences (N_e = 20 spikes vs 4 expected, c_e = 5.0)
        for e in 11:20
            occ = PureOccurrence(e, 1, 30.0 + e, R, 2; amplitude=5.0)
            occ.counts = [10, 10]
            occ.sum_t = [occ.tau + 0.01, occ.tau + 0.09]
            occ.u = [0.02 * randn(rng), 0.0]
            occs_birth[e] = occ
        end

        # Run sequential leave-one-out CRP
        resample_occurrence_regimes!(hyperstate, occs_birth, motifs; rng=rng)

        # Verify a 2nd regime was birthed dynamically
        @test hyperstate.K >= 2
        println("Dynamic sequential birth successful: K grew from 1 to ", hyperstate.K)
        # High-vigor burst occurrences should have coalesced into the new regime
        birthed_reg_id = occs_birth[11].regime_id
        @test birthed_reg_id >= 2
        high_vigor_coalesced = count(e -> occs_birth[e].regime_id == birthed_reg_id, 11:20)
        @test high_vigor_coalesced >= 8
        println("High-vigor occurrences coalesced into birthed Regime ", birthed_reg_id, ": ", high_vigor_coalesced, "/10")

        # 2. Test Candidate Staging & Lone Singleton Dissolution (N_min >= 2)
        # Create a lone singleton occurrence in Regime 2
        hyperstate2 = Level5Hyperstate(2, R, M; N_min=2)
        occs_single = Dict{Int, PureOccurrence}()
        for e in 1:15
            occ = PureOccurrence(e, 1, 10.0 + e, R, 2; amplitude=1.0)
            occ.regime_id = 1
            occ.counts = [2, 2]
            occ.sum_t = [occ.tau, occ.tau + 0.1]
            occ.u = [0.01 * randn(rng), 0.0]
            occs_single[e] = occ
        end
        # Occurrence 16 is a lone outlier
        lone_occ = PureOccurrence(16, 1, 50.0, R, 2; amplitude=1.2)
        lone_occ.regime_id = 2
        lone_occ.counts = [2, 2]
        lone_occ.sum_t = [lone_occ.tau, lone_occ.tau + 0.1]
        lone_occ.u = [0.01 * randn(rng), 0.0]
        occs_single[16] = lone_occ

        push!(hyperstate2.regimes[2].assigned_occurrences, 16)
        for e in 1:15
            push!(hyperstate2.regimes[1].assigned_occurrences, e)
        end

        # Run resample: lone candidate should dissolve and absorb into Regime 1
        resample_occurrence_regimes!(hyperstate2, occs_single, motifs; rng=rng)
        @test hyperstate2.K == 1
        @test !haskey(hyperstate2.regimes, 2)
        @test occs_single[16].regime_id == 1
        println("Lone singleton successfully dissolved; K compacted to 1.")

        # 3. Test Unregularized Data MLE Dormancy Test (S^{data}_{k,r} > 0.05)
        hyperstate3 = Level5Hyperstate(2, R, M; force_thresh=0.05)
        # Regime 2 has 10 occurrences with strong data residual drive along Axis 1, zero on Axis 2
        for e in 1:10
            occ = PureOccurrence(e, 1, 10.0 + e, R, 2; amplitude=1.0)
            occ.regime_id = 2
            occ.counts = [4, 4]
            # Mean timing res: (tau + 0.02) - tau = 0.02 on pkt 1, (tau - 0.02) - (tau + 0.1) = -0.12 on pkt 2
            occ.sum_t = [4.0 * (occ.tau + 0.02), 4.0 * (occ.tau + 0.08)]
            occ.u = [0.1, 0.0]
            push!(hyperstate3.regimes[2].assigned_occurrences, e)
            occs_single[e] = occ
        end
        sample_task_regimes!(hyperstate3, occs_single, motifs; T_total=100.0, rng=rng)
        @test hyperstate3.regimes[2].active_dims[1] == true
        @test hyperstate3.regimes[2].active_dims[2] == false
        @test hyperstate3.regimes[2].lambda_u[2] == hyperstate3.lambda_dormant
        println("Unregularized dormancy verified: Axis 1 active, Axis 2 clamped to dormant floor.")

        # 4. Test Exact Majority Scale-Gauge Handover & ARD Rescaling
        ard_state = ARDManifoldState(R; a0=1e-3, b0=1.0e-6)
        ard_state.alpha = [100.0, 200.0]

        # Setup: Regime 1 has 3 occurrences; Regime 2 has 12 occurrences (Regime 2 is majority)
        empty!(hyperstate3.regimes[1].assigned_occurrences)
        empty!(hyperstate3.regimes[2].assigned_occurrences)
        for e in 1:3
            occs_single[e].regime_id = 1
            push!(hyperstate3.regimes[1].assigned_occurrences, e)
        end
        for e in 4:15
            occs_single[e].regime_id = 2
            push!(hyperstate3.regimes[2].assigned_occurrences, e)
        end

        # Give Regime 2 non-trivial scale: lambda = 4.0, beta_c = 25.0 (alpha_c = 10.0 => E[c] = 0.4)
        hyperstate3.regimes[2].lambda_u = [4.0, 1000.0]
        hyperstate3.regimes[2].beta_c = 25.0

        # Snapshot pre-handover physical observables
        pre_shifts = [m1.W[1, 1] * occs_single[e].u[1] for e in 1:15]
        pre_poisson_rates = [occs_single[e].amplitude * m1.packets[1].mass for e in 1:15]

        # Execute Majority Scale-Gauge Handover
        k_crowned = maintain_majority_scale_gauge!(hyperstate3, occs_single, motifs, ard_state)
        @test k_crowned == 2

        # Invariant A: Newly crowned Regime 1 has unit precision and unit vigor expectation
        @test hyperstate3.regimes[1].lambda_u[1] == 1.0
        @test hyperstate3.regimes[1].beta_c == hyperstate3.regimes[1].alpha_c
        @test length(hyperstate3.regimes[1].assigned_occurrences) == 12

        # Invariant B: Exact physical temporal shifts W_{r,p} * u_{e,r} invariant to machine precision
        post_shifts = [m1.W[1, 1] * occs_single[e].u[1] for e in 1:15]
        max_shift_diff = maximum(abs.(post_shifts .- pre_shifts))
        @test max_shift_diff < 1e-12
        println("Physical temporal shift invariance: max diff = ", max_shift_diff)

        # Invariant C: Exact physical Poisson rates c_e * a_{m,p} invariant to machine precision
        post_poisson_rates = [occs_single[e].amplitude * m1.packets[1].mass for e in 1:15]
        max_rate_diff = maximum(abs.(post_poisson_rates .- pre_poisson_rates))
        @test max_rate_diff < 1e-12
        println("Physical Poisson rate invariance: max diff = ", max_rate_diff)

        # Invariant D: ARD precision correctly rescaled by lambda_{k_max, r} = 4.0
        @test isapprox(ard_state.alpha[1], 100.0 * 4.0, atol=1e-10)
        println("ARD precision alpha_1 exactly rescaled: 100.0 -> ", ard_state.alpha[1])

        # 5. Test Escobar-West Concentration Sampler
        initial_alpha_dp = hyperstate3.alpha_dp
        sample_dp_concentration!(hyperstate3; rng=rng)
        @test hyperstate3.alpha_dp > 0.01
        @test hyperstate3.alpha_dp <= 50.0
        println("Escobar-West alpha_dp update: ", round(initial_alpha_dp, digits=3), " -> ", round(hyperstate3.alpha_dp, digits=3))
    end

    # -------------------------------------------------------------
    # TEST 13: BIOPHYSICAL RAYLEIGH GATE & CONTINUOUS SNR PRUNING (RULES 1, 2, 3)
    # -------------------------------------------------------------
    @testset "Biophysical Rayleigh Gate & Continuous SNR Pruning (Rules 1, 2, 3)" begin
        # 1. Test Rule 1: Rayleigh / Biophysical Identifiability Gate for Splitting
        # Channel 10 has a packet at 50 ms (sigma = 5 ms)
        p1 = Packet(1, 10, 0.050, 0.000025, 1.0)
        m = PurePacketMotif(1, [p1], R_max, N_chans; W_window=1.0)
        E = 20
        occs = [PureOccurrence(e, 1, 10.0 + (e - 1) * 1.5, R_max, 1) for e in 1:E]

        spike_t = Float64[]
        spike_chan = Int[]
        assign = SpikeAssignment[]

        # Candidate spikes at 52 ms: Delta mu = 2.0 ms (< tau_refractory = 3.0 ms and < Rayleigh = 14 ms)
        spk_idx = 0
        for (e, occ) in enumerate(occs)
            for _ in 1:3
                spk_idx += 1
                push!(spike_t, occ.tau + 0.052 + 0.001 * randn(rng))
                push!(spike_chan, 10)
                push!(assign, SpikeAssignment(0, 0)) # background candidate spikes
            end
        end
        bg_rates = fill(0.5, N_chans)

        # Proposed candidate violates Rayleigh bimodality & refractory floor -> must be rejected
        birthed_unimodal = detect_and_birth_or_split_packets!(
            m, occs, spike_t, spike_chan, assign, bg_rates, N_chans;
            sigma_prior=0.005, alpha_a=1.0, beta_a=1.0, tau_refractory=0.003
        )
        @test birthed_unimodal == false
        @test length(m.packets) == 1
        println("Rule 1: Unimodal split proposal at Delta mu = 2.0 ms correctly rejected.")

        # Candidate spikes at 250 ms: Delta mu = 200 ms (>> Rayleigh) -> cleanly accepted
        spike_t2 = Float64[]
        spike_chan2 = Int[]
        assign2 = SpikeAssignment[]
        for (e, occ) in enumerate(occs)
            for _ in 1:4
                push!(spike_t2, occ.tau + 0.250 + 0.002 * randn(rng))
                push!(spike_chan2, 10)
                push!(assign2, SpikeAssignment(0, 0))
            end
        end
        birthed_bimodal = detect_and_birth_or_split_packets!(
            m, occs, spike_t2, spike_chan2, assign2, bg_rates, N_chans;
            sigma_prior=0.005, alpha_a=1.0, beta_a=1.0, tau_refractory=0.003
        )
        @test birthed_bimodal == true
        @test length(m.packets) == 2
        @test isapprox(m.packets[2].mu, 0.250, atol=0.015)
        println("Rule 1: Genuinely bimodal candidate at Delta mu = 200 ms successfully birthed: mu = ", round(m.packets[2].mu, digits=4))

        # 2. Test Rule 3: Continuous Packet Pruning (SNR <= 1.0)
        # Create motif with 3 packets:
        # Packet 1 (Channel 1): Mass 0.08, sigma = 0.005, bg = 0.5 -> peak = 6.38 Hz, SNR = 12.7 (Survives)
        # Packet 2 (Channel 2): Mass 0.01, sigma = 0.005, bg = 5.0 -> peak = 0.80 Hz, SNR = 0.16 (Dead, Parasitic!)
        # Packet 3 (Channel 3): Mass 0.06, sigma = 0.005, bg = 0.5 -> peak = 4.79 Hz, SNR = 9.57 (Survives)
        pk1 = Packet(1, 1, 0.000, 0.000025, 0.08)
        pk2 = Packet(2, 2, 0.100, 0.000025, 0.01)
        pk3 = Packet(3, 3, 0.200, 0.000025, 0.06)
        m3 = PurePacketMotif(2, [pk1, pk2, pk3], R_max, N_chans; W_window=1.0)

        occs3 = [PureOccurrence(1, 2, 50.0, R_max, 3)]
        occs3[1].counts[1] = 5
        occs3[1].counts[2] = 1 # 1 accidental spike on parasite packet
        occs3[1].counts[3] = 4

        assign3 = [
            SpikeAssignment(1, 1), # Spike 1 on Packet 1
            SpikeAssignment(1, 2), # Spike 2 on Packet 2 (to be evicted)
            SpikeAssignment(1, 3)  # Spike 3 on Packet 3
        ]
        occs3[1].members = [1, 2, 3]

        custom_bg = fill(0.5, N_chans)
        custom_bg[2] = 5.0 # High baseline noise on channel 2

        pre_mass = sum(p.mass for p in m3.packets)
        pruned_pkts = prune_dead_packets!(m3, occs3, custom_bg, N_chans; assign=assign3)

        @test 2 in pruned_pkts
        @test length(m3.packets) == 2
        # Invariant A: Parasitic packet 2 dead and evicted
        @test assign3[2] == SpikeAssignment(0, 0)
        @test assign3[1].packet_idx == 1
        @test assign3[3].packet_idx == 2
        @test occs3[1].members == [1, 3]

        # Invariant B: Total motif mass atomically reduced
        post_mass = sum(p.mass for p in m3.packets)
        @test isapprox(post_mass, pre_mass - 0.01, atol=1e-12)
        println("Rule 3: Parasitic packet (SNR = 0.16) pruned. Spike 2 evicted to background. Mass: ", pre_mass, " -> ", post_mass)
    end

    # -------------------------------------------------------------
    # TEST 14: TIME-RESCALING THEOREM & MULTI-CHAIN RANK-NORMALIZED R-HAT / ESS
    # -------------------------------------------------------------
    @testset "Time-Rescaling Theorem & Multi-Chain Rank-Normalized R-hat / ESS" begin
        # 1. Stratified Audit Channel Selection
        m_test = PurePacketMotif(1, [
            Packet(1, 5, 0.050, 0.000025, 1.0),
            Packet(2, 5, 0.250, 0.000025, 1.0),
            Packet(3, 8, 0.100, 0.000025, 1.0)
        ], R_max, 50; W_window=1.0)
        test_motifs = Dict(1 => m_test)
        test_bg = fill(0.5, 50)
        
        audit = select_stratified_audit_channels(test_motifs, test_bg, 50; n_signal=2, n_quiet=3)
        @test audit.signal_channels[1] == 5 # Most packets (2 packets)
        @test audit.signal_channels[2] == 8 # Second most (1 packet)
        @test length(audit.quiet_channels) == 3
        for q in audit.quiet_channels
            @test length(m_test.chan_to_packets[q]) == 0
        end
        println("Stratified audit channel selection verified: Signal = ", audit.signal_channels, " | Quiet = ", audit.quiet_channels)

        # 2. Time-Rescaling Theorem Closed-Form KS Test on Ground-Truth Poisson Process
        # Generate true Poisson spike train with rate lambda_0 = 2.0 Hz over 1000s (~2000 spikes)
        lam_true = 2.0
        t_spk = Float64[]
        curr_t = 0.0
        while curr_t < 1000.0
            curr_t += rand(rng, Exponential(1.0 / lam_true))
            curr_t < 1000.0 && push!(t_spk, curr_t)
        end
        chan_spks = [Int[] for _ in 1:50]
        chan_spks[20] = collect(1:length(t_spk))
        empty_occs = Dict{Int, PureOccurrence}()
        empty_motifs = Dict{Int, PurePacketMotif}()
        bg_vector = fill(lam_true, 50)
        act_mask = falses(4)

        # Evaluate Time-Rescaling KS test under true model intensity lambda_0
        res = compute_time_rescaling_ks(empty_motifs, empty_occs, t_spk, chan_spks, bg_vector, [20], act_mask)
        @test res.is_valid == true
        @test res.D_ks <= res.D_95
        println(@sprintf("Time-Rescaling KS on true Poisson: D_KS = %.5f <= 95%% Threshold = %.5f (PASS: Inside Envelope)", res.D_ks, res.D_95))

        # 3. Multi-Chain Rank-Normalized R-hat and Bulk-ESS (Vehtari et al. 2021)
        # Case A: Two stationary, well-mixed chains drawn from same stationary distribution
        chain_a = randn(rng, 100)
        chain_b = randn(rng, 100)
        res_mixed = compute_rank_normalized_rhat_ess([chain_a, chain_b])
        @test res_mixed.rhat < 1.05
        @test res_mixed.ess > 50.0
        println(@sprintf("Rank-Normalized R-hat on stationary chains: R_hat = %.4f (< 1.05) | Bulk-ESS = %.1f (PASS)", res_mixed.rhat, res_mixed.ess))

        # Case B: Two non-converged, separated chains (shifted means)
        chain_c = randn(rng, 100) .+ 5.0 # Separated mode
        res_separated = compute_rank_normalized_rhat_ess([chain_a, chain_c])
        @test res_separated.rhat > 1.20
        println(@sprintf("Rank-Normalized R-hat on separated chains:  R_hat = %.4f (> 1.20 correctly flags non-convergence)", res_separated.rhat))

        # 4. Joint Log-Posterior Energy U_t Computation
        total_spks_chan = [100, 200]
        bg_rates_test = [1.0, 2.0]
        ard_st_test = ARDManifoldState(2; a0=1e-3, b0=1e-6)
        ard_st_test.active_mask[1] = true
        ard_st_test.alpha[1] = 100.0
        hyper_test = Level5Hyperstate(1, 2, 1)
        p_toy = Packet(1, 1, 0.05, 0.0001, 0.8)
        m_toy = PurePacketMotif(1, [p_toy], 2, 2)
        m_toy.W[1, 1] = 0.01
        toy_motifs = Dict(1 => m_toy)
        occ_toy = PureOccurrence(1, 1, 10.0, 2, 1; amplitude=1.0)
        occ_toy.counts[1] = 5
        occ_toy.sum_t[1] = 5.0 * 10.05
        occ_toy.sum_t2[1] = 5.0 * (10.05^2 + 0.0001)
        toy_occs = Dict(1 => occ_toy)

        energy = compute_joint_energy(
            toy_motifs, toy_occs, total_spks_chan, bg_rates_test,
            ard_st_test, hyper_test, 100.0
        )
        @test isfinite(energy)
        @test energy > 0.0
        println(@sprintf("Joint Log-Posterior Energy U_t: %.4f (finite, positive)", energy))
    end

    # -------------------------------------------------------------
    # TEST 15: DYNAMIC MOTIF BOUNDING ENVELOPES & BACKWARD COMPATIBILITY
    # -------------------------------------------------------------
    @testset "Dynamic Motif Bounding Envelopes (AABB) & Backward Compatibility" begin
        # 1. Degenerate empty motif guard
        m_empty = PurePacketMotif(99, Packet[], 2, 2)
        @test m_empty.delta_left == 0.0
        @test m_empty.delta_right == 0.0

        # 2. Exact envelope bounds without deformation
        # Packet 1 at mu=0, sigma=10ms (var=0.0001)
        # Packet 2 at mu=350ms, sigma=20ms (var=0.0004)
        p1 = Packet(1, 1, 0.0, 0.0001, 1.0)
        p2 = Packet(2, 2, 0.350, 0.0004, 1.0)
        m_dyn = PurePacketMotif(1, [p1, p2], 2, 2)
        @test isapprox(m_dyn.delta_left, -0.040, atol=1e-5)
        @test isapprox(m_dyn.delta_right, 0.430, atol=1e-5)

        # 3. Manifold deformation with Level 5 regime precision floor
        m_dyn.W[1, 1] = 0.015 # 15 ms loading on axis 1
        # Baseline regime (lambda_1 = 1.0): delta_p = 3.0 * 15ms = 45ms -> left = -40ms - 45ms = -85ms
        update_motif_bounds!(m_dyn, [1.0, 1.0])
        @test isapprox(m_dyn.delta_left, -0.085, atol=1e-5)

        # Ballistic regime (lambda_min = 0.25): delta_p = 3.0 * (15ms / 0.5) = 90ms -> left = -40ms - 90ms = -130ms
        update_motif_bounds!(m_dyn, [0.25, 1.0])
        @test isapprox(m_dyn.delta_left, -0.130, atol=1e-5)
        println("Dynamic envelope verified: Baseline delta_left = -85ms, Ballistic delta_left = -130ms")

        # 4. Channel-Local Bounding Optimization Tests
        @test m_dyn.chan_active[1] == true
        @test m_dyn.chan_active[2] == true
        # Channel 1 local bounds: Packet 1 with ballistic deformation -> [-130ms, +130ms]
        @test isapprox(m_dyn.chan_bounds[1][1], -0.130, atol=1e-5)
        @test isapprox(m_dyn.chan_bounds[1][2], 0.130, atol=1e-5)
        # Channel 2 local bounds: Packet 2 (mu=350ms, sigma=20ms) -> [270ms, 430ms] (tight 160ms window!)
        @test isapprox(m_dyn.chan_bounds[2][1], 0.270, atol=1e-5)
        @test isapprox(m_dyn.chan_bounds[2][2], 0.430, atol=1e-5)
        println("Channel-local bounding verified: Chan 1 span = 260ms, Chan 2 local span = 160ms (decoupled from Chan 1)")

        # 5. Checkpoint backward compatibility
        ck_path = joinpath(@__DIR__, "checkpoint_pure_packet_scratch_50sweeps.jls")
        if isfile(ck_path)
            ck_loaded = open(deserialize, ck_path)
            @test length(ck_loaded.motifs) == 9
            for (mid, m) in ck_loaded.motifs
                @test m.delta_left < 0.0
                @test m.delta_right > 0.0
                @test length(m.chan_active) == length(m.chan_to_packets)
                @test length(m.chan_bounds) == length(m.chan_to_packets)
            end
            println("Checkpoint backward compatibility: Successfully loaded 50-sweep scratch checkpoint with channel bounds intact.")
        end
    end

    # =====================================================================
    # 13. EXACT CONTINUOUS-TIME CTMC UNIFORMIZATION & REGION-RANK-1 BASELINE
    # =====================================================================
    @testset "Exact Continuous-Time CTMC Uniformization & Region-Rank-1 Baseline" begin
        rng = MersenneTwister(1234)
        K = 3
        N_chans = 426
        W_min = 0.25
        alpha_q = 2.0
        T_total = 2114.0

        # 1. CTMC Initialization & Structural Nyquist Carrier Bound
        ctmc = ContinuousTimeCTMC(K, N_chans; W_min=W_min, alpha_q=alpha_q, T_total=T_total, cp_range=99:301)
        @test ctmc.K == 3
        @test isapprox(ctmc.beta_q, (K - 1) * alpha_q * W_min) # (3-1)*2*0.25 = 1.0 s
        # Expected total exit rate must equal 1 / W_min = 4.0 Hz
        expected_exit = sum(ctmc.Q[1, j] for j in 2:3)
        @test isapprox(expected_exit, 1.0 / W_min) # 4.0 Hz
        @test isapprox(ctmc.Q[1, 1], -4.0)
        # Verify regional mapping: MOp (1..98, 302..426), CP (99..301)
        @test ctmc.channel_region[1] == 1
        @test ctmc.channel_region[98] == 1
        @test ctmc.channel_region[99] == 2
        @test ctmc.channel_region[301] == 2
        @test ctmc.channel_region[302] == 1
        @test count(==(2), ctmc.channel_region) == 203
        @test count(==(1), ctmc.channel_region) == 223

        # 2. Auxiliary Candidate Times T_cand ~ Poisson(Omega)
        times = sample_candidate_times!(ctmc, T_total; rng=rng)
        @test issorted(times)
        @test times[1] == 0.0
        @test times[end] == T_total
        @test length(times) > 1000

        # 3. Two-Pointer Background Spike Accumulation
        N_spk = 10_000
        mock_t = sort(rand(rng, N_spk) .* T_total)
        mock_neurons = rand(rng, 1:N_chans, N_spk)
        mock_assign = [SpikeAssignment(0, 0) for _ in 1:N_spk]
        B = length(times) - 1
        N_mop = zeros(Int, B)
        N_cp = zeros(Int, B)
        accumulate_interval_background_counts!(N_mop, N_cp, times, mock_t, mock_neurons, mock_assign, ctmc.channel_region)
        total_counted = sum(N_mop) + sum(N_cp)
        @test total_counted == N_spk
        println("Linear O(N + B) spike accumulation verified: ", total_counted, " / ", N_spk, " spikes exact match.")

        # 4. FFBS State Trajectory and Occurrence Coupling
        hyperstate = Level5Hyperstate(K, 2, 5)
        occs = Dict{Int, PureOccurrence}()
        for e in 1:10
            occ = PureOccurrence(e, 1, 100.0 * e, 2, 2; amplitude=1.0)
            occ.counts = [5, 5]
            occ.sum_t = [5.0 * occ.tau, 5.0 * (occ.tau + 0.05)]
            occs[e] = occ
        end
        p1 = Packet(1, 1, 0.0, 0.0001, 1.0)
        p2 = Packet(2, 2, 0.05, 0.0001, 1.0)
        motifs = Dict(1 => PurePacketMotif(1, [p1, p2], 2, N_chans))
        bg_rates = fill(2.0, N_chans)

        sample_ctmc_state_trajectory!(
            ctmc, hyperstate, occs, motifs,
            mock_t, mock_neurons, mock_assign, bg_rates, T_total;
            rng=rng
        )
        @test length(ctmc.cand_states) == length(ctmc.cand_times) - 1
        @test all(1 <= s <= K for s in ctmc.cand_states)
        # Verify occurrence coupling k_e == S(tau_e)
        for occ in values(occs)
            b_occ = searchsortedlast(ctmc.cand_times, occ.tau)
            b_occ = clamp(b_occ, 1, B)
            @test occ.regime_id == ctmc.cand_states[b_occ]
        end
        # Verify dwell time partition sums to T_total
        @test isapprox(sum(ctmc.T_k), T_total, atol=1e-5)
        println("Exact FFBS continuous trajectory verified: sum(T_k) = ", round(sum(ctmc.T_k), digits=2), " s (Exact T_total)")

        # 5. Triad Scale Anchor & Conjugate Updates
        update_ctmc_parameters!(
            ctmc, hyperstate, occs, motifs,
            bg_rates, mock_neurons, mock_assign, T_total;
            rng=rng
        )
        # Triad anchor checks
        @test ctmc.gamma_tone[1, 1] == 1.0 # MOp Regime 1
        @test ctmc.gamma_tone[2, 1] == 1.0 # CP Regime 1
        @test hyperstate.regimes[1].beta_c == hyperstate.regimes[1].alpha_c # E[c | 1] == 1.0
        @test hyperstate.regimes[1].lambda_u == [1.0, 1.0] # Lambda_{u,1} == I_R
        # Generator matrix rows sum to 0.0
        for k in 1:K
            @test isapprox(sum(ctmc.Q[k, :]), 0.0, atol=1e-10)
            @test ctmc.Q[k, k] < 0.0
        end
        # Uniformization rate Omega strictly dominates max exit rate
        max_exit = maximum(abs(ctmc.Q[k, k]) for k in 1:K)
        @test ctmc.Omega > max_exit
        println("Triad Scale Gauge verified: Regime 1 strictly anchored; Omega = ", round(ctmc.Omega, digits=4), " > max |q_kk| = ", round(max_exit, digits=4))

        # 6. Cramér-Rao Manifold Pruning Default
        ard_st = ARDManifoldState(4)
        @test ard_st.prune_threshold == 6500.0
        println("Cramér-Rao manifold pruning bound verified: alpha_prune = 6500.0 s^-2")
    end

    @testset "Continuous OU Gauge & Anti-Cannibalization Invariants" begin
        rng = MersenneTwister(42)
        T_mock = 50.0 # 50 seconds
        C_mock = 20 # 20 channels
        ctx = ContinuousOUContext(
            C_mock, T_mock;
            cp_range=11:20,
            dt_net=0.1, dt_cell=1.0, dt_motif=1.0,
            tau_net=0.5, tau_cell=5.0, tau_motif=5.0,
            sigma_net=0.5, sigma_cell=0.5, sigma_motif=0.5,
            alpha_c=10.0
        )

        # 1. Zero-Mean Gauge Invariant under Asymmetric Spiking
        N_spk = 2000
        mock_t = sort(rand(rng, N_spk) .* T_mock)
        mock_chan = rand(rng, 1:C_mock, N_spk)
        # Create asymmetric spikes: concentrated in first half of session
        for i in 1:1500
            mock_t[i] = rand(rng) * (T_mock * 0.3)
        end
        sort!(mock_t)
        mock_assign = [SpikeAssignment(0, 0) for _ in 1:N_spk]
        bg_rates = fill(2.0, C_mock)

        # Run Newton-Raphson OU updates
        update_regional_ou!(ctx, mock_assign, mock_t, mock_chan, bg_rates; max_iter=3)
        update_cell_ou!(ctx, mock_assign, mock_t, mock_chan, bg_rates; max_iter=3)

        # Check regional zero-mean gauge
        for reg in 1:2
            mean_xr = mean(ctx.x_region[reg, :])
            @test isapprox(mean_xr, 0.0, atol=1e-12)
            @test all(-2.5 .<= ctx.x_region[reg, :] .<= 2.5)
        end

        # Check cell zero-mean gauge
        for c in 1:C_mock
            mean_dxc = mean(ctx.delta_x[c, :])
            @test isapprox(mean_dxc, 0.0, atol=1e-12)
            @test all(-2.5 .<= ctx.delta_x[c, :] .<= 2.5)
        end

        # Check motif zero-mean gauge
        mock_motifs = Dict{Int, PurePacketMotif}(
            1 => PurePacketMotif(1, [Packet(1, 1, 0.2, 0.015^2, 1.0)], C_mock, 1; W_window=1.0)
        )
        mock_occs = Dict{Int, PureOccurrence}(
            1 => PureOccurrence(1, 1, 5.0, 1, 1; amplitude=1.0, regime_id=1)
        )
        mock_hyperstate = Level5Hyperstate(1, 1, 1)
        update_motif_ou!(ctx, mock_occs, mock_motifs, mock_hyperstate; max_iter=3)
        mean_ym = mean(ctx.y_motif[1, :])
        @test isapprox(mean_ym, 0.0, atol=1e-12)
        @test all(-2.5 .<= ctx.y_motif[1, :] .<= 2.5)
        println("OU Zero-mean gauge invariant verified: all means < 1e-12 and within [-2.5, 2.5]")

        # 2. Baseline Exposure T_eff & Background Rate Non-Deflation Invariant
        counts_bg = zeros(Int, C_mock)
        update_continuous_background_rates!(
            bg_rates, ctx, counts_bg, mock_assign, mock_chan, T_mock;
            alpha_bg=1.0, beta_bg=1.0, rng=rng
        )
        # Verify baseline rates did not deflate to zero
        @test all(bg_rates .> 0.1)
        @test mean(bg_rates) > 0.5
        println("Continuous background non-deflation verified: mean rate = ", round(mean(bg_rates), digits=3), " Hz (> 0.5 Hz)")

        # 3. Leg 2 Motor Vigor Anchor Invariant
        diverse_occs = Dict{Int, PureOccurrence}(
            i => PureOccurrence(i, 1, Float64(i), 1, 1; amplitude=0.5 + 0.1*i, regime_id=1)
            for i in 1:10
        )
        update_vigor_shape!(ctx, diverse_occs)
        @test ctx.alpha_c == 10.0
        println("Motor vigor anchor invariant verified: alpha_c strictly anchored at 10.00")
    end
end
