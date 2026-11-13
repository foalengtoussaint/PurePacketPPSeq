# test/test_packet_drift.jl
# Direct test of Level m=1 packet birth/death stationarity and noise specificity
# Includes high-recurrence (Em = 1200) stress test with bursty noise (CV > 1) and sub-threshold modulation (SNR = 0.40)

using Test
using Random
using Statistics
using Distributions
using Printf
using PurePacketPPSeq

println("=================================================================")
println("STARTING LEVEL m=1 PACKET STATIONARITY & NOISE SPECIFICITY TEST")
println("=================================================================")

@testset "Level m=1 Packet Stationarity & Biophysical SNR Truncation" begin
    # -----------------------------------------------------------------
    # Part 1: Baseline Poisson Noise (Em = 50, 45 noise channels)
    # -----------------------------------------------------------------
    println("\n--- Part 1: Baseline Poisson Noise (Em = 50, 45 noise channels) ---")
    rng = Random.MersenneTwister(42)
    N_chans = 50
    T_total = 100.0
    bg_rate = 2.0
    sigma_prior = 0.015
    var_prior = sigma_prior^2

    true_latencies = [0.00, 0.05, 0.10, 0.15, 0.20]
    P_true = length(true_latencies)
    E_true = 50
    occ_times = sort(rand(rng, E_true) .* (T_total - 1.0) .+ 0.5)

    spike_t = Float64[]
    spike_chan = Int[]

    for tau in occ_times
        for (p, mu) in enumerate(true_latencies)
            if rand(rng) < 0.8
                push!(spike_t, tau + mu + 0.003 * randn(rng))
                push!(spike_chan, p)
            end
        end
    end

    for c in 1:N_chans
        n_spk = rand(rng, Poisson(bg_rate * T_total))
        times = sort(rand(rng, n_spk) .* T_total)
        append!(spike_t, times)
        append!(spike_chan, fill(c, n_spk))
    end

    p_sort = sortperm(spike_t)
    spike_t = spike_t[p_sort]
    spike_chan = spike_chan[p_sort]
    data = SpikeDataset(spike_t, spike_chan; N_channels=N_chans, T_total=T_total)

    init_packets = [Packet(p, p, true_latencies[p], var_prior, 0.8) for p in 1:P_true]
    m = PurePacketMotif(1, init_packets, 1, N_chans; W_window=0.40)
    motifs = Dict(1 => m)
    model = initialize_scratch_model(data, motifs; R_init=1)
    for (e, tau) in enumerate(occ_times)
        model.occurrences[e] = PureOccurrence(e, 1, tau, 1, P_true)
    end

    fit!(model, data; sweeps=10, allow_motif_birth=false, frozen_motif_ids=Set{Int}())

    m_final = model.motifs[1]
    noise_pkts = count(p -> p.channel > 5, m_final.packets)
    signal_pkts = count(p -> p.channel <= 5, m_final.packets)
    @test noise_pkts == 0
    @test signal_pkts == 5
    println("Part 1 PASS: Signal packets = $signal_pkts, Noise packets = $noise_pkts")

    # -----------------------------------------------------------------
    # Part 2: High-Recurrence Stress Test (Em = 1200, CV > 1, SNR = 0.40)
    # -----------------------------------------------------------------
    println("\n--- Part 2: High-Recurrence Stress Test (Em = 1200, CV > 1, SNR = 0.40) ---")
    rng2 = Random.MersenneTwister(123)
    N_chans2 = 20
    T_total2 = 300.0
    E_stress = 1200
    occ_times2 = sort(rand(rng2, E_stress) .* (T_total2 - 2.0) .+ 1.0)

    spike_t2 = Float64[]
    spike_chan2 = Int[]

    # True signal packets on channels 1..5 (SNR >> 1.0)
    for tau in occ_times2
        for (p, mu) in enumerate(true_latencies)
            if rand(rng2) < 0.8
                push!(spike_t2, tau + mu + 0.003 * randn(rng2))
                push!(spike_chan2, p)
            end
        end
    end

    # Weak sub-threshold rate modulation on channels 6..10 (SNR = 0.40, a_sub = 0.4 * sqrt(2pi) * sigma * lam0)
    a_sub = 0.40 * sqrt(2π) * sigma_prior * bg_rate
    for tau in occ_times2
        for c in 6:10
            if rand(rng2) < a_sub
                push!(spike_t2, tau + 0.12 + 0.003 * randn(rng2))
                push!(spike_chan2, c)
            end
        end
    end

    # Bursty noise background (Gamma renewal with shape = 0.5, CV = sqrt(1/0.5) = 1.41 > 1)
    for c in 1:N_chans2
        t = 0.0
        rate_scale = 1.0 / (bg_rate * 0.5)
        while t < T_total2
            dt = rand(rng2, Gamma(0.5, rate_scale))
            t += dt
            if t < T_total2
                push!(spike_t2, t)
                push!(spike_chan2, c)
            end
        end
    end

    p_sort2 = sortperm(spike_t2)
    spike_t2 = spike_t2[p_sort2]
    spike_chan2 = spike_chan2[p_sort2]
    data2 = SpikeDataset(spike_t2, spike_chan2; N_channels=N_chans2, T_total=T_total2)

    init_packets2 = [Packet(p, p, true_latencies[p], var_prior, 0.8) for p in 1:P_true]
    m2 = PurePacketMotif(1, init_packets2, 1, N_chans2; W_window=0.40)
    motifs2 = Dict(1 => m2)
    model2 = initialize_scratch_model(data2, motifs2; R_init=1)
    for (e, tau) in enumerate(occ_times2)
        model2.occurrences[e] = PureOccurrence(e, 1, tau, 1, P_true)
    end

    fit!(model2, data2; sweeps=10, allow_motif_birth=false, frozen_motif_ids=Set{Int}())

    m2_final = model2.motifs[1]
    noise_pkts2 = count(p -> p.channel > 5, m2_final.packets)
    sub_pkts2 = count(p -> (p.channel in 6:10), m2_final.packets)
    signal_pkts2 = count(p -> p.channel <= 5, m2_final.packets)
    @test noise_pkts2 == 0
    @test sub_pkts2 == 0
    @test signal_pkts2 == 5
    println("Part 2 PASS: Signal packets = $signal_pkts2, Sub-threshold packets = $sub_pkts2, Noise packets = $noise_pkts2")
end

println("=================================================================")
println("ALL PACKET DRIFT & NOISE SPECIFICITY TESTS PASSED!")
println("=================================================================")
