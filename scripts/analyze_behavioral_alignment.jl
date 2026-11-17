# analyze_behavioral_alignment.jl
# Cross-correlates native occurrence timestamps tau_e of all 20 motifs with behavioral telemetry

using PurePacketPPSeq
using Serialization, Statistics, Printf, JSON, Random

# 1. Load Converged Checkpoint (Sweep 100)
ck_path = joinpath(@__DIR__, "checkpoint_pure_packet_cold_start_100sweeps.jls")
raw = open(ck_path, "r") do io
    deserialize(io)
end
println("Loaded Model Checkpoint: $(length(raw.motifs)) Motifs, $(length(raw.occurrences)) Occurrences.")

# 2. Load Behavioral Telemetry
beh_path = "/home/foaleng/Downloads/motor_cortex_top4/lgcy0288__20230519_114115__ephys/behavior_aligned.json"
beh = JSON.parsefile(beh_path)
println("Loaded Behavioral Telemetry: $(length(beh["trials"])) Trials, $(length(beh["pokes"])) Pokes, $(length(beh["reward_state_entries"])) Reward Entries.")

T_total = 2114.2852511102637

# Extract Behavioral Event Timestamps
trial_starts = Float64[t["start_s"] for t in beh["trials"]]
trial_ends   = Float64[t["end_s"] for t in beh["trials"]]

# Target sequence is 2 -> 1 -> 6 -> 3 -> 7
pokes_p2 = Float64[p["time_task_s"] for p in beh["pokes"] if p["port"] == 2]
pokes_p1 = Float64[p["time_task_s"] for p in beh["pokes"] if p["port"] == 1]
pokes_p6 = Float64[p["time_task_s"] for p in beh["pokes"] if p["port"] == 6]
pokes_p3 = Float64[p["time_task_s"] for p in beh["pokes"] if p["port"] == 3]
pokes_p7 = Float64[p["time_task_s"] for p in beh["pokes"] if p["port"] == 7]

rewards_final = Float64[r["time_task_s"] for r in beh["reward_state_entries"] if r["state"] == "reward_poke_4"]
rewards_init  = Float64[r["time_task_s"] for r in beh["reward_state_entries"] if r["state"] == "reward_poke_0"]

event_types = [
    ("Trial Start", trial_starts),
    ("Port 2 (Init)", pokes_p2),
    ("Port 1", pokes_p1),
    ("Port 6", pokes_p6),
    ("Port 3", pokes_p3),
    ("Port 7 (Term)", pokes_p7),
    ("Reward Delivery", rewards_final)
]

# 3. PETH Parameters: [-1.0s, +1.0s], bin = 25ms (80 bins)
win_left = -1.0
win_right = 1.0
dt_bin = 0.025
lags = collect(win_left:dt_bin:win_right)
n_bins = length(lags) - 1
bin_centers = [0.5 * (lags[i] + lags[i+1]) for i in 1:n_bins]

function compute_peth(occ_taus::Vector{Float64}, event_times::Vector{Float64})
    counts = zeros(Float64, n_bins)
    n_events_valid = 0
    sort_taus = sort(occ_taus)
    for t_ev in event_times
        # require full observation window
        if t_ev + win_left < 0.0 || t_ev + win_right > T_total
            continue
        end
        n_events_valid += 1
        i_lo = searchsortedfirst(sort_taus, t_ev + win_left)
        i_hi = searchsortedlast(sort_taus, t_ev + win_right)
        for idx in i_lo:i_hi
            dt = sort_taus[idx] - t_ev
            bin_idx = floor(Int, (dt - win_left) / dt_bin) + 1
            if 1 <= bin_idx <= n_bins
                counts[bin_idx] += 1.0
            end
        end
    end
    n_events_valid == 0 && return (zeros(n_bins), 0.0, 0.0)
    # Rate in Hz
    rate = counts ./ (n_events_valid * dt_bin)
    return (rate, n_events_valid)
end

println("\n" * "="^125)
@printf("%-6s | %-5s | %-15s | %-12s | %-10s | %-10s | %-10s | %-8s | %-16s\n",
        "Motif", "Occs", "Best Event Target", "Base Rate", "Peak Rate", "Mod Ratio", "Peak Lag", "p-value", "Functional Role")
println("-"^125)

rng = Random.Xoshiro(42)
N_SHIFTS = 200

for (mid, pm) in sort(collect(raw.motifs), by=x->x[1])
    occ_taus = Float64[occ.tau for occ in values(raw.occurrences) if occ.motif == mid]
    n_occs = length(occ_taus)
    if n_occs == 0
        @printf("M%-4d | %5d | %-15s | %-12s | %-10s | %-10s | %-10s | %-8s | %-16s\n",
                mid, 0, "None", "0.00 Hz", "0.00 Hz", "1.0x", "0.0 ms", "1.000", "Dead / Unused")
        continue
    end
    base_rate = n_occs / T_total

    best_target = ""
    best_mod = 1.0
    best_peak_rate = 0.0
    best_peak_lag = 0.0
    best_p_val = 1.0

    for (ev_name, ev_times) in event_types
        rate, n_v = compute_peth(occ_taus, ev_times)
        n_v == 0 && continue
        peak_rate, max_idx = findmax(rate)
        mod_ratio = peak_rate / max(base_rate, 1e-4)

        if mod_ratio > best_mod
            best_mod = mod_ratio
            best_target = ev_name
            best_peak_rate = peak_rate
            best_peak_lag = bin_centers[max_idx]

            # Circular Shift Null Test for this candidate target
            observed_stat = peak_rate
            null_stats = Float64[]
            for _ in 1:N_SHIFTS
                shift = rand(rng) * (T_total - 20.0) + 10.0
                shifted_taus = [(tau + shift) % T_total for tau in occ_taus]
                sh_rate, _ = compute_peth(shifted_taus, ev_times)
                push!(null_stats, maximum(sh_rate))
            end
            best_p_val = (1.0 + count(s -> s >= observed_stat, null_stats)) / (N_SHIFTS + 1)
        end
    end

    # Determine Functional Role
    role = if best_p_val > 0.05 || best_mod < 1.35
        "Spontaneous / Inter-Trial"
    elseif occursin("Trial Start", best_target) || (occursin("Port 2", best_target) && abs(best_peak_lag) <= 0.25)
        "Task Initiation"
    elseif occursin("Reward", best_target) && best_peak_lag >= -0.1
        "Reward Consumption"
    elseif occursin("Port 7", best_target) && best_peak_lag >= -0.2
        "Sequence Termination"
    elseif occursin("Port", best_target)
        "Sequence Execution"
    else
        "Behavior-Aligned"
    end

    @printf("M%-4d | %5d | %-15s | %8.3f Hz | %8.3f Hz | %7.2fx   | %+7.1f ms | %7.4f  | %-16s\n",
            mid, n_occs, best_target, base_rate, best_peak_rate, best_mod, best_peak_lag * 1000, best_p_val, role)
end
println("="^125)
