# benchmark_step11_speed.jl
# Measures the execution time of Step 11 after inverted background packet search optimization

include("/home/foaleng/Downloads/mop_awake_ppseq/custom_sampler/regime_triad_sampler/run_regime_sampler.jl")
using PurePacketPPSeq
using Printf, Serialization

data = load_awake_spikes()
ck_path = joinpath(@__DIR__, "checkpoint_pure_packet_cold_start_100sweeps.jls")
model = resume_model_from_checkpoint(ck_path, data; R_max=4)

println("Benchmarking Step 11 (detect_and_birth_or_split_packets!) across all $(length(model.motifs)) motifs...")



t_total_start = time()
for (mid, pm) in sort(collect(model.motifs), by=x->x[1])
    occ_list = [occ for occ in values(model.occurrences) if occ.motif == mid]
    isempty(occ_list) && continue
    t0 = time()
    PurePacketPPSeq.detect_and_birth_or_split_packets!(
        pm, occ_list, data.t, data.neuron, model.assign, model.bg_rates, data.N_channels;
        chan_spikes=data.chan_to_spikes, ou_ctx=model.ou_ctx, sigma_prior=0.015,
        alpha_a=1.0, beta_a=1.0,
        active_mask=model.ard_state.active_mask, ard_alphas=model.ard_state.alpha
    )
    dt = time() - t0
    @printf("  Motif %2d (E=%4d, P=%3d): %6.3f s\n", mid, length(occ_list), length(pm.packets), dt)
end
t_total_end = time()
@printf("\nTotal Step 11 Execution Time: %.3f s\n", t_total_end - t_total_start)
