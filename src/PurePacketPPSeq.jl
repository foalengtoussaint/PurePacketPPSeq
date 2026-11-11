module PurePacketPPSeq

using Random
using Statistics
using LinearAlgebra
using StaticArrays
using Distributions
using Printf
using SpecialFunctions
using Serialization

include("types.jl")
include("bounds.jl")
include("kinematics.jl")
include("packet_birth_split.jl")
include("occurrence_birth_prune.jl")
include("gibbs_resample.jl")
include("ou_engine.jl")
include("triad_vigor.jl")
include("metrics.jl")
include("scales.jl")

export
    # Point-Process Hierarchy & Mechanisms
    AbstractPointProcessGenerator,
    AbstractPointProcessEvent,
    NeuronGenerator,
    SpikeEvent,
    MetaOccurrence,

    # m-Mechanisms
    AbstractMechanism,
    Level0RateFloor,
    Level1EmissionPacket,
    Level2CarrierComposition,
    Level3GaugeSymmetry,
    Level4KinematicManifold,
    Level5ContinuousOU,
    RateFloor,
    EmissionPacketMechanism,
    CarrierComposition,
    GaugeSymmetry,
    KinematicManifold,
    ContinuousOU,

    # n-Scale Hierarchy
    AbstractScaleLevel,
    ScaleLevel,
    active_m_levels,
    is_m_active,
    incoming_data_type,
    outgoing_data_type,
    evaluate_scale_intensity,

    Packet,
    PurePacketMotif,
    PureOccurrence,
    SpikeAssignment,
    ContinuousOUContext,
    ARDManifoldState,
    AuditPacketInstance,

    # Bounds & Inverted Index
    update_motif_bounds!,
    update_all_motif_bounds!,
    rebuild_inverted_index!,
    update_chan_to_motifs,

    # Kinematics
    project_translational_gauge!,
    sample_packet_loadings!,
    sample_occurrence_coordinate!,
    resample_occurrence_onset!,
    sample_global_ard_precisions!,
    prune_inactive_axes!,
    check_and_birth_axis!,

    # Packets
    sample_packet_masses!,
    update_packet_corridors!,
    prune_dead_packets!,
    detect_and_birth_or_split_packets!,

    # Occurrences
    calc_marginal_log_evidence,
    prune_dead_occurrences!,
    detect_and_birth_occurrences!,
    sample_occurrence_amplitudes!,

    # Gibbs Resampling
    resample_spike_pure_packet!,
    resample_all_spikes_pure_packet_threaded!,
    resample_background_pure_packet!,

    # OU Engine
    exp_integral_segment,
    solve_sym_tridiagonal!,
    smooth_point_process_intensity!,
    evaluate_background_log_intensity,
    update_regional_ou!,
    update_cell_ou!,
    update_motif_ou!,
    update_continuous_background_rates!,

    # Vigor & Student-t
    update_vigor_shape!,
    sample_continuous_occurrence_amplitudes!,
    sample_student_t_precisions!,

    # Metrics
    standard_normal_cdf,
    select_stratified_audit_channels,
    compute_time_rescaling_ks,
    compute_rank_normalized_rhat_ess,
    compute_joint_energy

end # module PurePacketPPSeq
