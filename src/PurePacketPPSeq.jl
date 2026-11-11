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
include("mechanisms/m0_floor.jl")
include("mechanisms/m1_packet.jl")
include("mechanisms/m2_carrier.jl")
include("mechanisms/m3_gauge.jl")
include("mechanisms/m4_manifold.jl")
include("mechanisms/m5_ou.jl")
include("scales.jl")
include("sampler.jl")
include("metrics.jl")

export
    # Point-Process Hierarchy & Events
    AbstractPointProcessGenerator,
    AbstractPointProcessEvent,
    NeuronGenerator,
    SpikeEvent,
    MetaOccurrence,

    # Universal m-Mechanisms
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

    # Core Data Structures
    Packet,
    PurePacketMotif,
    PureOccurrence,
    SpikeAssignment,
    ContinuousOUContext,
    ARDManifoldState,
    AuditPacketInstance,

    # Mechanism m=2: Dynamic Bounding & Indexing
    update_motif_bounds!,
    update_all_motif_bounds!,
    rebuild_inverted_index!,
    update_chan_to_motifs,

    # Mechanism m=3: Gauge Invariance
    project_translational_gauge!,
    compute_gauge_error,

    # Mechanism m=4: Kinematic Manifold & ARD
    sample_packet_loadings!,
    sample_occurrence_coordinate!,
    sample_global_ard_precisions!,
    prune_inactive_axes!,
    check_and_birth_axis!,

    # Mechanism m=1: Packet Emissions
    sample_packet_masses!,
    update_packet_corridors!,
    prune_dead_packets!,
    detect_and_birth_or_split_packets!,

    # Mechanism m=2: Carrier Occurrences
    calc_marginal_log_evidence,
    resample_occurrence_onset!,
    prune_dead_occurrences!,
    detect_and_birth_occurrences!,
    sample_occurrence_amplitudes!,

    # Point-Process Gibbs Samplers
    resample_spike_pure_packet!,
    resample_all_spikes_pure_packet_threaded!,
    resample_background_pure_packet!,

    # Mechanism m=0 & m=5: Continuous OU & Rate Floor
    exp_integral_segment,
    solve_sym_tridiagonal!,
    smooth_point_process_intensity!,
    evaluate_background_log_intensity,
    evaluate_integrated_background,
    update_regional_ou!,
    update_cell_ou!,
    update_motif_ou!,
    update_continuous_background_rates!,

    # Mechanism m=5: Vigor & Student-t Precisions
    update_vigor_shape!,
    sample_continuous_occurrence_amplitudes!,
    sample_student_t_precisions!,

    # Objective Standard Metrics
    standard_normal_cdf,
    select_stratified_audit_channels,
    compute_time_rescaling_ks,
    compute_rank_normalized_rhat_ess,
    compute_joint_energy

end # module PurePacketPPSeq
