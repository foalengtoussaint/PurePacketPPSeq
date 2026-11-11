# scales.jl - Hierarchical Point-Process Scale Framework
#
# Partitions the scale architecture into:
# 1. scales/contract.jl: Formal definition of what a Scale is in point-process theory
#    (AbstractScale, trait queries, intensity evaluation contracts).
# 2. scales/implementations.jl: Concrete domain implementations
#    (ScaleLevel{0} / NeuronBaselineScale, ScaleLevel{1} / NeuralSequenceScale).

include("scales/contract.jl")
include("scales/implementations.jl")
