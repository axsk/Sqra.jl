module Sqra

using Plots: right
using StatsBase: params

using Base: @locals, NamedTuple, Integer
using StatsBase
using Plots
using LinearAlgebra
using Statistics
using Parameters
using SparseArrays
import Base.run
import IterativeSolvers
using Random
using JLD2
using Memoize

#include("isokann.jl")
#include("metasgd.jl")
#include("molly.jl")
#include("neurcomm.jl")

include("eulermaruyama.jl")
include("permadict.jl")



include("picking.jl")
include("sparseboxes.jl")
include("spdistances.jl")
include("sqra_core.jl")
include("voronoi_lp.jl")
include("lennardjones.jl")
include("experiment.jl")
include("batch.jl")

end
