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

include("permadict.jl")

# simulation
include("eulermaruyama.jl")

# voronoi
include("picking.jl")
#include("voronoi_lp.jl")

# sparse boxes
include("sparseboxes.jl")
include("spdistances.jl")

# sqra
include("sqra_core.jl")
include("committor.jl")
# models and experiments
#include("lennardjones.jl")
include("models.jl")
include("experiment.jl")
include("experiment2.jl")
#include("batch.jl")
include("errors.jl")

module Voronoi
	include("voronoi/Voronoi.jl")
end

end
