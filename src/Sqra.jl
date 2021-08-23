module Sqra


#include("cmdtools.jl")
using Plots: right
using StatsBase: params
include("eulermaruyama.jl")
include("permadict.jl")
#include("isokann.jl")
#include("metasgd.jl")
#include("molly.jl")
#include("neurcomm.jl")

include("picking.jl")
include("sparseboxes.jl")

include("spdistances.jl")
include("sqra_core.jl")
include("voronoi_lp.jl")
include("lennardjones.jl")

end
