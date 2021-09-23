include("sbdict.jl")

SparseBoxes = SparseBoxesDict

function autoboundary(x)
    hcat(minimum(x, dims=2), maximum(x, dims=2))
end

function cartesiancoords(points, ncells, boundary=autoboundary(points))
	#affine transformation of boundary box onto the unit cube (ncells)
	normalized = (points .- boundary[:,1]) ./ (boundary[:,2] - boundary[:,1]) .* ncells  # (289)
	cartesians = ceil.(Int, normalized)  # round to next int
    cartesians[normalized.==0] .= 1  # and adjust for left boundary
	return cartesians
end


#include("sbmatrix.jl")
