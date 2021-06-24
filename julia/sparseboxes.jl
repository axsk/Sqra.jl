using Distances
using SparseArrays

spboxes(points::Vector, args...) = spboxes(reshape(points, (1, length(points))), args...)

function spboxes(points::Matrix, ncells, boundary=autoboundary(points))
    #affine transformation of boundary box onto the unit cube ( ncells)
    normalized = (points .- boundary[:,1]) ./ (boundary[:,2] - boundary[:,1]) * ncells

    # round to next int
	cartesians = ceil.(Int, normalized)
    # and adjust for left boundary
    cartesians[normalized.==0] .= 1

    cartesians = unique(cartesians, dims=2)

	dims = repeat([ncells], size(points, 1))
	lininds = to_linearinds(cartesians, dims)
	A = boxneighbors(lininds, dims)

    cartesians, A #, sparse(A)
end

function autoboundary(x)
    hcat(minimum(x, dims=2), maximum(x, dims=2))
end

function test_spboxes()
    x = [0 0.5 0
	 0 0   1]
	cartesians, A = spboxes(x, 2)
	@assert all(A .== @show (pairwise(Cityblock(), cartesians) .== 1))
end


""" given a sorted list of linear indices and the resective dimension of the grid
compute the neighbors by seraching for each possible (forward) neighbor.
we can reduce the search by starting at buffered positions from the preceding check
"""
	function boxneighbors(lininds, dims)
	pointers = ones(Int, length(dims))
	@show offsets = cumprod([1;dims[1:end-1]])
	n = length(lininds)
	cart = CartesianIndices(tuple(dims...))
	A = spzeros(length(lininds), length(lininds))
	for (i, current) in enumerate(lininds)
		for dim in 1:length(dims)
			target = current + offsets[dim]

			p = pointers[dim]
			range = view(lininds, p:n)
			j = findfirst(x -> x >= target, range)
			if isnothing(j)
				pointers[dim] = n + 1
				continue
			else
				j = j + p - 1
			end
			if (lininds[j] == target) && # target neighbor is present
				cart[i][dim] < dims[dim] # and we are not looking across the boundary
				A[i,j] = A[j,i] = 1
				pointers[dim] = j + 1
			else
				pointers[dim] = j
			end
		end
	end
	return A
end

function to_linearinds(cartinds, dims)
	li = LinearIndices(tuple(dims...))
	map(x->li[x...], eachcol(cartinds))
end
