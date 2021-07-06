using Distances
using SparseArrays
using ProgressMeter

spboxes(points::Vector, args...) = spboxes(reshape(points, (1, length(points))), args...)

function spboxes(points::Matrix, ncells, boundary=autoboundary(points))
    #affine transformation of boundary box onto the unit cube ( ncells)
    normalized = (points .- boundary[:,1]) ./ (boundary[:,2] - boundary[:,1]) * ncells

    # round to next int
	cartesians = ceil.(Int, normalized)
    # and adjust for left boundary
    cartesians[normalized.==0] .= 1

    cartesians, neigh_inds = uniquecoldict(cartesians)

	inside = all(1 .<= cartesians .<= ncells, dims=1) |> vec
	cartesians = cartesians[:, inside]
	neigh_inds = neigh_inds[inside]

	@show size(normalized, 2),  sum(inside)

	dims = repeat([ncells], size(points, 1))
	lininds = to_linearinds(cartesians, dims)
	A = boxneighbors(lininds, dims)



	A, cartesians, neigh_inds #, sparse(A)
end

function sparseboxpick(points::AbstractMatrix, ncells, potentials, boundary=autoboundary(points))
	n, m = size(points)
	#affine transformation of boundary box onto the unit cube ( ncells)
    normalized = (points .- boundary[:,1]) ./ (boundary[:,2] - boundary[:,1]) * ncells

	cartesians = ceil.(Int, normalized)  # round to next int
    cartesians[normalized.==0] .= 1  # and adjust for left boundary

	order=[]

	# select the (index of) the point with lowest potential for each cartesian box
	pickdict = Dict{typeof(cartesians[:,1]), Int}()
	@showprogress "sparse box picking" for i in 1:m
		c = cartesians[:,i]
		!inside(c, ncells) && continue  # skip over outside boxes
		best = get(pickdict, cartesians[:,i], nothing)
		if best === nothing
			pickdict[c] = i
			push!(order, i)
		elseif potentials[i] < potentials[best]
			pickdict[c] = i
		end
	end

	picks = values(pickdict) |> collect

	A = boxneighbors(cartesians[:, picks], ncells)

	@show length(picks)

	return A, picks, order
end

inside(cart, ncells) = all(1 .<= cart .<= ncells)

function boxneighbors(cartesians, ncells)
	dims = [ncells for i in 1:size(cartesians, 1)]
	lininds = to_linearinds(cartesians, dims)
	A = _boxneighbors(lininds, dims)
	return A
end





function uniquecoldict(x)
	n = size(x, 2)
	ua = Dict{Any, Vector{Int}}()
	for i in 1:n
		ua[x[:,i]] = push!(get(ua, x[:,i], Int[]), i)
	end
	return reduce(hcat, keys(ua)), values(ua)|>collect
end

function autoboundary(x)
    hcat(minimum(x, dims=2), maximum(x, dims=2))
end

function boxcenters(cartesians, boundary, ncells)
	delta = (boundary[:,2]-boundary[:,1])
	return (cartesians .- 1/2)  .* delta ./ (ncells) .+ boundary[:,1]
end


function test_spboxes()
    x = [0 0.5 0
	 0 0   1]
	cartesians, A = spboxes(x, 2)
	@assert all(A .== @show (pairwise(Cityblock(), cartesians) .== 1))
end


""" given a list of linear indices and the resective dimension of the grid
compute the neighbors by seraching for each possible (forward) neighbor.
we can reduce the search by starting at buffered positions from the preceding check
"""
function _boxneighbors(lininds, dims)
	perm = sortperm(lininds)
	lininds = lininds[perm]
	pointers = ones(Int, length(dims))
	offsets = cumprod([1;dims[1:end-1]])
	n = length(lininds)
	cart = CartesianIndices(tuple(dims...))
	A = spzeros(length(lininds), length(lininds))
	@showprogress "collecting neighbours" for (i, current) in enumerate(lininds)
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
	return A[invperm(perm), invperm(perm)]
end

function to_linearinds(cartinds, dims)
	li = LinearIndices(tuple(dims...))
	map(x->li[x...], eachcol(cartinds))
end
