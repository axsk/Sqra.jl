using Distances
using SparseArrays
using ProgressMeter

export SparseBoxes

struct SparseBoxes
	ncells::Int
	boundary::Matrix{Float64}
	boxes::Matrix{Int}  # cartesian coordinates of the boxes
	inds::Vector{Vector{Int}}  # indices to the points contained in each box
end


function SparseBoxes(points, ncells, boundary=autoboundary(points))
	carts = cartesiancoords(points, ncells, boundary)
	boxes, inds = uniquecols(carts, ncells)

	SparseBoxes(ncells, boundary, boxes, inds)
end


""" merge two SparseBoxes, corresponding to SparseBoxes of the concat trajectory.
`offset` is the length of the trajectory underlying `a` """
function merge(a::SparseBoxes, b::SparseBoxes, offset::Int)
	@assert a.ncells == b.ncells
	@assert a.boundary == b.boundary

	boxes = hcat(a.boxes, b.boxes)
	binds = map(i -> i .+ offset, b.inds)
	inds = vcat(a.inds, binds)

	boxes, iis = uniquecols(boxes, a.ncells)
	inds = map(ii->reduce(vcat, inds[ii]), iis)

	return SparseBoxes(a.ncells, a.boundary, boxes, inds)
end


adjacency(sb::SparseBoxes) = boxneighbors(sb.boxes, sb.ncells)

""" chance to discover a new box per samlpe at the tail of the data """
function convergence(s::SparseBoxes, tailpercent = 0.05)
	discoveries = map(first, s.inds) |> sort |> diff
	tail = ceil(Int, length(discoveries) * (1 - tailpercent))
	1 / mean(discoveries[tail:end])
end

function convergencedata(s::SparseBoxes)
	discoveries = sort(first.(s.inds))
	discoveries, 1:length(discoveries)
end



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

function uniquecols(c, ncells)
	p = sortperm(collect(eachcol(c)))

	inds = Vector{Int}[]
	last = @view c[:, p[1]]
	inside(last, ncells) && push!(inds, [p[1]])

	for i in 2:length(p)
		pp = p[i]
		curr = view(c, :, pp)
		!inside(curr, ncells) && continue
		if last == curr
			push!(inds[end], pp)
		else
			push!(inds, [pp])
		end
		last = curr
	end

	ii = map(first, inds)
	b = c[:, ii]

	return b, inds
end

inside(cart, ncells) = all(1 .<= cart .<= ncells)  # todo: 240 is there sthg faster?

function boxneighbors(cartesians, ncells)
	dims = [ncells for i in 1:size(cartesians, 1)]
	lininds = to_linearinds(cartesians, dims)
	A = _boxneighbors(lininds, dims)
	return A
end

function to_linearinds(cartinds, dims)
	li = LinearIndices(tuple(dims...))
	map(x->li[x...], eachcol(cartinds))
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
	#A = spzeros(length(lininds), length(lininds))
	I = Int[]
	J = Int[]
	#@showprogress "collecting neighbours"
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
				#A[i,j] = A[j,i] = 1
				push!(I, i)
				push!(J, j)
				pointers[dim] = j + 1
			else
				pointers[dim] = j
			end
		end
	end
	A = sparse(I, J, ones(Bool, length(I)), n, n)
	A = A + A'
	return A[invperm(perm), invperm(perm)]
end


### deprecated


# deprecated since the picking is part of sqra not sparse boxes, only here for tests
function sparseboxpick(points, ncells, u, boundary=autoboundary(points))
	carts = cartesiancoords(points, ncells, boundary)
	boxes, allinds = uniquecols(carts, ncells)

	inds = map(i -> i[argmin(u[i])], allinds)
	A = boxneighbors(boxes, ncells)

	return A, inds
end

function sparseboxpick_old(points::AbstractMatrix, ncells, potentials, boundary=autoboundary(points))
	n, m = size(points)
	cartesians = cartesiancoords(points, ncells, boundary)
	#order=[]

	# select the (index of) the point with lowest potential for each cartesian box
	pickdict = Dict{typeof(cartesians[:,1]), Int}()
	#@showprogress "sparse box picking"
	for i in 1:m
		c = cartesians[:,i]
		!inside(c, ncells) && continue  # skip over outside boxes
		best = get(pickdict, cartesians[:,i], nothing)
		if best === nothing
			pickdict[c] = i
			#push!(order, i)
		elseif potentials[i] < potentials[best]
			pickdict[c] = i
		end
	end

	picks = values(pickdict) |> collect


	A = boxneighbors(cartesians[:, picks], ncells)

	return A, picks
end

function boxcenters(cartesians, boundary, ncells)
	delta = (boundary[:,2]-boundary[:,1])
	return (cartesians .- 1/2)  .* delta ./ (ncells) .+ boundary[:,1]
end


#=
### Very old implementation

spboxes(points::Vector, args...) = spboxes(reshape(points, (1, length(points))), args...)

function spboxes(points::Matrix, ncells, boundary=autoboundary(points))
	cartesians = cartesiancoords(points, ncells, boundary)

    cartesians, neigh_inds = uniquecoldict(cartesians)

	inside = all(1 .<= cartesians .<= ncells, dims=1) |> vec
	cartesians = cartesians[:, inside]
	neigh_inds = neigh_inds[inside]

	dims = repeat([ncells], size(points, 1))
	lininds = to_linearinds(cartesians, dims)
	A = boxneighbors(lininds, dims)


	A, cartesians, neigh_inds #, sparse(A)
end


function uniquecoldict(x)
	n = size(x, 2)
	ua = Dict{Any, Vector{Int}}()
	for i in 1:n
		ua[x[:,i]] = push!(get(ua, x[:,i], Int[]), i)
	end
	return reduce(hcat, keys(ua)), values(ua)|>collect
end

=#
