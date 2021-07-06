using Base: @locals, NamedTuple, Integer
using StatsBase
using Plots: limsType
using Distances
using LinearAlgebra
using PyCall
using Statistics

include("picking.jl")
include("sqra.jl")
include("sparseboxes.jl")
include("isokann.jl")

cmd = pyimport("cmdtools")


function Base.run(;
    x0 = x0gen,
    epsilon = 1,
    r0 = 1/3,
    harm = 1,
    sigma = 1/2,
    dt=0.001,
    nsteps=100000,
    maxdelta=0.1,
    npicks=100,
    neigh = 3*6,
	method = :voronoi,
	ncells = 6,
	prune = 100,
    beta = sigma_to_beta(sigma),
	boundary = [-ones(6) ones(6)] .* 0.8
)
   
    potential(x) = lennard_jones_harmonic(x; epsilon=epsilon, sigma=r0, harm=harm)
    x = eulermaruyama(x0 |> vec, potential, sigma, dt, nsteps, maxdelta=maxdelta)

	us = mapslices(potential, x, dims=1) |> vec

	if method == :voronoi
		Q, inds = sqra_voronoi(x, us, npicks, beta, neigh)
	else
		Q, inds = sqra_sparse_boxes(x, us, ncells, beta, boundary )
	end

	#pinds = prune_inds_Q(Q, prune)
	#Q = Q[pinds, pinds]

	Q, pinds = prune_Q(Q, prune)
	inds = inds[pinds]
	picks = x[:, inds]
	us = us[inds]

	classes = classify(picks)
    c = try
		println("solving committor...")
		#@time solve_committor(Q, classes)[1]
    catch
        nothing
    end

    return namedtuple(Base.@locals)
end

namedtuple(d::Dict) = (; d...)

" solve the committor system where we encode A==1 and B as anything != 0 or 1"
function solve_committor(Q, classes)
    QQ = copy(Q)
    b = copy(classes)
    for i in 1:length(classes)
        if b[i] != 0
            QQ[i,:] .= 0
            QQ[i,i] = 1
            if b[i] != 1
                b[i] = 0
            end
        end
    end
	c = QQ \ b
    return c, QQ, b
end

function convergence_error(r::NamedTuple, ns)
	errors = []
	for n in ns
		let u = r.u[1:n],
			pdist = r.pdist[1:n, 1:n],
			classes = r.classes[1:n]

			@show size(pdist)
			A = threshold_adjacency(pdist, r.neigh)
			Q = sqra(u, A, r.beta)
			c = solve_committor(Q, classes)[1]
			push!(errors, mean(abs2, c - r.c[1:n]))
		end
	end
	errors
end



### Sparse Boxes

function sqra_sparse_boxes(traj::AbstractMatrix, us::AbstractVector, ncells::Integer, beta, boundary=autoboundary(traj))
	@time A, picks = sparseboxpick(traj, ncells, us, boundary)
	@time Q = sqra(us[picks], A, beta)

	let fullsize = ncells^size(traj, 1), spsize = size(A,1)
		println("sparsity: $spsize/$fullsize=$(spsize/fullsize)")
	end

	return Q, picks
end


### Voronoi picking

function pick(traj::Matrix, n)
	picks, inds, dists = picking(traj,n)
    dists = dists[inds,:]
    return inds, sqrt.(dists)
end

""" compute the adjacency by thresholding pdist such that
on avg. the prescribed no. of neighbours is assigned """
function threshold_adjacency(pdist, avg_neighbor)
    d = sort(pdist[:])
    t = d[(avg_neighbor+1) * size(pdist, 1)]
    println("distance threshold is $t")
    A = sparse(0 .< pdist .<= t)
	check_connected(A)
    return A
end

function check_connected(A)
    unconn = findall((sum(A, dims=2) |> vec) .== 0)
    if length(unconn) > 0
        @warn "$length(unconn) states are not connected to any other states"
    end
    return unconn
end

function sqra_voronoi(traj, us, npicks, beta, average_neighbors = 3*size(traj,2))
	inds, pdist = pick(traj, npicks) # also return picked indices
	A = threshold_adjacency(pdist, average_neighbors)
	Q = sqra(us[inds], A, beta)
	return Q, inds
end




### Lennard Jones specifics

function lennard_jones_harmonic(x; sigma=1/4, epsilon=1, harm=1)
    #@show x
    x = reshape(x, 2, 3)
    _, m = size(x)
    u = 0.
    for i in 1:m
        u += sum(abs2, x[:,i]) * harm
        for j in i+1:m
            r = sigma^2 / sum(abs2, (x[:,i] .- x[:,j]))
            u += 4*epsilon * (r^6 - r^3)
        end
    end
    return u
end

classify(coords::Matrix) = mapslices(classify, coords, dims=1) |> vec

function classify(coords::Vector)
    ab = coords[3:4] - coords[1:2]
    ac = coords[5:6] - coords[1:2]

    angle = acos(min(dot(ab, ac) / norm(ab) / norm(ac), 1))
    offset = angle - pi/3  # offset to 60 degree
    if (abs(offset) < pi/12)  # +- 15 degrees
        return sign(ab[1]*ac[2] - ab[2]*ac[1])
    else
        return 0
    end
end

normalform(x::Matrix) = mapslices(normalform, x, dims=1)

" shift the first particle to 0 and rotate the second onto the x axis"
function normalform(x)
    x = reshape(x, 2, div(length(x),2))
    x = x .- x[:,1]

    one = [1,0]
    b   = normalize(x[:,2])
    B   = [b[1] -b[2]
           b[2]  b[1]]
    E   = [1 0
           0 1]
    A   =  E / B
    reshape(A * x, length(x))
end

const x0gen =  [0.19920158482463968
0.13789462153196408
-0.1709575705426315
0.0784533378749835
0.06778720715969005
-0.2112155752270007]



### Plotting functions

function plot_trajectories(x; kwargs...)
	scatter!(x[1:2:end,:]', x[2:2:end,:]'; kwargs...)
end

function plot_triangles!(n; kwargs...)
	xs = [n[1:2:end,:]; n[[1],:]]
	ys = [n[2:2:end,:]; n[[2],:]]
	plot!(xs, ys; kwargs...)
end

function plot_normalized(x, c)
	plot_trajectories(normalform(x),alpha=0.5, legend=false)
	plot_triangles(normalform(x), alpha=0.3, line_z = c, legend=false, seriescolor=:roma)
end



#= Graveyard

function snapshot(v::Vector)
    x = reshape(v, 2, div(length(v), 2))
    for i in 1:size(x, 2)
        scatter!([x[1,i]], [x[2,i]])
    end
    xlims!(-1,2)
    ylims!(-1,2)
end

macro extract(d)
    return :(
        ex=:(); for (k,v) in $d
           ex = :($ex; try global $k=$v catch end)
       end; eval(ex); $d
    )
end

### Diffusion Maps 
function diffusionmaps(x, n=3; alpha=1,sigma=1)
	D = cmd.estimation.diffusionmaps.DiffusionMaps(x', sigma, alpha, n=n)
	return D.dms
end


function pickingdists(dists)
    d = copy(dists)
    d[diagind(d)] .= Inf
    median(minimum(d, dims=2))
end

function mutual_distances(x::Matrix)
	d1 = sum(abs2, x[1:2,:] .- x[3:4,:], dims=1)
	d2 = sum(abs2, x[3:4,:] .- x[5:6,:], dims=1)
	d3 = sum(abs2, x[5:6,:] .- x[1:2,:], dims=1)
	[d1; d2; d3]
end

=#