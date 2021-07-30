using Base: @locals, NamedTuple, Integer
using StatsBase
using Plots
using LinearAlgebra
using Statistics
using Parameters
using SparseArrays
import Base.run
import IterativeSolvers


### System simulation

@with_kw struct Simulation
	x0 = x0gen
	epsilon = 1
    r0 = 1/3
    harm = 1
    sigma = 1/2
    dt=0.001
    nsteps=100000
    maxdelta=0.1
	x=nothing
	u=nothing
end

function run(params::Simulation)
	@unpack_Simulation params

	potential(x) = lennard_jones_harmonic(x; epsilon=epsilon, sigma=r0, harm=harm)

	x = eulermaruyama(x0 |> vec, potential, sigma, dt, nsteps, maxdelta=maxdelta)
	u = mapslices(potential, x, dims=1) |> vec

	@pack_Simulation
end


### Discretization

@with_kw mutable struct VoronoiDiscretization
	prune = Inf
	npicks = 100
	neigh = 3*6
	Q = nothing
	inds = nothing
	picks = nothing
	u = nothing
end

@with_kw mutable struct SpBoxDiscretisation
	prune = Inf
	ncells = 6
	boundary = [-ones(6) ones(6)] .* 0.8
	Q = nothing
	inds = nothing
	picks = nothing
	u = nothing
	cartesians = nothing
end

sqra(d::SpBoxDiscretisation, x, u, beta) = sqra_sparse_boxes(x, u, d.ncells, beta, d.boundary)
sqra(d::VoronoiDiscretization, x, u, beta) = sqra_voronoi(x, u, d.npicks, beta, d.neigh)

function discretize(discretization, sim::Simulation)
	@unpack x, u, sigma = sim
	@unpack prune = discretization
	sigma = sim.sigma
	beta = sigma_to_beta(sigma)

	#@unpack_SpBoxDiscretisation params
	#Q, inds = sqra_sparse_boxes(x, u, ncells, beta, boundary)

	Q, inds = sqra(discretization, x, u, beta)

	Q, pinds = prune_Q(Q, prune)
	inds = inds[pinds]
	picks = x[:, inds]
	u = u[inds]

	if isa(discretization,SpBoxDiscretisation)
		cartesians = cartesiancoords(picks, discretization.ncells, discretization.boundary)
		@pack! discretization = cartesians
	end

	@pack! discretization = Q, inds, picks, u
	return discretization
end

beta_to_sigma(beta) = sqrt(2/beta)
sigma_to_beta(sigma) = 2 / sigma^2


### Committor computation

function committor(discretization, method = idr; maxiter=1000, precondition=false)
	@unpack Q, picks = discretization
    cl = classify(picks)

	A, b = committor_system(Q, cl)
	#=

	if precondition == :left
		Pinv =  inv(Diagonal(A))
		c = solver(Pinv*A, collect(Pinv*b), method, maxiter)
	elseif precondition == :right
		Pinv =  inv(Diagonal(A))
		y = solver(A*Pinv, b, method, maxiter)
		c = Pinv * y
	else
		c = solver(A, b, method, maxiter)
	end
	=#

	c = IterativeSolvers.gmres(A, b; maxiter=maxiter, Pl=Diagonal(A))

	res = sqrt(sum(abs2, A*c - b))
	println("Committor residual: ", res)

	return c
end


@enum CommittorSolver begin
	direct
	lsqr
	lsmr
	gmres
	idrs
end

function solver(A, b, method, maxiter)


	if method == direct
		try
			c = A \ b
		catch e
			println("Could not solve the committor system directly: $e")
			c = fill(NaN, size(b))
		end
	else
		f = eval(:(IterativeSolvers.$(Symbol(string(method)))))
		c = try
			f(A, b; maxiter=maxiter)#, Pl = Diagonal(A))
		catch
			println("z")
			zero(b)
		end
	end

	return c
end

" solve the committor system where we encode A==1 and B as anything != 0 or 1"
function committor_system(Q, classes)
    #QQ = copy(Q)
	QQ = sparse(Q') # we work on the transpose since csc column access is fast
    b = copy(classes)
    for i in 1:length(classes)
        if b[i] != 0  # we have a boundary condition
            QQ[:,i] .= 0
			zerocol!(QQ, i)  # note that we work with the transpose
            QQ[i,i] = 1
            if b[i] != 1  # boundary is not 1
                b[i] = 0
            end
        end
    end
	QQ = sparse(QQ')
	#c = QQ \ b

    #return c, QQ, b
	return QQ, Float64.(b)
end


# set column i of the CSRMatrix Q to 0
# basically the same as `Q[:,i] .= 0`, but way faster
function zerocol!(Q::SparseMatrixCSC, i)
    Q.nzval[Q.colptr[i]:Q.colptr[i+1]-1] .= 0
end

function test_zerocol!(n=10000)
	x = sprand(n,n,.01)
	y = copy(x)
	@time y[:,1] .= 0
	@time zerocol!(x, 1)
	@assert x == y
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
