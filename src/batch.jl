

export run, run_parallel, Simulation, SpBoxDiscretisation, discretize, committor


# batch code
function runbatch()
	with_perma(true, true) do
		batch(Simulation(nsteps=1_000_000), seeds=1:1, levels=vcat(3:20, 30, 40))
		batch(Simulation(nsteps=10_000_000), seeds=1:1, levels=vcat(3:20, 30, 40))
		batch(Simulation(nsteps=100_000_000), seeds=1:1, levels=vcat(3:20, 30, 40))
		batch(Simulation(nsteps=1_000_000_000), seeds=1:1, levels=vcat(3:20, 30, 40))
	end
end

@memoize PermaDict("cache/batch_") function batch(sim=Simulation(); seeds=1:1, levels=3:14)
	Random.seed!(0)

	# simulate trajectory
	sim = Sqra.run_parallel(sim, seeds=seeds)

	# compute discretizations
	n = length(levels)
	ds = Array{Any}(undef, n)
	cs = Array{Any}(undef, n)


	Threads.@threads for i in 1:n
		ncells = levels[i]
		t1 = @elapsed r = Sqra.discretize(Sqra.SpBoxDiscretisation(ncells=ncells), sim)
		@info "$t1 seconds for discretization $i"
		t2 = @elapsed c = Sqra.committor(r)
		@info "$t2 seconds for committor $i"
		ds[i] = r
		cs[i] = c
	end


	errs = Array{Any}(undef, n)
	Threads.@threads for i in 1:n
		t = @elapsed errs[i] = Sqra.sp_mse(cs[i], cs[end], ds[i].sb, ds[end].sb)
		@info "$t seconds for mse $i"
	end

	return sim, ds, cs, errs
end

### System simulation
#include("system.jl")


### Discretization

@with_kw struct VoronoiDiscretization
	prune = Inf
	npicks = 100
	neigh = 3*6
	Q = nothing
	inds = nothing
	picks = nothing
	u = nothing
end

@with_kw struct SpBoxDiscretisation
	prune = Inf
	ncells = 6 # in sb
	boundary = [-ones(6) ones(6)] .* 0.8 # in sb
	sb = nothing
	Q = nothing
	u = nothing
	picks = nothing
end


@memoize PermaDict("cache/dis_") function discretize(d::SpBoxDiscretisation, sim::Simulation)

	d.prune < Inf && error("Pruning is not currently supported. Iterative solver should suffice.")


	sb = SparseBoxes(sim.x, d.ncells, d.boundary)

	Q, inds = sqra(sb, sim.u, sim.sigma)

	u = sim.u[inds]
	picks = sim.x[:,inds]

	#=
	Q, pinds = prune_Q(Q, d.prune)
	picks = picks[:, pinds]
	u = u[pinds]
	sb = prune(sb, pinds)
	=#

	SpBoxDiscretisation(d, sb=sb, Q=Q, u=u, picks=picks)
end

#include("experiment.jl")


function trim(d::SpBoxDiscretisation, sim::Simulation, n)
	sb = trim(d.sb, n)
	Q, inds = sqra(sb, sim.u, sim.sigma)
end


using Setfield
function prune(b::SparseBoxes, pinds)
	b = @set b.boxes = b.boxes[:, pinds]
	b = @set b.inds = b.inds[pinds]
end



# warning: this part is old and should be rewritten as above
@memoize PermaDict("cache/dis_") function discretize(discretization::VoronoiDiscretization, sim::Simulation)
	@unpack x, u, sigma = sim
	@unpack prune = discretization
	sigma = sim.sigma
	beta = sigma_to_beta(sigma)

	Q, inds = sqra_voronoi(x, u, discretization.npicks, beta, discretization.neigh)

	Q, pinds = prune_Q(Q, prune)
	inds = inds[pinds]
	picks = x[:, inds]
	u = u[inds]

	discretization = typeof(discretization)(discretization, Q=Q, inds=inds, picks=picks, u=u)
	return discretization
end


### Committor computation

function changepoints(c)
	cc = copy(c)
	for i in 2:length(c)
		cc[i] != 0 && continue
		cc[i] = cc[i-1]
	end
	findall(diff(cc) .!= 0).+1
end



#=
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
=#












### Plotting functions
#=
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
	plot_triangles!(normalform(x), alpha=0.3, line_z = c, legend=false, seriescolor=:roma)
end
=#

using RecipesBase

function sparsity(d::SpBoxDiscretisation)
	l = d.ncells
	dim, n = size(d.picks)
	sparsity = n / l ^ dim * 100
end

@recipe function plot(d::SpBoxDiscretisation)
	title --> "LJ-Cluster, SparseBoxes"


	#@series begin
		l = d.ncells
		dim, n = size(d.picks)
		sp = round(sparsity(d), sigdigits=2)
		annotations := ((.0,-.1), ("l=$l,  n=$n ($sp%)", 8, :black, :left))
		w, h = round.((d.boundary[1:2,2] .- d.boundary[1:2,1]) / d.ncells, sigdigits=2)
		xticks --> [-w/2,w/2]
		yticks --> [-h/2,h/2]
		CloudPlot((d.picks, ))
	#=end

	@series begin
		seriestype := :shape
		aspect_ratio --> :equal
		seriesalpha --> 0.1
		series_color --> :beige
		legend --> false
		rectangle(w, h, x, y) = (x .+ [0,w,w,0], y .+ [0,0,h,h])
		#x, y = minimum(d.picks, dims=2)
		x, y = -.5, -.5
		w, h = (d.boundary[1:2,2] .- d.boundary[1:2,1]) / d.ncells
		rectangle(w,h, x, y)
	end=#
end







#= Graveyard †

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
