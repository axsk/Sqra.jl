""" the glue code """

export Setup, Experiment

@with_kw struct Setup
	sim = Simulation()
	boundary = [-ones(6) ones(6)] .* 0.8 # in sb
	level = 6
end

struct Experiment
	sim::Simulation
	sb::SparseBoxes
	Q::Matrix{Float64}
	picks::Vector{Int}
	committor
end

function Experiment(c::Setup)
	sim = run(c.sim)
	sb = SparseBoxes(sim.x, c.level, c.boundary)
	Q, picks = sqra(sb, sim.u, sim.sigma)
	cl = classify(sim.x[:, picks])
	c = committor(Q, cl, maxiter = 1000)
	Experiment(sim, sb, Q, picks, c)
end

picks(e::Experiment) = e.sim.x[:, e.picks]


function sqra(sb::SparseBoxes, u, sigma)
	A = adjacency(sb)
	inds = min_u_inds(sb.inds, u)
	Q = sqra(u[inds], A, sigma_to_beta(sigma))
	return Q, inds
end


beta_to_sigma(beta) = sqrt(2/beta)
sigma_to_beta(sigma) = 2 / sigma^2

function stationary(e::Experiment)
	beta = sigma_to_beta(e.sim.sigma)
	u = e.sim.u[e.picks]
	p = exp.(-u .* (beta / 2))
	p / sum(p)
end

min_u_inds(inds::Vector{Vector{Int}}, u::Vector) = map(i -> i[argmin(u[i])], inds)

function trim(s::SparseBoxes, n)
	inds = map(is->filter(i->i<=n, is), s.inds)
	select = findall(length.(inds) .> 0)
	SparseBoxes(sb.ncells, sb.boundary, sb.boxes[:,select], inds[select])
end

function extend(s::Experiment, n)
	sim = extend(s.sim, n)
	sb = merge(s.sb, SparseBoxes(sim.x[:, end-n+1:end], s.sb.ncells, s.sb.boundary), size(s.sim.x, 2))
	Q, picks = sqra(sb, sim.u, sim.sigma)
	cl = classify(sim.x[:, picks])
	c = committor(Q, cl, maxiter = 1000)
	Experiment(sim, sb, Q, picks, c)
end

function sparsity(e::Experiment)
	l = e.sb.ncells
	dim, n = size(e.sb.boxes)
	sparsity = n / l ^ dim * 100
end


using Printf
@recipe function plot(e::Experiment)
	sb = e.sb
	l = sb.ncells
	x = picks(e)
	dim, nb = size(x)
	n = size(e.sim.x, 2)
	sp = round(sparsity(e), sigdigits=2)
	w, h = round.((sb.boundary[1:2,2] .- sb.boundary[1:2,1]) / sb.ncells, sigdigits=2)

	title --> "LJ-Cluster, SparseBoxes"
	n = @sprintf "%.1e" n
	annotations := ((.0,-.1), ("l=$l,  ns = $n, nb=$nb ($sp%)", 8, :black, :left))
	xticks --> [-w/2,w/2]
	yticks --> [-h/2,h/2]
	com --> e.committor
	CloudPlot((x, ))
end

function logspace(a,b,n)
	exp.(range(log(a),log(b), length=n))
end

function logspace(::Type{Int}, a, b, n)
	round.(Int, logspace(a,b,n))
end

function errors(es::Vector{Experiment})
	n = length(es)# - 1
	ee = es[end]
	p = stationary(ee) * length(ee.picks)  # dpi / dlebesgue
	errs = zeros(n)
	Threads.@threads for i in 1:n
		e = es[i]
		t = @elapsed errs[i] = Sqra.sp_mse(
			ee.committor, e.committor,
			ee.sb, e.sb, p)
		#@info "$t seconds for MSE l=$(e.sb.ncells)"
	end
	return errs
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


function committor(Q, cl; maxiter=1000)
	A, b = committor_system(Q, cl)
	c = copy(b)

	Pl = Diagonal(A)
	for i in 1:size(Pl, 1)
		Pl[i,i] == 0 && (Pl[i,i] = 1)
	end

	IterativeSolvers.gmres!(c, A, b; maxiter=maxiter, Pl=Pl)
	# TODO: report nonconvergence

	res = sqrt(sum(abs2, (Pl^-1)*(A*c - b)))
	println("Committor residual mean: ", res)

	return c
end

@memoize PermaDict("cache/com_") function committor(discretization, maxiter=1000)
	@unpack Q, picks = discretization
    cl = classify(picks)
	return committor(Q, cl; maxiter=maxiter)
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


### Committor computation

function changepoints(c)
	cc = copy(c)
	for i in 2:length(c)
		cc[i] != 0 && continue
		cc[i] = cc[i-1]
	end
	findall(diff(cc) .!= 0).+1
end
