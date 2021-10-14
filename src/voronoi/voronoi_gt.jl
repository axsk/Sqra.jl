const Point{T} = AbstractVector{T} where T<:Real
const Points = AbstractVector{<:Point}
const Sigma = AbstractVector{<:Integer}  # encoded by the ids of its generators
const Vertex = Tuple{<:Sigma, <: Point}
const Vertices = Dict{<:Sigma, <:Point}  # sig -> coords

struct SearchIncircle
	tmax::Float64
	tree::KDTree
end

voronoi(x; kwargs...) = voronoi(vecvec(x); kwargs...)

""" construct the voronoi diagram from `x` through breadth-first search """
function voronoi(xs::Points; tmax=1000)
	searcher = SearchIncircle(tmax, KDTree(xs))
	sig, r = descent(xs, searcher)
	v = explore(sig, r, xs, searcher)
	return v::Vertices, xs
end

voronoi_random(x, args...; kwargs...) = voronoi_random(vecvec(x), args...; kwargs...)

""" construct a (partial) voronoi diagram from `x` through a random walk """
function voronoi_random(xs::Points, iter=1000; tmax=1000, maxstuck=typemax(Int))
	searcher = SearchIncircle(tmax, KDTree(xs))
	sig, r = descent(xs, searcher)
	v = walk!(sig, r, iter, xs, searcher, maxstuck)
	return v::Vertices, xs
end

vecvec(x::Matrix) = map(SVector{size(x,1)}, eachcol(x))
vecvec(x::Vector{<:SVector}) = x

""" starting at given points, run the ray shooting descent to find vertices """
function descent(xs::Points, searcher, start = 1)
	sig = [start]
	r = xs[start]
	d = length(r)
	for k in d:-1:1  # find an additional generator for each dimension
		u = randray(xs[sig])
		(tau, t) = raycast(sig, r, u, xs, searcher)
		if t == Inf
			u = -u
			(tau, t) = raycast(sig, r, u, xs, searcher)
		end
		if t == Inf
			error("Could not find a vertex in both directions of current point." *
				"Consider increasing search range (tmax)")
		end
		sig = tau
		r = r + t*u
	end
	return (sig, r)::Vertex
end


""" starting at vertices, walk nsteps along the voronoi graph to find new vertices """
function walk!(sig::Sigma, r::Point, nsteps::Int, xs::Points, searcher, maxstuck = typemax(Inf))
	S = Dict(sig => r)
	nonew = 0
	prog = Progress(maxstuck, 1, "Voronoi walk")
	progmax = 0
	for s in 1:nsteps
		nonew += 1
		progmax = max(progmax, nonew)
		ProgressMeter.update!(prog, progmax)
		sig, r = walkray(sig, r, xs, searcher)
		get!(S, sig) do
			nonew = 0
			return r
		end
		nonew > maxstuck && break
	end
	return S
end

""" starting at vertex (v,r), return a random adjacent vertex """
walkray(v, r, xs, searcher) = walkray(v, r, xs, searcher, rand(1:length(v)))

""" find the vertex connected to `v` by moving away from its `i`-th generator """
function walkray(v::Sigma, r::Point, xs::Points, searcher, i)
	#e = v[1:end .!= i]
	e = deleteat(v, i)
	u = randray(xs[e])
	if (u' * (xs[v[i]] - xs[e[1]])) > 0
		u = -u
	end
	vv, t = raycast(e, r, u, xs, searcher)
	if t < Inf
		v = vv
		r = r + t*u
	end
	return v, r
end

""" BFS of vertices starting from `S0` """
function explore(sig, r, generators::Points, searcher)
	#Q = copy(S0)
	Q = Dict(sig=>r)
	S = empty(Q)
  Z = Dict{Vector{Int64}, Int}()

	cache = true
	hit = 0
	miss = 0
	conts = 0
	infs = 0

	while length(Q) > 0
		(v,r) = pop!(Q)
		for i in 1:length(v)

			if cache && get(Z, deleteat(v, i), 0) == 2
				conts += 1
				continue
			end

			vn, rn = walkray(v, r, generators, searcher, i)

			if vn == v
				infs += 1
				continue
			end

			if !haskey(S, vn)
				push!(Q, vn => rn)
				push!(S, vn => rn)
				if cache
					for j in 1:length(vn)
						vj = deleteat(vn, j)
						Z[vj] = get(Z, vj, 0) + 1
					end
				end
				hit += 1
			else
				miss += 1
			end

		end

	end
	@show hit, miss, conts, infs
	return S
end

deleteat(v, i) = deleteat!(copy(v), i)


## lowlevel subroutines


""" generate a random ray orthogonal to the subspace spanned by the given points """
function randray(x::Points)
	k = length(x)
	d = length(x[1])
	v = similar(x, k-1)

	# Gram Schmidt
	for i in 1:k-1
		v[i] = x[i] .- x[k]
		for j in 1:(i-1)
			v[i] = v[i] .- dot(v[i], v[j]) .* v[j]
		end
		v[i] = normalize(v[i])
	end

	u = randn(d)
	for i in 1:k-1
		u = u - dot(u, v[i]) * v[i]
	end
	u = normalize(u)
	return u
end


struct SearchBruteforce end

""" shooting a ray in the given direction, find the next connecting point.
This is the bruteforce variant, using a linear search to find the closest point """
function raycast(sig::Sigma, r, u, P, searcher::SearchBruteforce)
	eps = 1e-10
	(tau, ts) = [0; sig], Inf
	x0 = P[sig[1]]
	for i in 1:length(P)
		i in sig && continue
		x = P[i]
		t = (sum(abs2, r .- x) - sum(abs2, r .- x0)) / (2 * u' * (x-x0))
		if eps < t < ts
			(tau, ts) = vcat(sig, [i]), t
		end
	end

	# begin # check if new point is equidistant to its generators
	# 	rr = r + ts*u
	# 	diffs = [sum(abs2, rr.-s) for s in tau]
	#   allapprox(x) = all(isapprox(x[1], y) for y in x)
	# 	!allapprox(diffs) && error()
	# end
	return sort(tau), ts
end


struct SearchBisection
	eps::Float64
	tree::KDTree
end

""" shooting a ray in the given direction, find the next connecting point.
This variant (by Poliaski, Pokorny) uses a binary search """
function raycast(sig::Sigma, r::Point, u::Point, P::Points, searcher::SearchBisection)
	tau, tl, tr = [], 0, searcher.tmax
	x0 = P[sig[1]]
	iter = 0


	while tr-tl > searcher.eps
		tm = (tl+tr)/2
		i, _ = nn(searcher.tree, r+tm*u)
		x = P[i]
		if i in sig
			tl = tm
		else
			tr = (sum(abs2, r .- x) - sum(abs2, r .- x0)) / (2 * u' * (x-x0))
			tau = vcat(sig, [i])

			# early stopping
			idxs, dists = knn(searcher.tree, r+tr*u, length(sig)+1, true)
			length(intersect(idxs, [sig; i])) == length(sig)+1 && break
		end

	end

	if tau == []
		tau = [0; sig]
		tr = Inf
	end

	return sort(tau), tr
end

""" Shooting a ray in the given direction, find the next connecting point.
This variant uses an iterative NN search """
function raycast(sig::Sigma, r::Point, u::Point, P::Points, searcher::SearchIncircle)
	i = 0
	t = 1
	x0 = P[sig[1]]
	local d, n
	# find a t large enough to include a non-boundary (sig) point
	while t < searcher.tmax
		n, d = nn(searcher.tree, r+t*u)
		if d==Inf
			warn("d==Inf in raycast expansion, this should never happen")
			return [0; sig], Inf
		end

		if n in sig
			t = t * 2
		else
			i = n
			break
		end
	end

	if i == 0
		return [0; sig], Inf
	end

	# sucessively reduce incircles unless nothing new is found
	while true
		x = P[i]
		t = (sum(abs2, r - x) - sum(abs2, r - x0)) / (2 * u' * (x-x0))
		j, _ = nn(searcher.tree, r+t*u)
		if j in [sig; i]
			break
		else
			i = j
		end
	end

	tau = sort([i; sig])

	return tau, t
end

function raycast_compare(sig, r, u, P, searcher)
	r1  = raycast_incircle(sig,r,u,P,searcher)
	r2  = raycast_intersect(sig,r,u,P,searcher)
	r3  = raycast_bruteforce(sig,r,u,P)
	@assert r1[1] == r2[1]
	@assert r2[1] == r3[1]
	return r1
end
