using LinearAlgebra
using BenchmarkTools
using StaticArrays
using NearestNeighbors
using QHull, CDDLib


const SVertex = AbstractVector{<:Integer}  # SVertex komplex consisting of the ids of the generators
const SVertices = AbstractVector{<:SVertex}
const Point{T} = AbstractVector{T} where T<:Real
const Points = AbstractVector{<:Point}
const Vertices = Dict{<:SVertex, <:Point}

struct NNSearch
	tmax::Float64
	eps::Float64
	tree::KDTree
end

function voronoi(x::Matrix, iter=1000, particles=1; tmax=1000, eps=1e-8, maxstuck=Inf)
	P = vecvec(x)
	searcher = NNSearch(tmax, eps, KDTree(x))
	s0 = descent(P, P[collect(1:particles)], searcher)
	v = walk(s0, iter, P, searcher, maxstuck)
	return v::Vertices, P
end

vecvec(x::Matrix) = map(SVector{size(x,1)}, eachcol(x))
vecvec(x::Vector{<:SVector}) = x

""" generate a random ray orthogonal to the subspace spanned by the given points """
function randray(x::Points)
	k = length(x)
	d = length(x[1])
	v = similar(x, k-1)

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

""" shooting a ray in the given direction, find the next connecting point """
function raycast_bruteforce(sig::SVertex, r, u, P)
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

function raycast_intersect(sig::SVertex, r::Point, u::Point, P::Points, searcher::NNSearch)
	tau, tl, tr = [], 0, searcher.tmax
	x0 = P[sig[1]]
	iter = 0
	#@show norm(x0-r)
	#@show idxs, dists = knn(searcher.tree, r, length(sig)+5, true, i->(dot(P[i]-r, u) <= 0))

	#try @show [(sum(abs2, r .- x) - sum(abs2, r .- x0)) / (2 * u' * (x-x0)) for x in P[idxs]] catch; end

	while tr-tl > searcher.eps
		#@show iter += 1
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
			#@show idxs, i, sig, dists
			length(intersect(idxs, [sig; i])) == length(sig)+1 && break

			#if length(intersect()
		end

	end

	if tau == []
		tau = [0; sig]
		tr = Inf
	end
	#@show iter, tr, sort(tau)
	return sort(tau), tr
end


function raycast_incircle(sig::SVertex, r::Point, u::Point, P::Points, searcher::NNSearch)
	i = 0
	t = 1
	x0 = P[sig[1]]
	local d, n
	# find a t large enough to include a non-boundary (sig) point
	while t < searcher.tmax
		n, d = nn(searcher.tree, r+t*u)
		d==Inf && warn("d==Inf in raycast expansion, this should never happen")

		if n in sig
			t = t * 2
		else
			i = n
			break
		end
	end

	if i == 0 || d == Inf
		#@show "inv"
		#@show sig, i, t, d,n
		tau = sort([0; sig])
		return tau, Inf
	end

	# sucessively reduce incircles unless nothing new is found
	while true
		x = P[i]
		t = (sum(abs2, r .- x) - sum(abs2, r .- x0)) / (2 * u' * (x-x0))
		#@show t, u
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

#raygen(sig, r, u, P, searcher::NNSearch) = raycast_intersect(sig, r, u, P, searcher)
#raygen(sig, r, u, P, searcher) = raycast_bruteforce(sig, r, u, P)

function raygen_compare(sig, r, u, P, searcher)
	r1  = raycast_incircle(sig,r,u,P,searcher)
	r2  = raycast_intersect(sig,r,u,P,searcher)
	r3  = raycast_bruteforce(sig,r,u,P)
	@assert r1[1] == r2[1]
	@assert r2[1] == r3[1]
	return r1
end

raygen(x...) = raycast_incircle(x...)

""" starting at given points, run the ray shooting descent to find vertices """
function descent(PP, P, searcher)
	d = length(P[1])
	Sd1 = [[i] for i in 1:length(P)]
	Sd2 = [xi for xi in P]
	for k in d:-1:1
		Sdm1 = []
		Sdm2 = []
		for (sig, r) in zip(Sd1, Sd2)
			u = randray(PP[sig])
			(tau, t) = raygen(sig, r, u, PP, searcher)
			if t == Inf
				#println("invert direction")
				u = -u
				(tau, t) = raygen(sig, r, u, PP, searcher)
			end
			#@show (tau, t)
			if !(tau in Sdm1)
				push!(Sdm1, tau)
				push!(Sdm2, r + t*u)
			end
		end
		Sd1, Sd2 = Sdm1, Sdm2
	end
	#mscat(Sd2[1])
	return Dict((a,b) for (a,b) in zip(Sd1, Sd2))  # transform to array of tuples
end

function walkray(v, r, xs, searcher)
	i = rand(1:length(v))
	e = v[1:end .!= i]
	u = randray(xs[e])
	if (u' * (xs[v[i]] - xs[e[1]])) > 0
		u = -u
	end
	vv, t = raygen(e, r, u, xs, searcher)
	if t < Inf
		v = vv
		r = r + t*u
	end
	return v, r
end

""" starting at vertices, walk nsteps along the voronoi graph to find new vertices """
function walk(S0, nsteps, PP, searcher, maxstuck = Inf)
	S = empty(S0)
	nonew = 0
	for (v, r) in S0
		for s in 1:nsteps
			nonew += 1
			v, r = walkray(v, r, PP, searcher)
			get!(S, v) do
				nonew = 0
				return r
			end
			nonew > maxstuck && break
		end
	end
	return S
end
