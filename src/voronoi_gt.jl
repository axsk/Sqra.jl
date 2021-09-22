using LinearAlgebra
using BenchmarkTools
using StaticArrays
using NearestNeighbors
using QHull, CDDLib



const Sigma = AbstractVector{<:Integer}  # Sigma komplex consisting of the ids of the generators
const Point{T} = AbstractVector{T} where T<:Real
const Points = AbstractVector{<:Point}
const Vertices = Dict{<:Sigma, <:Point}

struct NNSearch
	tmax::Float64
	eps::Float64
	tree::KDTree
end

function voronoi(x::Matrix, iter=1000, particles=1; tmax=1000, eps=1e-8, maxstuck=100)
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
function raycast_bruteforce(sig::Sigma, r, u, P)
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

function raycast_intersect(sig::Sigma, r::Point, u::Point, P::Points, searcher::NNSearch)
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


function raycast_incircle(sig::Sigma, r::Point, u::Point, P::Points, searcher::NNSearch)
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

# end of basic implementation
# boundary computations

""" given vertices in generator-coordinates,
collect the verts belonging to generator pairs, i.e. boundary vertices """
function adjacency(v::Vertices)
	conns = Dict{Tuple{Int,Int}, Vector{Vector{Int}}}()
	#conns=Dict()
	for (sig, r) in v
		for a in sig
			for b in sig
				a <= b && continue
				v = get!(conns, (a,b), [])
				push!(v, sig)
			end
		end
	end
	conns
end

using Polyhedra

function boundary(g1::Int, g2::Int, inds::AbstractVector{<:Sigma}, vertices::Vertices, points)
#function boundaries(vertices, conns, points)
	#Ahv = map(collect(conns)) do ((g1,g2), inds)
	vertex_coords = map(i->vertices[i], inds)
	push!(vertex_coords, points[g1])  # append one voronoi center for full volume
	#p =
	V = try
			volume(polyhedron(vrep(vertex_coords), QHull.Library()))
		catch e #::QHull.PyCall.PyError
			0
		end
	#plot!(p)

	h = norm(points[g1] - points[g2])
	A = 2 * V / h
	A, h, V
	#end
	#return Ahv
end

using SparseArrays
using ProgressMeter


function connectivity_matrix(vertices, P::AbstractVector)
	conns = adjacency(vertices)
	@show length(conns)
	#Ahv = boundaries(vertices, conns, P)
	I = Int[]
	J = Int[]
	V = Float64[]
	Vs = zeros(length(P))
	@showprogress for ((g1,g2), sigs) in conns
	#for ((A, h, v), (g1,g2)) in zip(Ahv, keys(conns))
		push!(I, g1)
		push!(J, g2)
		A, h, v = boundary(g1, g2, sigs, vertices, P)
		push!(V, A/h)
		Vs[g1] += v
		Vs[g2] += v
	end
	A = sparse(I, J, V, length(P), length(P))
	A = A + A'
	Vsi = 1 ./ Vs # check if we want row or col
	A = A .* Vsi
	return A, Vs
end



using Plots
# this works after filtering out the boundary-outbound vertices
function test(n=5, iter=10000)
	plot(legend=false);
	x = hcat(hexgrid(n)...)
	x .+= randn(2,n*n) .* 0.01
	@time v, P  = voronoi(x, iter)

	v = Dict(filter(collect(v)) do (k,v)
		norm(v) < 10
		end)

	#c = extractconn(v)
	@time A, Vs = connectivity_matrix(v, P)

	AA = map(x->x>.0, A)
	plot_connectivity!(AA .* 2, P)
	scatter!(eachrow(hcat(values(v)...))...)
	xlims!(1,6); ylims!(0,5)
end


function benchmark(n=100, d=6, iter=100, particles=10)
	x = rand(d, n)
	@benchmark voronoi($x, $iter, $particles)
end


function hexgrid(n)
	P = []
	for i in 1:n
		for j in 1:n
			x = i + j/2
			y = j * sqrt(3) / 2
			push!(P, [x,y])
		end
	end
	P
end




### Plotting

mscat(x) = scatter!(eachrow(x)...)

function plot_connectivity!(A, P)
	for (i,j,v) in zip(findnz(A)...)
		x1, y1 = P[i]
		x2, y2 = P[j]
		plot!([x1, x2], [y1, y2], linewidth=log.(v), color=:black)

	end
	plot!()
end

function plotface!(sig, r)
	for s in sig
		plot!([s[1], r[1]], [s[2], r[2]], linestyle=:dash)
	end
	plot!()
end

function plotconns!(conns)
	for c in conns
		a,b = c
		x1, y1 = a
		x2, y2 = b
		plot!([x1, x2], [y1, y2])
	end
end

function plotcenters!(P)
	scatter!(collect(eachrow(reduce(hcat, P)))...)
end

function plotwalk!(P, s0 = descent(P))
	s = walk(s0, 1, P)
	if length(s) > 0
		@show s[1][2], s0[1][2]
		x1, y1 = s0[1][2]
		x2, y2   = s[1][2]
		plot!([x1, x2], [y1, y2])
		return s
	end
	return s0
end

function uniquefaces(s)
	sigs = []
	rs = []
	for x in s
		sig, r = x
		if !(sig in sigs)
			push!(sigs, sig)
			push!(rs, r)
		end
	end
	return sigs, rs
end

function testreal(d=6, n=200)
	x = rand(d, n)
	@time v,p = voronoi(rand(d,n), 10_000_000; maxstuck=100_000);
	@show length(v)
	a = adjacency(v)
	println("avg. no. of neighbors: ", length(a)/n)
	println("avg. no. of vertices per face: ", mean(length.(values(a))))
	@time c = connectivity_matrix(v, p)
	v,p,c
end


function hrp(v, a=adjacency(v))
	for ((g1,g2), vs) in a
		d=Dict()
		for i in 1:length(vs)
			for j in 1:length(vs[i])
				if vs[i][j] in (g1,g2)
					continue
				end
				sig = vs[i][1:end .!= j]
				x = get!(d, vs[i][j], [])
				push!(x, i)
			end
		end
		return d, first(values(a))
	end
end

## experimental code

function orthonorm(x)
	v = deepcopy(x)
	k = length(x)
	for i in 1:k
		v[i] = x[i]# .- x[k]
		for j in 1:(i-1)
			v[i] = v[i] .- dot(v[i], normalize(v[j])) .* normalize(v[j])# / norm(v[j])^2
		end
		#v[i] = normalize(v[i])
	end
	v
end

function t(v)
	# look in the adjacency list for verts with 3 same gens, these lie on a hyperplane
	h, vs = hrp(v)
	vinds = first(values(h))
	c = map(i->v[i], vs[vinds])
	cc = c.-[c[1]]
	cc = Vector.(cc)
	# probably we want to set cc[5] to the inside generator
	o = orthonorm([cc[1:5]; [ones(6)]])
	o = orthonorm(o) # whyever this makes a difference?!
	u = o[6] |> normalize
	b = c[1]' * u
	@show [c'*u for c in c ]
	u, b
end
