using LinearAlgebra
using BenchmarkTools
using StaticArrays
using NearestNeighbors
using QHull #, CDDLib

const Sigma = AbstractVector{<:Integer}  # Sigma komplex consisting of the ids of the generators
const Point{T} = AbstractVector{T} where T<:Real
const Points = AbstractVector{<:Point}
const Vertices = Dict{<:Sigma, <:Point}

struct NNSearch
	tmax::Float64
	eps::Float64
	tree::KDTree
end

function voronoi(x::Matrix, iter=1000, particles=1, tmax=1000, eps=1e-8)
	P = vecvec(x)
	searcher = NNSearch(tmax, eps, KDTree(x))
	s0 = descent(P, P[collect(1:particles)], searcher)
	v = walk(s0, iter, P, searcher)
	return v::Vertices, P
end

vecvec(x) = map(SVector{size(x,1)}, eachcol(x))

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
	(tau, ts) = [], Inf
	x0 = P[sig[1]]
	for i in 1:length(P)
		i in sig && continue
		x = P[i]
		t = (sum(abs2, r .- x) - sum(abs2, r .- x0)) / (2 * u' * (x-x0))
		if 0 < t < ts
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
		tr = Inf
	end
	#@show iter, tr, sort(tau)
	return sort(tau), tr
end


function raycast_incircle(sig::Sigma, r::Point, u::Point, P::Points, searcher::NNSearch)
	i = 0
	t = 1
	x0 = P[sig[1]]

	# find a t large enough to include a non-boundary (sig) point
	while t < searcher.tmax
		i, _ = nn(searcher.tree, r+t*u)
		if i in sig
			t = t * 2
		else
			break
		end
	end

	if i == 0
		tau = sort([sig, 0])
		return tau, Inf
	end

	# sucessively reduce incircles unless nothing new is found
	while true
		x = P[i]
		t = (sum(abs2, r .- x) - sum(abs2, r .- x0)) / (2 * u' * (x-x0))
		j, _ = nn(searcher.tree, r+t*u)
		if j in [sig; i]
			break
		else
			i = j
		end
	end

	tau = sort([sig; i])

	return tau, t
end


""" starting at given points, run the ray shooting descent to find vertices """
function descent(PP, P, searcher)
	raygen(sig, r, u, P) = raycast_intersect(sig, r, u, P, searcher)
	d = length(P[1])
	Sd1 = [[i] for i in 1:length(P)]
	Sd2 = [xi for xi in P]
	for k in d:-1:1
		Sdm1 = []
		Sdm2 = []
		for (sig, r) in zip(Sd1, Sd2)
			u = randray(PP[sig])
			(tau, t) = raygen(sig, r, u, PP)
			if t == Inf
				#println("invert direction")
				u = -u
				(tau, t) = raygen(sig, r, u, PP)
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

""" starting at vertices, walk nsteps along the voronoi graph to find new vertices """
function walk(S0, nsteps, PP, searcher)
	# raygen = raycast_bruteforce
 	raygen(sig, r, u, P) = raycast_intersect(sig, r, u, P, searcher)
	S = empty(S0)
	for (v, r) in S0
		for s in 1:nsteps
			i = rand(1:length(v))
			e = v[1:end .!= i]
			u = randray(PP[e])
			if (u' * (PP[v[i]] - PP[e[1]])) > 0
				u = -u
			end
			vv, t = raygen(e, r, u, PP)
			if t < Inf
				v = vv
				r = r + t*u
				push!(S, (v=>r))
			end
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

function connectivity_matrix(vertices, P)
	conns = adjacency(vertices)
	@show length(conns)
	#Ahv = boundaries(vertices, conns, P)
	I = Int[]
	J = Int[]
	V = Float64[]
	Vs = zeros(length(P))
	for ((g1,g2), sigs) in conns
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
