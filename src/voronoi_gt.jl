using LinearAlgebra
using NearestNeighbors

function voronoi(x::Matrix, iter=1000, particles=1, tmax=1000, eps=1e-8)
	P = colrows(x)
	@time searcher = NNSearch(tmax, eps, KDTree(x))
	@time s0 = descent(P, P[collect(1:particles)], searcher)
	@time s1 = walk(s0, iter, P, searcher)
	@time v = enumeratesig(s1, P)
	P, v
end

struct NNSearch
	tmax::Float64
	eps::Float64
	tree::KDTree
end


""" generate a random ray orthogonal to the subspace spanned by the given points """
function randray(x)
	k = length(x)
	d = length(x[1])
	v = similar(x, k-1)

	for i in 1:k-1
		v[i] = x[i] .- x[k]
		for j in 1:(i-1)
			v[i] .= v[i] .- dot(v[i], v[j]) .* v[j]
		end
		normalize!(v[i])
	end
	u = randn(d)
	for i in 1:k-1
		u = u - dot(u, v[i]) * v[i]
	end
	normalize!(u)
	return u
end

""" shooting a ray in the given direction, find the next connecting point """
function raycast_bruteforce(sig, r, u, P)
	(tau, ts) = [], Inf
	x0 = sig[1]
	for x in P
		x in sig && continue
		t = (sum(abs2, r .- x) - sum(abs2, r .- x0)) / (2 * u' * (x-x0))
		if 0 < t < ts
			(tau, ts) = vcat(sig, [x]), t
		end
	end

	# begin # check if new point is equidistant to its generators
	# 	rr = r + ts*u
	# 	diffs = [sum(abs2, rr.-s) for s in tau]
	# 	if !allapprox(diffs)
	# 		@show diffs
	# 		@show tau
	# 		@show ts
	# 		@show u
	# 		error()
	# 	end
	# end
	return sort(tau), ts
end

using NearestNeighbors
function raycast_intersect(sig, r, u, P, searcher::NNSearch)
	tau, tl, tr = [], 0, searcher.tmax
	x0 = sig[1]
	#iter = 0
	while tr-tl > searcher.eps
		tm = (tl+tr)/2
		idxs, dists = knn(searcher.tree, r+tm*u, 1, false)
		x = P[idxs[1]]
		if x in sig
			tl = tm
		else
			tr = (sum(abs2, r .- x) - sum(abs2, r .- x0)) / (2 * u' * (x-x0))
			tau = vcat(sig, [x])
		end
		#iter += 1
	end
	#@show iter
	if tau == []
		tr = Inf
	end
	return sort(tau), tr
end

allapprox(x) = all(isapprox(x[1], y) for y in x)

""" starting at given points, run the ray shooting descent to find vertices """
function descent(PP, P, searcher)
	raygen(sig, r, u, P) = raycast_intersect(sig, r, u, P, searcher)
	d = length(P[1])
	Sd1 = [[xi] for xi in P]
	Sd2 = [xi for xi in P]
	for k in d:-1:1
		Sdm1 = []
		Sdm2 = []
		for (sig, r) in zip(Sd1, Sd2)
			u = randray(sig)
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
			u = randray(e)
			if (u' * (v[i] - e[1])) > 0
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

""" given vertices in generator-coordinates,
collect the verts belonging to generator pairs, i.e. boundary vertices """
function extractconn(sigs)
	conns = Dict()
	for (sig, r) in sigs
		for a in sig
			for b in sig
				a == b && continue
				a < b && continue
				v = get!(conns, sort([a,b]), [])
				push!(v, sig)
			end
		end
	end
	conns
end

using Polyhedra
function boundaries(vertices, conns, P)
	Ahv = map(collect(conns)) do ((g1,g2), inds)
		coords = map(i->vertices[i], inds)
		push!(coords, P[g1])  # append voronoi center for full volume
		p = polyhedron(vrep(coords))
		V = volume(p)
		plot!(p)

		h = norm(P[g1] - P[g2])
		A = 2 * V / h
		A, h, V
	end
	return Ahv
end

using SparseArrays

function connectivity_matrix(vertices, conns, P)
	Ahv = boundaries(vertices, conns, P)
	I = Int[]
	J = Int[]
	V = Float64[]
	Vs = zeros(length(P))
	for ((A, h, v), (g1,g2)) in zip(Ahv, keys(conns))
		push!(I, g1)
		push!(J, g2)
		push!(V, A/h)
		Vs[g1] += v
		Vs[g2] += v
	end
	A = sparse(I, J, V, length(P), length(P))
	A = A + A'
	Vsi = 1 ./ Vs
	A = A .* Vsi
	return A, Ahv, Vs
end



using Plots
# this works after filtering out the boundary-outbound vertices
function test(n=5, iter=10000)
	plot(legend=false);
	x = hcat(hexgrid(n)...)
	x .+= randn(2,n*n) .* 0.01
	P, v  = voronoi(x, iter)

	v = Dict(filter(collect(v)) do (k,v)
		norm(v) < 10
		end)

	c = extractconn(v)
	A, Ahv, Vs = connectivity_matrix(v, c, P)

	AA = map(x->x>.0, A)
	plot_connectivity!(AA .* 2, P)
	scatter!(eachrow(hcat(values(v)...))...)
	xlims!(1,6); ylims!(0,5)
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







colrows(x) = Vector{Float64}.((collect(eachcol(x))))

function toid(P, x)
	d = Dict(v=>i for (i,v) in enumerate(P))
	d[x]
end

function enumeratesig(sigs, P)
	Dict(map(x->toid(P,x), k) => v for (k,v) in sigs)
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
