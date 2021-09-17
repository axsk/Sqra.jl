using LinearAlgebra

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

function raygen(sig, r, u, P)
	(tau, ts) = [], Inf
	x0 = sig[1]
	for x in P
		x in sig && continue
		t = (sum(abs2, r .- x) - sum(abs2, r .- x0)) / (2 * u' * (x-x0))
		if 0 < t < ts
			(tau, ts) = vcat(sig, [x]), t
		end
	end

	begin # check if new point is equidistant to its generators
		rr = r + ts*u
		diffs = [sum(abs2, rr.-s) for s in tau]
		if !allapprox(diffs)
			@show diffs
			@show tau
			@show ts
			@show u
			error()
		end
	end
	return sort(tau), ts
end

allapprox(x) = all(isapprox(x[1], y) for y in x)

function descent(PP, P = PP[[1]])
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
				println("invert direction")
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
	mscat(Sd2[1])
	return [(a,b) for (a,b) in zip(Sd1, Sd2)]  # transform to array of tuples
end

mscat(x) = scatter!(eachrow(x)...)

function walk(S0, nsteps, PP)
	S = Set(S0)
	for (v, r) in S0
		for s in 1:nsteps
			@show i = rand(1:length(v))
			e = v[1:end .!= i]
			u = randray(e)
			if (u' * (v[i] - e[1])) > 0
				u = -u
			end
			vv, t = raygen(e, r, u, PP)
			if t < Inf
				v = vv
				r = r + t*u
				push!(S, (v,r))
			end
		end
	end
	return S
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

function plotface!(sig, r)
	for s in sig
		plot!([s[1], r[1]], [s[2], r[2]], linestyle=:dash)
	end
	plot!()
end

function extractconn(sigs)
	conns = Set()
	for s in sigs
		for a in s
			for b in s
				a == b && continue
				push!(conns, (a,b))
			end
		end
	end
	conns
end

function plotconns!(conns)
	for c in conns
		a,b = c
		x1, y1 = a
		x2, y2 = b
		plot!([x1, x2], [y1, y2])
	end
end

function delmap(sigs)
	d = Dict()
	for (i, s) in enumerate(sigs)
		for x in s
			v = get!(d, x, [])
			push!(v, i)
		end
	end
	d
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


function voronoi(x::Matrix)
	P = map(collect, sort(collect(eachcol(x))))
	descent(P, P)
end



function test()
	x=[0 1 0.
          0 0 1]
	voronoi(x)
end
