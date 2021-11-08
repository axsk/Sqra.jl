using Plots

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
