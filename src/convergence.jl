function compare(setups::Vector{<:Setup2})
	n = length(setups)
	errs = fill(NaN, n)

  	ref = Experiment(setups[end])

	for i in 1:n-1
		try
			e = Experiment(setups[i])
			err = error(ref, e)
			errs[i] = err
		catch end

	end

	plot(errs)
end

function compare(experiments)
	errs = map(experiments) do e
		#try
			error(experiments[end], e)
		#catch
		#	NaN
		#end
	end
	ns = [size(e.x, 2) for e in experiments]
	ns, errs
end

function compare_experiments()
	ev = [@time Experiment(Sqra.Setup2(discretization=Sqra.SqraVoronoi(npick=n), sampler=Sqra.EMSimulation(N=100_000, dt=.1)))   for n in logspace(Int, 4, 10000, 10)]
	es = [@time Experiment(Sqra.Setup2(discretization=Sqra.SqraSparseBox(level=n), sampler=Sqra.EMSimulation(N=100_000, dt=.1))) for n in logspace(Int, 2, 100, 10)]
	ev, es
end

function crosscompare()
	ev, es = compare_experiments()
	plot(xaxis=:log)
	plot!(Sqra.compare([ev; ev[end]]), label="V|V")
	plot!(Sqra.compare([es; ev[end]]), label="V|B")
	plot!(Sqra.compare([es; es[end]]), label="B|B")
	plot!(Sqra.compare([ev; es[end]]), label="B|V")
end
