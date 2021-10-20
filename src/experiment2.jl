

@with_kw struct Setup2{T}
	model = TripleWell()
	x0 = x0default(model)
	sampler = EMSimulation(model)
	discretization::T = SqraVoronoi()
	committor = Committor()
end

@with_kw struct SqraVoronoi
	npick = 100
	tmax = 10
end

@with_kw struct SqraSparseBox
	level = 6
end

@with_kw struct Committor
	solveriter = 1000
end

function Experiment(setup::Setup2)
	samples, u = run(setup.sampler, setup.model, setup.x0)
	Q, x, report = discretize(setup, samples, u)
	c = committor(setup, (Q,x ))
	(; (@locals)...)
end

function discretize(setup::Setup2{SqraVoronoi}, x, u)
	npick = setup.discretization.npick
	tmax = setup.discretization.tmax
	x, idxs, _ = picking(x, npick)
	v, P = VoronoiGraph.voronoi(x, ;tmax=tmax)
    Q = sqra_voronoi(u[idxs], beta(setup.model), v, P)
	#A, Vs = connectivity_matrix(v, P)
	#Q = sqra(u[idxs], A, beta(setup.model))
	return (Q, x, (;v, P))
end

function discretize(setup::Setup2{SqraSparseBox}, x, u)
	sb = SparseBoxes(x, setup.discretization.level, setup.model.box)
	Q, picks = sqra(sb, u, sigma(setup.model))
	return (Q, x[:, picks], (;picks, sb))
end

function committor(setup::Setup2, (Q, x))
	model = setup.model
	classes = classify(model, x)
	c = committor(Q, classes, maxiter = setup.committor.solveriter)
end
