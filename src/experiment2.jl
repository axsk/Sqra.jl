

@with_kw struct Setup2{T}
	model = TripleWell()
	x0 = x0default(model)
	sampler = EMSimulation(model)
	discretization::T = SqraVoronoi()
	committor = Committor()
end

@with_kw struct SqraVoronoi
	npick = 100
	viter = npick * 100
	vstuck = npick * 10
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
	Q, x = discretize(setup, samples, u)
	committor = cmt(setup, (Q,x ))
	(; (@locals)...)
end

function discretize(setup::Setup2{SqraVoronoi}, x, u)
	npick = setup.discretization.npick
	viter =	setup.discretization.viter
	vstuck = setup.discretization.vstuck
	tmax = setup.discretization.tmax

	x, idxs, _ = picking(x, npick)
	v, P = Voronoi.voronoi(x, viter; maxstuck = vstuck, tmax=tmax)
	A, Vs = Voronoi.connectivity_matrix(v, P)
	Q = sqra(u[idxs], A, beta(setup.model))
	return (Q, x)
end

function discretize(setup::Setup2{SqraSparseBox}, x, u)
	sb = SparseBoxes(x, setup.discretization.level, setup.model.box)
	Q, picks = sqra(sb, u, sigma(setup.model))
	return (Q, x[:, picks])
end

function cmt(setup::Setup2, (Q, x))
	model = setup.model
	classes = classify(model, x)
	cmt = committor(Q, classes, maxiter = setup.committor.solveriter)
end