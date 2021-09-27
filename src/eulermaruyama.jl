using ProgressMeter
using ForwardDiff, DiffResults

@with_kw struct EMSimulation
	dt = 0.1
	N = 100
	maxdelta = Inf  # adaptive stepsize control if < Inf
	seed = 1
	progressbar = true
end

@memoize PermaDict("cache/sim_") function run(sim::EMSimulation, model, x0)
	Random.seed!(sim.seed)

	x, u = eulermaruyama(x0, x->potential(model, x), sigma(model), sim.dt, sim.N,
		maxdelta = sim.maxdelta,
		progressbar = sim.progressbar)

end

#=
function extend(s::Simulation, n)
	e = Simulation(s, x0 = s.x[:, end], nsteps=n)
	e = run(e)
	Simulation(s, x=hcat(s.x, e.x), u = vcat(s.u, e.u), nsteps=s.nsteps+n)
end
=#


function eulermaruyama(x0::AbstractVector, potential::Function, sigma::Real, dt::Real, steps::Integer; maxdelta=Inf, progressbar=true)
    dim = length(x0)
    p = Progress(steps; dt=1, desc="EM Sampling ", enabled=progressbar)

    grad = DiffResults.GradientResult(x0)
    cfg = ForwardDiff.GradientConfig(potential, x0)

    x = copy(x0)
    xs = similar(x0, dim, steps)
	us = similar(x0, steps)

    for t in 1:steps
		ForwardDiff.gradient!(grad, potential, x, cfg)
		us[t] = DiffResults.value(grad)
		g = DiffResults.gradient(grad)

		delta = sqrt(sum(abs2, g))  * dt
		if delta > maxdelta
			n = ceil(Int, delta / maxdelta)
			x = eulermaruyama(x, potential, sigma, dt/n, n, maxdelta=maxdelta, progressbar=false)[1][:, end]
		else
			x .+= -g * dt .+ sigma * sqrt(dt) * randn(dim)
		end

		xs[:, t] = x
        next!(p)
		#yield()
    end
	us = [us[2:end]; potential(x)]  # because we stored the potential of the previous x

    return xs, us
end
