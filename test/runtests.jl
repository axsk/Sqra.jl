using Sqra

function runtests()

	include("sparseboxes.jl")

	s = run(Simulation())
	d = discretize(SpBoxDiscretisation(), s)
	c = Sqra.committor(d)
end

runtests()
