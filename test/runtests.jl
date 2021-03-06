using Test
using Sqra

@testset "Experiment" begin
	@time Experiment(Setup())
end

#@testset "Voronoi" begin
#	Sqra.Voronoi.tests()
#end

@testset "Error computation" begin
	e1 = Sqra.Experiment(Setup())
	e2 = Sqra.VExperiment(Setup())

	@test_broken begin
		for a in [e1, e2], b in [e1,e2]
			Sqra.error(e1,e2)
		end
	end
end

@testset "Setup2 TripleWell" begin
	m = Sqra.TripleWell()
	for d in [Sqra.SqraVoronoi(), Sqra.SqraSparseBox()]
		s = Sqra.Setup2(model=m, discretization=d)
		@test (Sqra.Experiment(s); true)
	end
end


#=
@testset "Sparse Boxes" begin
	include("sparseboxes.jl")
end
=#
#=
@testset "SpBoxDiscretisation" begin
	s = run(Simulation())
	d = discretize(SpBoxDiscretisation(), s)
	c = Sqra.committor(d)
end

@testset "Refactor" begin
	s = run(Simulation())
	d = discretize(SpBoxDiscretisation(), s)
	c = Sqra.committor(d)

	e = Experiment(Setup())

	@test c == e.committor
end
=#
