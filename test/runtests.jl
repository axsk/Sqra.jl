using Test
using Sqra

@testset "Experiment" begin
	@time Experiment(Setup())
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
