using Sqra


s = run(Simulation())
d = discretize(SpBoxDiscretisation(), s)
c = Sqra.committor(d) 


include("sparseboxes.jl")