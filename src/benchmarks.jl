using Profile
using Sqra

#Profile.clear()
#@profile run(Simulation)
#Profile.print()


function pprint(needle="")
    io = IOBuffer()
    c=IOContext(io, :displaysize=>(1000,160))
    Profile.print(c)
    # eachline(io)
    s = String(take!(io))
    s = split(s, "\n")
    s = filter(x->contains(x, needle), s)
    #for s in s
    #n(s)
    #end

end
