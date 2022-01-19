# testing nested gradient computation with zygote
# testing different combinations of gradient nesting

import Zygote, ForwardDiff, Flux
using BenchmarkTools

f = Flux.Chain(Flux.Dense(2,10,σ), Flux.Dense(10,10), Flux.Dense(10,1))

p, builder = Flux.destructure(f)

x = rand(2,1)

fg(f, x) = ForwardDiff.gradient(f, x)
function bg(f, x)
    df, = Zygote.gradient(f, x)
    df
end

bgi(f, x) = Zygote.gradient(f, x)[1]

function bench()

t1 = @benchmark fg(p->sum(abs2, bg(x->builder(p)(x)[1], x)), p)
#@benchmark bf = bg(p->sum(abs2, fg(x->builder(p)(x)[1], x)), p)
t2 = @benchmark fg(p->sum(abs2, fg(x->builder(p)(x)[1], x)), p)
t3 = @benchmark bg(p->sum(abs2, bg(x->builder(p)(x)[1], x)), p)

end


function model_b(p)
    fx = builder(p)
    y, pb = Flux.pullback(fx, x)
    df, = pb(ones(size(x,2))')
    sum(abs2, df, dims=1)
end

function model_f(p)
    fx = builder(p)
    df = map(eachcol(x)) do x
    ForwardDiff.gradient(f, x)
    end
    ForwardDiff.jacobian(fx, x)
    sum(abs2, df, dims=1)
end
