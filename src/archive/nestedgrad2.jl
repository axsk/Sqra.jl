using Flux
using ReverseDiff
using Zygote

x = rand(2)
m = Chain(Dense(2,1))


Flux.gradient(params(m)) do
    gradient(m,x) |> sum |> sum
end

ReverseDiff.gradient(theta0) do theta0
    u = u_from_theta(theta0)
    Flux.gradient(u, x)[1] |> sum
end

y, pb = Flux.pullback(params(m)) do
    ReverseDiff.gradient(m, x) |> sum
end

pb(1)


## double reversediff on flux model
# even with with DiffEqFlux:
#ERROR: MethodError: *(::ReverseDiff.TrackedArray{Float32,Float32,2,Array{Float32,2},Array{Float32,2}}, ::ReverseDiff.TrackedArray{Float64,Float64,1,Array{Float64,1},Array{Float64,1}}) is ambiguous.

    theta0, u_from_theta = Flux.destructure(m)
    ReverseDiff.gradient(theta0) do theta0
        u = u_from_theta(theta0)
        ReverseDiff.gradient(u, x) |> sum
    end

## ReverseDiff on flux diff
# works

theta0, u_from_theta = Flux.destructure(m)
dudt = ReverseDiff.gradient(theta0) do theta0
    u = u_from_theta(theta0)
    Flux.gradient(x->u(x)[1], x)[1] |> sum
end

using Flux

m = Chain(Dense(2, 100, σ), Dense(100,100, σ), Dense(100, 1, σ))
x = rand(2,1000)
loss(f,x) = sum(abs2, Flux.gradient(x->f(x) |> sum, x) |> sum)

@benchmark Flux.pullback(params(m)) do
    loss(m,x)
end[2](1)

@benchmark Flux.pullback(params(m)) do
    m(x)
end[2](1)