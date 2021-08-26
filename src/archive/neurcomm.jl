module NN

using Flux
using Plots
using LinearAlgebra
using StatsBase
import Flux.Zygote.@ignore

import Flux.Zygote.hook
import Flux.Zygote.dropgrad
import Flux.Zygote.@adjoint

export test_variational
export test_sampled
export test_sqra

export Langevin
export triplewell, Triplewell, triplewellbox
export RadialCrisp, RadialExp
export CommittorSampled, CommittorSQRA, CommittorVariational

export losses
export randuniform

### plotting

dim(m::Flux.Chain) = size(m.layers[1].W, 2)
dim(m) = 2

function boxgrid(bounds, nx=40, ny=30)
    xs = range(bounds[1,:]..., length = nx)
    ys = range(bounds[2,:]..., length = ny)
    zs = zeros(2, length(xs) * length(ys))

    i=1
    for y in ys
        for x in xs
            zs[1,i] = x
            zs[2,i] = y
            i+=1
        end
    end

    xs, ys, zs
end


function plot(m, bounds, nx=40, ny=30)
    xs, ys, zs = boxgrid(bounds, nx, ny)

    zs = reshape(m(zs), length(xs), length(ys))'
    heatmap(xs, ys, zs)
    contour!(xs, ys, zs, linewidth=2)
end

### Boundary Types

abstract type Boundary end

function boundaryfixture(b::Boundary, f)
    function (x)
        a, _, r = getboundary(b, x)
        f(x) .* r .+ a
    end
end

## Crisp Boundary

struct CustomCrisp{T} <: Boundary
    bndcheck::T
end
getboundary(b::CustomCrisp, x) = crispboundary(b.bndcheck, x)

struct RadialCrisp{T<:AbstractVector} <: Boundary
    a::T
    b::T
    r::Float64
end
getboundary(c::RadialCrisp, x) = crispboundary(x->bndcheck(c, x), x)

function bndcheck(c::RadialCrisp, x::AbstractVector)
    if sum(abs2, c.a - x) <= c.r^2
        1
    elseif sum(abs2, c.b - x) <= c.r^2
        0
    else
        -1
    end
end

function crispboundary(bndcheck, x)
    bnd = map(bndcheck, eachcol(x))'
    a = bnd .== 1
    b = bnd .== 0
    r = 1 .- a .- b
    a, b, r
end
@adjoint crispboundary(bndcheck, x) = crispboundary(bndcheck, x), delta->(nothing, nothing)

## Smooth Boundary

struct RadialExp{T<:AbstractVector} <: Boundary
    a::T
    b::T
    e::Float64
end

getboundary(c::RadialExp, x) = smoothboundary(c.a, c.b, c.e, x)
function smoothboundary(a, b, e, x)
    a = exp.(e * sum(abs2, x .- a, dims=1)) # set a with bnd = 1
    b = exp.(e * sum(abs2, x .- b, dims=1)) # set b with bnd = 0
    r = 1 .- a .- b
    a, b, r
end


### Committor Types

## SQRA Committor
struct CommittorSQRA{F, B}
    potential::F
    h::Float64
    beta::Float64
    boundary::B
end

function losses(c::CommittorSQRA, f, x)

    fb = boundaryfixture(c.boundary, f)

    losses = sqraloss_batch(fb, x, c.potential, c.h, beta=c.beta)

    # weight losses only inside the solution domain
    _, _, r = getboundary(c.boundary, x)
    losses = losses .* r
end

function sqraloss_batch(f, x, u, h; beta=5)
    n,m = size(x)
    stencil = dropgrad(hcat(zeros(n), diagm(h * ones(n)), diagm(-h * ones(n))))
    testpoints = reshape(x, n, 1, m) .+ stencil |> x->reshape(x, n, m*(2n + 1))
    ut = u(testpoints) |> x->reshape(x, 2*n+1, m)
    ft = f(testpoints) |> x->reshape(x, 2*n+1, m)
    w = exp.(-1/2 * beta * (ut .- ut[1, :]'))
    #w[1,:] = -sum(w[2:end, :], dims=1)
    w = vcat(-sum(w[2:end, :], dims=1), w[2:end,:])
    loss = abs.(sum(w .* ft, dims=1))
end

## Sampled Committor

struct CommittorSampled{B}
    boundary::B
end

function losses(c::CommittorSampled, f, data::Array{T, 3}) where T
    f = boundaryfixture(c.boundary, f)
    (i,j,k) = size(data)
    xs = reshape(data, i, j*k)
    ys = reshape(f(xs), j, k)

    losses = abs.(mean(ys[1,:]' .- ys[2:end,:], dims=1))

    _, _, r = getboundary(c.boundary, data[:,1,:])
    losses .* r
end

## Variational Committor

struct CommittorVariational{B,R}
    boundary::B
    reweight::R # dpi/dx
end

function losses(c::CommittorVariational, f, x::Matrix)
    f = boundaryfixture(c.boundary, f)
    y, pb = Flux.pullback(f, x)
    dx, = pb(ones(size(x,2))')
    l2 = sum(abs2, dx, dims=1)
    r = c.reweight(x)
    l2 .* r'
end

### TEST THE VARIATIONAL PART, also see the nestedgrad*.jl files

import FiniteDifferences
import ForwardDiff

function losses_f(c::CommittorVariational, f, x::Matrix)
    f = boundaryfixture(c.boundary, f)
    df = map(eachcol(x)) do x
        ForwardDiff.gradient(x->f(x)[1], x) end
    df = hcat(df...)
    l2 = sum(abs2, df, dims=1)
    r = c.reweight(x)
    l2 .* r'
end

function test_losses_variational()
    p = Langevin(triplewell, RadialExp([1.,0],[-1.,0], -10.), triplewellbox, 5)
    c = CommittorVariational(p)
    m = mlp([10,10])
    x = rand(2,10)

    @assert isapprox(losses(c,m,x), losses_f(c,m,x); rtol=1e-6)

    p, builder = Flux.destructure(m)
    fp(p) = sum(abs2, losses(c, builder(p), x))

    # 2x back
    ps = Flux.params(m)
    l1, pb = Flux.pullback(ps) do
        sum(abs2, losses(c, m, x))
    end
    df1 = pb(1).grads[m[1].W]

    # forward-back
    df2 = ForwardDiff.gradient(fp, p)
    l2  = sum(abs2, losses_f(c, m, x))

    # 2x back with destructure
    l3, pb3 = Flux.pullback(fp, p)
    df3 = pb3(1)[1]

    df4 = FiniteDifferences.grad(FiniteDifferences.central_fdm(5,1), fp, p)[1]

    @assert df1[1] == df2[1] == df3[1]
    df1, df2, df3, df4 , l1, l2, l3
end
#df1, df2, df3, df4, l1, l2, l3 = NN.test_losses_variational();

### END OF VARIATIONAL TESTS


#### utility

function randbox(bounds, n)
    shift = bounds[:,2]
    scale = (bounds[:,2] - bounds[:,1])
    rand(size(bounds, 1), n) .* scale .- shift
end

function mlp(x, in=2, sig=false)
    first = Dense(in, x[1], σ)
    if sig
        last = Dense(x[end], 1, σ)
    else
        last = Dense(x[end], 1)
    end
    Chain(first, [Dense(x[i], x[i+1], σ) for i=1:length(x)-1]..., last)
end

using Parameters

@with_kw struct Langevin
    potential
    boundary
    box
    beta
end


CommittorVariational(L::Langevin) = CommittorVariational(L.boundary, x -> exp.(-L.beta * L.potential(x)))
CommittorSampled(L::Langevin) = CommittorSampled(L.boundary)


potential(p::Langevin) = p.potential
box(t::Langevin) = t.box
boundary(t::Langevin) = t.boundary
sample(t::Langevin, n) = randbox(t.box, n)
randdata(t::Langevin, branches, dt, steps) = RandomData(n->sampletrajectories(t, n, branches, dt=dt, steps=steps))
randuniform(t::Langevin) = RandomData(n->sample(t, n))

""" batch sampling of trajectories for the process `t`.
At each of the `n` sampled start points compute `branches` trajectories,
with step-size `dt` and `steps` steps.
returns an `N` x (`steps`+1) x `n` array, where `N` is the states space dimension"""
function sampletrajectories(t::Langevin, n, branches; dt=1, steps=1)
    xs = sample(t, n)
    u = potential(t)
    sigma = sqrt(2/t.beta)
    data = zeros(2, branches+1, n)
    for i in 1:n
        x = xs[:,i]
        data[:, 1, i] = x
        for j in 1:branches
            data[:, 1+j, i] = eulermaruyama(x, u, sigma, dt, steps)
        end
    end
    data
end




function test_sqra(p::Langevin = Triplewell(); hidden=[10,10], h=.1, r=.1, samples=100, plotevery=10, epochs=1000)
    model = NN.mlp(hidden)
    c = CommittorSQRA(p.potential, h, p.beta, RadialCrisp([1,0],[-1,0], r))
    #data = randbox(triplewellbox, samples)
    data = RandomData(n->randbox(triplewellbox, n))#sampletrajectories(t, n, branches, dt=dt, steps=steps))
    train(model, c, data, bounds=triplewellbox, plotevery=plotevery, batch=samples, epochs=epochs)
end

function test_sampled(p::Langevin = Triplewell(); hidden=[10,10], samples=100, branches=4, plotevery=10, dt=.1, steps=1, epochs=1000)
    model = mlp(hidden)
    c = CommittorSampled(boundary(p))
    #data = sampletrajectories(p, samples, branches, dt=dt, steps=steps)
    data = randdata(p, branches, dt, steps)
    train(model, c, data, bounds=p.box, plotevery=plotevery, batch=samples, epochs=epochs)
end

function test_variational(process::Langevin = Langevin(triplewell, RadialExp([1.,0],[-1.,0], -5.), triplewellbox, 1); hidden=[10,10], samples=100, plotevery=10, epochs=10)
    model = NN.mlp(hidden)
    c = CommittorVariational(process)
    data = randuniform(process)
    train(model, c, data, bounds=process.box, plotevery=plotevery, batch=samples, epochs=epochs)
end

function train(model, c, data;
    epochs=1,
    batch=1000,
    bounds=[],
    opt = ADAM(0.001),
    plotevery=1
    )

    ps = Flux.params(model)
    losshist = []

    for i in 1:epochs
        for x in Flux.Data.DataLoader(data, batchsize=batch)
            local pointlosses

            l, pb = Flux.pullback(ps) do
                pointlosses = losses(c, model, x)
                sum(abs2, pointlosses) / size(x, 2)
            end

            push!(losshist, l)
            grad = pb(1)
            Flux.Optimise.update!(opt, ps, grad)

            fm = boundaryfixture(c.boundary, model)
            if length(losshist) % plotevery == 0
                visualize(fm, bounds, pointlosses, x, losshist)
            end
        end
    end
    model, losshist
end




function visualize(model, bounds, pointlosses, x, losshist)
    maxloss = maximum(pointlosses)
    p1 = plot(model, bounds)
    if isa(x,Matrix{Float64})
        p1 = Plots.scatter!(x[1,:], x[2,:], legend=false,
            marker=((pointlosses' / maxloss) * 10, stroke(0)))
    elseif isa(x, Array{T,3} where T)
        Plots.scatter!(x[1,1,:], x[2,1,:], legend=false,
            marker=((pointlosses' / maxloss) * 10, stroke(0)))
        plottrajs(x[:,:,1:20])
    end
    p2 = Plots.plot(losshist, yscale=:log10)
    Plots.plot(p1, p2, layout=@layout([a;b]), size=(800,800)) |> display
end

function plottrajs(data)
    _, n, m = size(data)
    x = zeros(2, (n-1)*m)
    x[1,:] = repeat(data[1,1,:], inner=n-1)
    x[2,:] = data[1,2:n,:]

    y = zeros(2, (n-1)*m)
    y[1,:] = repeat(data[2,1,:], inner=n-1)
    y[2,:] = data[2,2:n,:]

    Plots.plot!(x, y, legend=false, color=:dodgerblue)
end


import Zygote.gradient

function eulermaruyama(x, potential, sigma, dt, steps)
    for i in 1:steps
        force = -gradient(potential, x)[1]
        x .+=  force * dt .+ sigma * randn(size(x)) * sqrt(dt)
    end
    x
end

struct RandomData{T}
    generator::T
end

function Flux.Data.DataLoader(d::RandomData; batchsize=1)
    [d.generator(batchsize)]
end

function (d::RandomData)(n)
    d.generator(n)
end



export train

# concrete models

MullerBrown() = Langevin(potential=mullerbrown, boundary = RadialCrisp([0.6,0], [-0.55,1.4], .1), box=mullerbrownbox)

const mullerbrownbox = [-1.5 1; -.5 2]
function mullerbrown(x::AbstractMatrix)
    A = (-200, -100, -170, 15)
    a = (-1, -1, -6.5, .7)
    b = (0, 0, 11, .6)
    c = (-10, -10, -6.5, .7)
    x0 = (1, 0, -.5, -1)
    y0 = (0, .5, 1.5, 1)

    x, y = x[1,:], x[2,:]
    sum(@. A[k] * exp(a[k]*(x-x0[k])^2 + b[k]*(x-x0[k])*(y-y0[k]) + c[k]*(y-y0[k])^2) for k=1:4)
end

mullerbrown(x::AbstractVector) = mullerbrown(reshape(x, (2,1)))[1]
mullerbrown(x, y) = mullerbrown([x,y])

## Triplewell

Triplewell(beta=2.) = Langevin(triplewell, RadialCrisp([1.,0],[-1.,0], .1), triplewellbox, beta)

const triplewellbox = [-3. 3; -2 2]

function triplewell(x::AbstractVector)
    x, y = x
    V =  (3/4 * exp(-x^2 - (y-1/3)^2)
        - 3/4 * exp(-x^2 - (y-5/3)^2)
        - 5/4 * exp(-(x-1)^2 - y^2)
        - 5/4 * exp(-(x+1)^2 - y^2)
        + 1/20 * x^4 + 1/20 * (y-1/3)^4)
end

function triplewell(x::Matrix)
    x, y = x[1,:], x[2,:]
    V = @. (3/4 * exp(-x^2 - (y-1/3)^2)
        - 3/4 * exp(-x^2 - (y-5/3)^2)
        - 5/4 * exp(-(x-1)^2 - y^2)
        - 5/4 * exp(-(x+1)^2 - y^2)
        + 1/20 * x^4 + 1/20 * (y-1/3)^4)
end

function triplewellcustombnd(v)
    x, y = v
    if abs(y) < 0.2
        if 0.8 < x < 1.2
            return 1.
        elseif -.8 > x > -1.2
            return 0.
        end
    end
    return NaN
end

function triplewellcustombnd(x::AbstractMatrix)
    triplewellbnd.(eachcol(x))
end
@adjoint triplewellcustombnd(x::AbstractMatrix) = triplewellcustombnd(x), d->(nothing, )#(zeros(size(@show d)),)

# Hyperwell
#=
function hyperproblem(n=2; h=.1, beta=5.)
    a = ones(n)
    b = ones(n)
    b[1] = -1
    CommittorSmoothBoundary(hyperwell, h, beta, a, b, 30.)
end =#

hyperwell(x) = 0.1*sum((x.^2 .-1).^2, dims=1)


end # module

using .NN
