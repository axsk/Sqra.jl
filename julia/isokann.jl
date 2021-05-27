using Flux
using Flux.Optimise: update!
using Statistics
using Plots

" multi-layer perceptron with sigmoid activation function,
`x` is an array with the widths of the hidden layers,
`in` denotes the input dimension and
`sig` enables/disables sigmoid on last layer"
function mlp(x=[2,2], in=2, sig=false)
    first = Dense(in, x[1], σ)
    if sig
        last = Dense(x[end], 1, σ)
    else
        last = Dense(x[end], 1)
    end
    Chain(first, [Dense(x[i], x[i+1], σ) for i=1:length(x)-1]..., last, A->dropdims(A; dims=1))
end

function shiftscale(values)
    a,b = extrema(values)
    (values .- a) ./ (b-a)
end

using StatsBase

function koopmanexpectation(ys, koopman)
    shiftscale(dropdims(mean(koopman(ys), dims=1), dims=1))
end

import Zygote

""" single iteration of isokann (𝐊 ← 𝑺 𝔼[𝐊]) """
function poweriterate!(model, koopman, data, opt)
    ps = params(model)

    # split data into xs, ys and predict next koopman
    xs = data[:, 1, :]
    ys = data[:, 2:end, :]
    target = koopmanexpectation(ys, koopman)

    # compute loss/gradient
    loss, back = Zygote.pullback(ps) do
        predict = model(xs)
        mean(abs2, target - predict)
    end
    grad = back(one(loss))
    update!(opt, ps, grad)
    loss
end

""" single iteration of fixed point eqn (0 ← 𝑺 𝔼[𝐊] - 𝐊) """
function fixedpointiterate!(model, data, opt)
    ps = params(model)
    xs = data[:, 1, :]
    ys = data[:, 2:end, :]
    loss, back = Zygote.pullback(ps) do
        k = model(data)
        diff = (shiftscale(mean(k, dims=1)) |> vec) - k[1,:]
        mean(abs2, diff)
    end
    grad = back(one(loss))
    update!(opt, ps, grad)
    loss
end

#=
struct RandomBatch
    data
    batchsize
    repeats
end

Base.start(b::RandomBatch)
=#

function isokann(;model=mlp(), data=diffusivedata(), iter=100, poweriter=10, opt=Nesterov(0.1), cb=(model, loss)->nothing)
    ls = [fixedpointloss(model, data)]
    for i in 1:poweriter
        koop = deepcopy(model)
        for j in 1:iter
            loss = poweriterate!(model, koop, data, opt)
            push!(ls, loss)
            cb(model, ls)
        end
    end
    model, ls
end

function fpnn(;model=mlp(), data=diffusivedata(), iter=100, opt=Nesterov(0.1))
    ls = [fixedpointloss(model, data)]
    for i in 1:iter
        loss = fixedpointiterate!(model, data, opt)
        push!(ls, loss)
    end
    model, ls
end

function compareopt(;alg=isokann, repeats=20, kwargs...)
    d = Dict()
     for i in 1:repeats
         for (name, opt) in [("ADAM 1",ADAM(1)),("ADAM 0.1",ADAM(0.1)),("ADAM 0.01",ADAM(0.01)),
                             ("Nesterov 1",Nesterov(1)),("Nesterov 0.1",Nesterov(0.1)),("Nesterov 0.01",Nesterov(0.01))]
             push!(get!(d, name, []), alg(opt=opt; kwargs...))
         end
     end
     compareplot(d) |> display
     d
 end


function compareplot(d)
    plot()
    for (k,v) in d
        v = map(v->v[2], v)
        loss = median(reduce(hcat,v), dims=2) |> vec
        plot!(loss, yaxis=:log, label=k, ylims = (minimum(loss),10 * loss[1,1]))
    end
    plot!()
end


function fixedpointloss(model, data=diffusivedata(1000,100))
    xs = data[:, 1, :]
    ys = data[:, 2:end, :]
    target = koopmanexpectation(ys, model)
    predict = model(xs)
    mean(abs2, target - predict)
end

function lossstatistic(iters=10;isoargs...)
    data = diffusivedata(10000, 1000)
    mean_and_std([fixedpointloss(isokann(;isoargs...), data) for i in 1:10])
end

function diffusivedata(n=100, m=10)
    xs = rand(2,1,n)
    ys = randn(2,m,n) / 10 .+ xs
    hcat(xs, ys)
end

import Plots: heatmap

function heatmap(m::Chain)
    d = [m([x,y])[1] for x in 0:.1:1, y in 0:.1:1]
    heatmap(d)
end

# ------- lets tackle the tripplewell ------- #

include("neurcomm.jl")

data = NN.sampletrajectories(NN.Triplewell(), 100, 10, dt = 0.01, steps=10)