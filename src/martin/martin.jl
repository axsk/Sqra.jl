

using ..Sqra
using IterativeSolvers
using LinearAlgebra
using ForwardDiff: jacobian, gradient
using Plots
using StaticArrays
using VoronoiGraph
using SparseArrays
using LaTeXStrings

include("plots.jl")
include("errors.jl")

p(x) = 2 * (x+1/2) * (1-x)^2

#u(x) = p(norm(x)*2) * (norm(x) < 1/2)  # isotropic on -1/2, 1/2, the old one
u(x) = p(norm(x)) * x[1] * (norm(x) < 1)  # anisotropic on -1, 1
#u(x) = p(norm(x)) * (norm(x) < 1)  # isotropic on -1, 1
#k(x) = exp(-norm(x)^(-alpha))
k(x) = 1.
f(y) = tr(jacobian(x->k(x) * gradient(u, x), y))  # f = ∇⋅(k∇u)
hessu(y) = jacobian(x->gradient(u,x), y)


function setup(;
    N = 200,
    alpha = 2,
    D = 2,
    xs = [SVector{D}(randn(D)) for i in 1:N] / 2)

    D = length(xs[1])
    N = length(xs)

    #xs = [SVector{D}(randn(D)) for i in 1:N] / 2  # sample normally distributed
    #xs = xs[sortperm(norm.(xs))]  # sort xs by radius

    ks = k.(xs)
    fs = f.(xs)
    us = u.(xs)

    v, P = VoronoiGraph.voronoi(xs)
    A, V = VoronoiGraph.volumes(v, P)
    #A, V = VoronoiGraph.mc_volumes(v, P, nmc)

    C = Sqra.sqra_weights(A, V, P)

    beta = 1  # we need beta=1 since we didnt care about it in the other formulas here
    Q = Sqra.sqra(ks, C, beta)
    return (;Base.@locals()...)
end

#function mcvols(v, P, min=100, reltol=0.1)


forwardproblem(d) = forwardproblem(;d...)

## forward problem: u, k |-> f
function forwardproblem(;
    Q = nothing,
    fs = nothing,
    us = nothing,
    N = nothing,
    D = nothing,
    kwargs...)

    err = [fs Q*us]
    @show sqrt(sum(abs2, err[:,1] - err[:,2]) / N)
    @show N
    plot(err, title = "N=$N, D=$D", labels = ["Rf" "Q*u_T"])
end

inverseproblem(d) = inverseproblem(;d...)

function boundary(v, V)
    B = zeros(Bool, length(V))

    # vertices outside unit circle
    for (sig,v) in v
        if norm(v) > 1
            B[sig] .= 1
        end
    end

    # cells with verts at infinity
    B[findall(V .== Inf)] .= 1

    return B
end

function boundary_legacy(xs)
    norm.(xs) .> 1
end


## inverse problem: f, k |-> u
function inverseproblem(;
    Q = nothing,
    xs = nothing,
    fs = nothing,
    us = nothing,
    V = nothing,
    v = nothing,
    boundarymethod = :center,
    kwargs...)

    # modified system with bnd conditions
    QQ = copy(Q)
    b = copy(fs)
    vs = copy(us)

    # boundary condition
    if boundarymethod == :center
        B = boundary_legacy(xs)
    elseif boundarymethod == :intersect
        B = boundary(v, V)
    end

    inner = findall(B.==0)
    for i in findall(B.==1)
            QQ[i, :] .= 0
            QQ[i, i] = 1
            b[i] = us[i]
    end

    # solve linear system
    _, hist = IterativeSolvers.gmres!(vs, QQ, b; maxiter=1000, log=true)
    !hist.isconverged && @warn "Solver did not converge"

    #@show norm(QQ * vs - b)

    return (;merge(kwargs, delete!(Base.@locals(), :kwargs))...)
end

using ProgressMeter
using Random
using Dates
using Memoize

using Sqra: PermaDict
using Random

@memoize PermaDict("cache/martin_") function experiment(n=n, d=D, method=:uniform, seed=1)
    Random.seed!(seed)
    xs = sample(d,n,method)
    if method in [:uniform, :normal, :legacy]
        boundary = :center
    elseif method in [:grad, :hess]
        boundary = :intersect
    else
        throw(ArgumentError("invalid sampling method"))
    end
    e = errorstats(inverseproblem(;boundarymethod=boundary, setup(xs=xs)...))
    e = (;method, e...)
#   return e
    return strip(e)
end



using ThreadPools


function smallbatch()
    qbatch(D=4, seeds=1:2, ns=[8000,4000,2000,1000], methods=[:legacy])
end


bigbatch() = qbatch(; D=4, seeds=1:5, ns=reverse([100,200,400,800,1600,3200,6400,12800,25_000,50_000,100_000]))

function qbatch(;D=4, seeds=1:5, ns=[100,200,400,800,1600,3200,6400,12800], methods=[:uniform, :normal, :grad, :hess])
    setups = [(;n, D, method, seed) for seed in seeds, method in methods, n in ns]
    qbatch(setups)
end

function qbatch(setups)
    prog = Progress(sum(s.n for s in setups))
    qmap(setups) do s
        #@show s
        n, d, method, seed = s.n, s.D, s.method, s.seed
        local e
        try
            tm = @elapsed e = experiment(n, d, method, seed)
        catch err
            @warn "error in experiment"
            @show n,d,method,seed,err
            e = err
        end
        #println("finished n=$n - time=$(now()) - taken $tm - thread=$(Threads.threadid())")
        #flush(Base.stdout)
        next!(prog, step=n)
        #flush(Base.stdout)
        return e
    end
end

import JLD2

strip(e) = Base.structdiff(e, NamedTuple{(:C, :Q, :QQ, :A, :v)})

function save(b)
    bb = map(strip, b)
    JLD2.save("batch$(length(b)).jld2", "b", bb)
end

function sample_normal(D,N)
    xs = [SVector{D}(randn(D)) for i in 1:N] / 2
end

# sample uniform in [-1,1]^D
function sample_uniform(D,N)
    xs = [SVector{D}(rand(D)*2 .- 1) for i in 1:N]
end



function sample_reject(D,N,f, radius=1.5)
    xs = [SVector{D}((rand(D)*2 .- 1)*radius) for i in 1:N]
    fs = f.(xs)
    uni = rand(N)
    while true
        ratio = fs ./ maximum(fs)
        acc = ratio .> uni
        if sum(acc) >= N
            length(acc), sum((acc))
            return xs[acc][1:N]
        else
            M = ceil(Int, (N / sum(acc) - 1) * length(xs))
            xsnew = [SVector{D}((rand(D)*2 .- 1)*radius) for i in 1:M]
            append!(xs, xsnew)
            append!(fs, f.(xsnew))
            append!(uni, rand(M))
        end
    end
end

using LinearAlgebra

function sample(d, n, method)
    sigma = 1/2
    radius = 1.5
    if method == :normal
        xs = sample_reject(d, n, x->(norm(x)<radius) * exp(-sum(abs2,x)/(2 * (sigma^2))))
    elseif method == :uniform
        xs = sample_reject(d, n, x->(norm(x)<radius))
    elseif method == :grad
        xs = sample_reject(d, n, x->norm(gradient(u, x)))
    elseif method == :hess
        xs = sample_reject(d, n, x->norm(hessu(x)))
    elseif method == :legacy
        xs = [SVector{d}(randn(d) * sigma) for i in 1:n]
    else
        @show method
        throw(Exception)
    end
    return xs
end

function run()
    #b = qbatch(D=4, seeds=[1], ns=[8000,4000,2000,1000], methods=[:uniform, :normal, :legacy, :grad, :hess])
    b = qbatch(D=4, seeds=[1], ns=[8000,4000,2000,1000], methods=[:uniform, :normal, :legacy])
    p = plot_h_i(b) |> display
    (;b,p)
end

precompile(run, ())
