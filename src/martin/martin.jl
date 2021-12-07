

using Sqra
using IterativeSolvers
using LinearAlgebra
using ForwardDiff: jacobian, gradient
using Plots
using StaticArrays
using VoronoiGraph
using SparseArrays
using LaTeXStrings

p(x) = 2 * (x+1/2) * (1-x)^2

#u(x) = p(norm(x)*2) * (norm(x) < 1/2)
u(x) = p(norm(x)) * x[1] * (norm(x) < 1)
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
    B = zeros(Int, length(V))

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

## inverse problem: f, k |-> u
function inverseproblem(;
    Q = nothing,
    xs = nothing,
    fs = nothing,
    us = nothing,
    V = nothing,
    v = nothing,
    kwargs...)

    # modified system with bnd conditions
    QQ = copy(Q)
    b = copy(fs)
    vs = copy(us)

    # boundary condition
    B = boundary(v, V)
    inner = findall(B.==0)
    for i in findall(B.==1)
    #for i in 1:length(xs)
    #    if !(bndinner < norm(xs[i]) < bndouter)
            QQ[i, :] .= 0
            QQ[i, i] = 1
            b[i] = us[i]
    #    end
    end

    # solve linear system
    _, hist = IterativeSolvers.gmres!(vs, QQ, b; maxiter=100000, log=true)
    !hist.isconverged && @warn "Solver did not converge"

    #@show norm(QQ * vs - b)

    #plot(us)
    #plot!(vs)

    step = cld(length(us), 500)
    X = map(x->x[1], xs)
    Y = map(x->x[2], xs)
    #=
    scatter(X[1:step:end], Y[1:step:end], marker_z=(vs-us)[1:step:end])
    =#

    #@show norm((us-vs) / length(us))

    #=

    plot((us), labels=L"R_T u");
    plot!((vs), label=L"u_T")
    title!("reconstruction error")
    xlabel!("K") |> display
    =#

    return (;merge(kwargs, delete!(Base.@locals(), :kwargs))...)
end

errorstats(d) = errorstats(;d...)
function errorstats(;
    v=nothing, P=nothing,
    A = nothing,
    ks = nothing,
    us = nothing,
    vs = nothing,
    inner = nothing,
    kwargs...)



    r, R = excentricity(v, P)
    h = maximum(R)

    Hs = H_Tk_sq(vs - us, A, ks, P)
    H = sqrt(sum(Hs[inner]))
    Is = I_2(v, P)
    I = sqrt(sum(Is[inner]))

    #=
    plot(R./r, label = L"C_\rm{uni}")
    title!("Voronoi regularity")
    plot!(R, label="R")|>display
    =#

    #=
    plot(I[inner], yaxis=:log, label="I2")
    plot!(H[inner], label="H") |> display
    =#

    return (;merge(kwargs, delete!(Base.@locals(), :kwargs))...)
end

function excentricity(verts, P)
    n = length(P)
    r = fill(Inf, n)
    R = zeros(n)

    for (gens, coords) in verts
        for g in gens
            p = P[g]
            d = norm(p - coords)
            r[g] > d && (r[g] = d)
            R[g] < d && (R[g] = d)
        end
    end

    return r, R
end

#=
""" ‖v‖²ₕₜₖ = ∑_σ m_σ h_σ k_σ |∂_σ v|²
 = Q_ij * π_i * m_i * beta """
function H_Tk_sq_Q(vs, Q, vols, ks)
    H = zeros(length(vs))
    beta = 1
    for (i,j,q) in spinds(Q)
        H[i] += q * vols[i] * ks[i] * beta * (vs[i] - vs[j]) ^ 2
    end
    return H / 2
end
=#

function H_Tk_sq(vs, areas, ks, P)
    H = zeros(length(vs))
    for (i,j,q) in spinds(areas)
        i == j && continue
        a = areas[i,j]
        a == Inf && continue
        H[i] += a / norm(P[i]-P[j]) * sqrt(ks[i]*ks[j]) * (vs[i] - vs[j]) ^ 2
    end
    return H / 2
end

H_Tk(e::NamedTuple) = sqrt(sum(H_Tk_sq((e.vs - e.us), e.A, e.ks, e.P)[e.inner]))

using Continuables
"""
    spinds(A)
like enumerate(A) but for sparse arrays, iterating (i,j,v) """
function spinds(A)
    c = @cont begin
        rows = rowvals(A)
        vals = nonzeros(A)
        for j in 1:size(A, 2)
            for r in nzrange(A, j)
                i = rows[r]
                v = vals[r]
                cont((i, j, v))
            end
        end
    end
    return aschannel(c)
end

function I_2(verts, P, nmc=100, nmc2=10)
    n = length(P)
    d = length(P[1])

    I = zeros(n)

    r, R = excentricity(verts, P)

    du = VoronoiGraph.mc_integrate(x -> sum(abs2, hessu(x)), verts, P, nmc, nmc2)[1]
    neigh = VoronoiGraph.neighbors(verts)

    for K in 1:length(P)
        I[K] = R[K]^2 * (R[K]/r[K])^(d+1) * du[K] * length(neigh[K])
    end

    return I
end

I_2(e::NamedTuple) = sqrt(sum(I_2(e.v, e.P, 100, 10)[e.inner]))

function hconvergence(D=4)
    st = plot()
    xaxis!(L"\# K")
    yaxis!(L"\Vert u_\mathcal{T}-\mathcal{R}_{\mathcal{T}}u\Vert_{H_{\mathcal{T},\kappa}}")
    yaxis!(:log)
    xaxis!(:log)
    plot!(legend=false)

    ih = plot()
    plot!(ih, legend=false)
    yaxis!(L"I_2", :log)
    xaxis!(L"H_{T,\kappa}",:log)
    #xlims!((1e-5,1))
    #ylims!((1e-5,50))

    for j in [100,200,400,800,1600,3200,6400,12800], i in 1:5
        e=errorstats(inverseproblem(setup(N=j, D=D)))
        scatter!(ih, [e.H], [e.I], color=Int(log2(j/100))) |> display
        @show e.I, e.H
        scatter!(st, [e.N], [e.H]) |> display
    end

    return st
end

function batch(;D=4, m=5, ns=[100,200,400,800,1600,3200,6400,12800])
    [
        errorstats(inverseproblem(setup(N=n, D=D)))
        for j in 1:m, n in ns
    ]
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
    e = errorstats(inverseproblem(setup(xs=xs)))
    return e
    return strip(e)
end



using ThreadPools

function qbatch(;D=4, seeds=1:5, ns=[100,200,400,800,1600,3200,6400,12800])
    qmap((n, seed) for seed in seeds, n in reverse(ns)) do (n, seed)
        tm = @elapsed e = experiment(n, D, seed)
        println("finished n=$n - time=$(now()) - taken $tm - thread=$(Threads.threadid())")
        flush(Base.stdout)
        return e
    end
end

function pbatch(;D=4, m=5, ns=[100,200,400,800,1600,3200,6400,12800])
    res = [[] for i in 1:Threads.nthreads()]
    restemp = []
    ns = shuffle(repeat(ns, inner=m))
    prog = Progress(sum(ns))
    #try
        Threads.@threads for n in ns
            local tm
            try
                #tm = @elapsed e = errorstats(inverseproblem(setup(N=n, D=D)))
                tm = @elapsed e = run(n, D)
                Base.push!(res[Threads.threadid()], e)
                Base.push!(restemp, e)
                if Threads.threadid() == 1

                    #plot_h_i(restemp) |> display
                end
                println("finished n=$n - time=$(now()) - taken $tm - thread=$(Threads.threadid())")
            catch e
                @warn e
                @show e
            end
            next!(prog, step=n)
        end
    #catch
    #end
    @show length.(res)
    reduce(vcat, res)
end

bigbatch() = pbatch(D=4, m=10, ns=[100,200,400,800,1600,3200,6400,12800,25_000,100_000])

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

function sample_reject(D,N,f)
    xs = [SVector{D}(rand(D)*2 .- 1) for i in 1:N]
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
            xsnew = [SVector{D}(rand(D)*2 .- 1) for i in 1:M]
            append!(xs, xsnew)
            append!(fs, f.(xsnew))
            append!(uni, rand(M))
        end
    end
end

function sample(d, n, method)
    if method == :normal
        xs = sample_normal(d, n)
    elseif method == :uniform
        xs = sample_reject(d, n, x->1)
    elseif method == :grad
        xs = sample_reject(d, n, x->norm(gradient(u, x)))
    elseif method == :hess
        xs = sample_reject(d, n, x->norm(hessu(x)))
    else
        @show method
        throw(Exception)
    end
    return xs
end

function scatter(xs::Vector{S}; kwargs...) where S <: SVector
    scatter(collect(eachrow(reduce(hcat, xs)))...; kwargs...)
end


using Dash
using DashHtmlComponents
using DashCoreComponents

function dashify(e)
    X = reduce(hcat, e.xs)
    p = PlotlyJS.plot(PlotlyJS.scatter(x=X[1,:],y=X[2,:],mode="markers", marker=attr(color=e.us)))
    app = dash(external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"])

    app.layout = html_div() do
            html_h1("Hello Dash"),
            html_div("Dash.jl: Julia interface for Dash"),
            dcc_graph(
                id = "example-graph",
                figure = p
            )
        end

    run_server(app, "0.0.0.0", 8080)
end
