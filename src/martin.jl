

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


uold(x) = abs(norm(x) - 1/2)^2 * (norm(x) < 1/2)
u(x) = p(norm(x)*2) * (norm(x) < 1/2)
#k(x) = exp(-norm(x)^(-alpha))
k(x) = 1.
f(y) = tr(jacobian(x->k(x) * gradient(u, x), y))  # f = ∇⋅(k∇u)
hessu(y) = jacobian(x->gradient(u,x), y)

function setup(;
    N = 200,
    alpha = 2,
    D = 2)

    xs = [SVector{D}(randn(D)) for i in 1:N] / 4  # sample normally distributed
    xs = xs[sortperm(norm.(xs))]  # sort xs by radius

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

## inverse problem: f, k |-> u
function inverseproblem(;
    Q = nothing,
    xs = nothing,
    fs = nothing,
    us = nothing,
    bndinner = 0,
    bndouter = 1/2,
    kwargs...)

    # modified system with bnd conditions
    QQ = copy(Q)
    b = copy(fs)
    vs = copy(us)

    inner = bndinner .<= norm.(xs) .<= bndouter

    # boundary condition
    for i in findall(.!inner)
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

    return (;merge(kwargs, Base.@locals)...)
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
    @show h = maximum(R)

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

    return (;merge(kwargs, Base.@locals)...)
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

function pbatch(;D=4, m=5, ns=[100,200,400,800,1600,3200,6400,12800])
    res = [[] for i in 1:Threads.nthreads()]
    ns = repeat(ns, inner=m)
    #try
        Threads.@threads for n in ns
            @show n
            e = errorstats(inverseproblem(setup(N=n, D=D)))
            Base.push!(res[Threads.threadid()], e)
        end
    #catch
    #end
    @show length.(res)
    reduce(vcat, res)
end

bigbatch() = pbatch(D=4, m=10, ns=[100,200,400,800,1600,3200,6400,12800,25_000,100_000])

function plot_h_i(b)
    ih = plot()
    plot!(ih, legend=false)
    xaxis!(L"I_2", :log)
    yaxis!(L"H_{T,\kappa}",:log)
    for e in b
        scatter!(ih, [e.I], [e.H], marker_z=log10(e.N))
    end
    mn, mx = extrema([e.I for e in bs])
    yn, yx = [1, mx/mn] .* minimum(e.H for e in bs)
    plot!(ih, [mn, mx], [yn, yx])
    yticks!([yn, yx])
    xticks!([mn, mx])
    ih
end

function plot_n_h(b)
    st = plot()
    xaxis!(L"\# K")
    yaxis!(L"\Vert u_\mathcal{T}-\mathcal{R}_{\mathcal{T}}u\Vert_{H_{\mathcal{T},\kappa}}")
    yaxis!(:log)
    xaxis!(:log)
    plot!(legend=false)

    for e in b
        scatter!(st, [e.N], [e.H])
    end
    st
end

function plotbatch(b)
    hi = plot_h_i(b)
    nh = plot_n_h(b)

    l = @layout[a; b]
    plot(hi, nh, layout = l, size=(800,1000))
end
