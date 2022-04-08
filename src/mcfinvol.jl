""" testing the finite volume approximation with monte carlo estimated fluxes instead of sqra """

using VoronoiGraph, Sqra
using LinearAlgebra, Arpack, Plots, Random


""" n-dim doublewell, with double well in first dim. and harm potential in all others """
doublewell(x::Real) = (x^2 - 1) ^ 2
doublewell(x) = doublewell(x[1]) + sum(abs2, 2* x[2:end])


DoubleWellTrial(;
    d=2,
    x0 = zeros(d),
    n = 1000,
    sigma = 1.,
    dt = 0.1,
    nmc = 10,
    nmc2 = 1,
    seedsample = rand(UInt),
    seedmc = rand(UInt)
    ) = (;Base.@locals()...)


""" monte carlo finite volumes

compute a finite volume approximation to the generator Q
by means of monte carlo integration of π along the cell boundaries """
function mcfv(setup=DoubleWellTrial())
    (; d, x0, n, sigma, dt, nmc, nmc2, seedsample, seedmc) = NamedTuple(setup)
    beta = 2/sigma^2

    Random.seed!(seedsample)
    xs, us = Sqra.eulermaruyama(x0, doublewell, sigma, dt, n; maxdelta=Inf, progressbar=true)
    @assert all(isfinite.(xs))

    perm = sortperm(xs[1,:])
    xs = xs[:,perm]

    Random.seed!(seedmc)
    Q = mc_finitevolume(xs, doublewell, beta, nmc, nmc2)

    v, X = eigs(Q, which=:SM)
    myscat(xs, marker_z = X[:,2] |> real) |> display

    return Base.@locals() |> NamedTuple
end

using VoronoiGraph: mc_integrate, vecvec
using Sqra: divide_by_h!, fixdiagonal
using SparseArrays
using ProgressMeter

"""
mc_finitevolume(xs, u, beta, nmc=100, nmc2=1)

given samples `xs` in R^(d x n), a potential function `u` and coldness `beta`
return the generator `Q` computed by a finite volume approximation on the induced voronoi grid,
estimated by monte carlo sampling on the volumes/boundaries.
`nmc` is the number of rays per point (surface integral)
`nmc2` the number of samples per ray (volume integral)"""

function mc_finitevolume(xs, u, beta, nmc=100, nmc2=1)
    xs = vecvec(xs)

    V = zeros(length(xs))
    A = spzeros(length(xs), length(xs))

    f1(x) = exp(-beta * u(x))

    @showprogress for i in 1:length(xs)
        vi, ai, _, _ = mc_integrate(f1, i, xs, nmc, nmc2)
        V[i] = vi
        A[:, i] = ai
    end

    Q = Diagonal(V .* beta)^-1 * A

    divide_by_h!(Q, xs)
    Q = fixdiagonal(Q)
    return Q
end


"""
square root approximation using either deterministic (nmc == 0) or monte carlo estimated
areas and volumes
"""
function trial(setup=DoubleWellTrial())
    (; d, x0, n, sigma, dt, nmc) = NamedTuple(setup)
    beta = 2/sigma^2

    xs, us = Sqra.eulermaruyama(x0, doublewell, sigma, dt, n; maxdelta=Inf, progressbar=true)
    @assert all(isfinite.(xs))

    perm = sortperm(xs[1,:])
    xs = xs[:,perm]
    us = us[perm]
    xs = VoronoiGraph.vecvec(xs)

    if nmc > 0
        A, V = VoronoiGraph.mc_volumes(xs, nmc)
    else
        v, p = VoronoiGraph.voronoi(xs)
        A, V = VoronoiGraph.volumes(v,p)
    end

    C = Sqra.sqra_weights(A, V, xs)
    Q = Sqra.sqra(us, C, beta)

    sel = cutQ(Q)
    Q = Q[sel, sel]
    xs = xs[sel]
    us = us[sel]

    # preconoditioning
    # D = Diagonal(sqrt.(exp.(-beta .* us)))
    # D = Diagonal(ones(length(us)))
    # QQ = D * Q * inv(D)
    # X = inv(D) * X

    v, X = eigs(Q, which=:SM)

    xx = real(X[:,2])
    xs = reduce(hcat, xs) |> collect
    myscat(xs, marker_z = xx) |> display

    return Base.@locals() |> NamedTuple
end

function unpack(r::NamedTuple)
    for (a,b) in pairs(r)
        eval(:($a=$b))
    end
end

function cutQ(Q)
    i = findall(Q[i,i] != 0 for i in 1:size(Q,1))
    return i
end

myscat(xs; kwargs...) = scatter(xs[1,:], xs[2,:]; kwargs...)
