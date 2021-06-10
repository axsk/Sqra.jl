using JuMP: MathOptInterface
using StatsBase: extrema
using JuMP
#import Tulip
import Clp

"""
    voronoi_adjacency(g[, tolerance])

Compute the adjacency matrix of a voronoi tesselation of given centroids by
the linear program by Lie.

# Arguments
- `g`: (n,d) dimensional arrays containing the n centroids in d dimensional space
- `tolerance`: Tolerance for detection (f_i < 0 - tolerance)

# Returns
- `N`: (n,n) array containing binary adjacency relations

# Sources
- Lie, F., W. - A Square Root [...] (2013) http://epubs.siam.org/doi/abs/10.1137/120899959
"""
function voronoi_adjacency(X, tolerance=0)
    N, _ = _voronoi_adjacency(X, tolerance)
    return N
end

function _voronoi_adjacency(X, tolerance=1e-6)
    g = X'
    n, d = size(g)
    b = vec(sum(abs2, g, dims=2))
    A = hcat(2*g, -ones(n))
    v = zeros(n,n)

    for i in 1:n
        e_i = zeros(n)
        e_i[i] = 1
        for j in i+1:n
            #m = Model(Tulip.Optimizer)
            m = Model(Clp.Optimizer)
            set_optimizer_attribute(m, "LogLevel", 0)
            #set_optimizer_attribute(m, "Algorithm", 4)
            @variable(m, x[1:(d+1)])
            @objective(m, Min, b[i] - A[i,:]' * x)
            @constraint(m, poly, b+e_i - A*x .>= 0)
            @constraint(m, edge, b[j] - A[j,:]' * x == 0)
            optimize!(m)
            v[i,j] = objective_value(m)
        end
    end
    v = v + v'
    N = v .< -tolerance
    return N, v
end

#= since below idea seems to be hard, we will probably go with voronoi above on the
subset of edegs who are close enough, maybe something like 2*max min dist, from the picking.
or d = median(sort(dists, dims=2))[n] where n is the number of expected neighbours in dimension d.

=#

#=
# test for implementing a better scaling version of the lp
# the main idea is to use shadow costs, or the sensivity of the obj wrt to the bcs
# (in particular the hyperplane conditions) to restore multiple/all neighbors at once
# this seems to fail however. imagine a polyhedron in the 2d plane,
# we can find some corners easily by a lp, but finding the others without knowing a priori
# in which direction to look is hard
# plan 1. remove the direction of the hyperplane/edge from the objective function that we touched
# however, this only leads us to d (dimension space) vertices, before eliminating all search directions.
# plan 2. rotate the objective function as to find the next edge. whilst this makes sense in 2d, it is not clear how to go about this in higher d
=#

function voronoi_fast(X)
    d, n = size(X)
    N = zeros(Int, n, n)
    for i in 1:n
        N[i,:] = draft(X,i)
    end
    N
end
function draft(X, i=1)
    g = X'
    n, d = size(g)
    b = vec(sum(abs2, g, dims=2))
    A = hcat(2*g, -ones(n))
    v = zeros(n,n)

    conn = BitVector(zeros(n))

    m = Model(Clp.Optimizer)
    set_optimizer_attribute(m, "LogLevel", 0)

    @variable(m, x[1:(d+1)])
    @constraint(m, poly, b - A*x .>= 0)
    @constraint(m, edge, b[i] - A[i,:]' * x == 0)

    # take the sum of all hyberplanes as objective
    obj = -sum(A, dims=1) |> vec

    for i=1:1000# true
        @objective(m, Min, obj' * x)
        optimize!(m)
        if termination_status(m) != MathOptInterface.OPTIMAL
            break
        end
        price = shadow_price.(poly)

        # remove the direction we just found from the searchdirection
        mod = (A'*price)
        #mod[end] = 0
        obj = obj - mod
        conn[price .< 0] .= 1
    end
    return conn
end


function testruns(
    ns = [10, 50, 100, 200, 300, 400, 500],
    ds = [2,3,6])
    neig = zeros(length(ns), length(ds))
    time = copy(neig)
    try
        for (i,n) in enumerate(ns)
            for (j,d) in enumerate(ds)
                @show n, d
                g = rand(d, n)
                time[i,j] = @elapsed A = @time voronoi_adjacency(g)[1]
                @show neig[i,j] = sum(A) / n
            end
        end
    catch
    end
    neig, time
end

include("cmdtools.jl")
using StatsBase
hist(x) = fit(Histogram, x, minimum(x):1:maximum(x))

using Random
function statistics(nsample, npick, dim, seed=rand(UInt))
    x = rand(MersenneTwister(seed), dim, nsample)
    y = picking(x, npick)
    A = voronoi_adjacency(y)
    N = degree(A)
    #md, sd = mean_and_std()

    dists = []

    for i = 1:npick
        connected = A[:,i] .== 1
        deltas = y[:,i] .- y[:, connected]
        dists = append!(dists, mapslices(norm, deltas, dims=1))
    end

    return y, A, N, mean_and_std(dists), extrema(dists)

end

edgedists(x, A) = [norm(x[:,i] - x[:,j]) for (i,j) in edges(A)]

degree(A) = sum(A, dims=1) |> vec


edges(A::AbstractMatrix) = map(Tuple, findall(A.==1))
edges(inds::Vector) = inds

function plotvoronoi(x, A)
    scatter(x[1,:],x[2,:], markersize=degree(A), legend=false)
    plotedges!(x,A)
end

function plotedges!(x, A)
    for (i,j) in edges(A)
        plot!(x[1,[i,j]],x[2,[i,j]], label=nothing)
    end
    return plot!()
end

function pruneedges(x, A, dist)
    e = edges(A)
    ds = edgedists(x, e)
    pruned = e[ds .<= dist]
    prunees = e[ds .> dist]
    return pruned, prunees
end



using Memoize

@memoize function batchstatistics(nsample, npick, dim, nbatch)
    n = []
    for i in 1:nbatch
        n = vcat(n, statistics(nsample, npick, dim, i))
    end
    n

    Accumulator((x=>v/nbatch/npick for (x,v) in counter(n))...)

end

function nplot(nsample, npick, dim, nbatch)
    label = "$npick"
    plot!(batchstatistics(nsample, npick, dim, nbatch), label=label)
end
