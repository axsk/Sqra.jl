using SpecialFunctions

mutable struct MCError{T}
    offset
    sum::T
    sumsq::T
    n::Int64
    sigmaint::Float64
end


MCError(conf=.95) = MCError(0., 0., 0., 0, erfinv(conf) * sqrt(2))

function push!(e::MCError, x)
    e.sum += x
    e.sumsq += x^2
    e.n += 1
end

mean(e::MCError) = e.sum / e.n
var(e::MCError) = (e.sumsq - e.sum^2 / e.n) / (e.n-1)
intervall(e::MCError) = 2.58 * 1.2 * sqrt(var(e)) / sqrt(e.n)
relerror(e::MCError) = intervall(e) / abs(mean(e))
converged(e::MCError; reltol=1e-6, abstol=1e-6) = (abs(mean(e)) * reltol + abstol) > intervall(e)

using VoronoiGraph
using VoronoiGraph: Vertices, Points, neighbors, RaycastBruteforce

function mc_integrate(f, σ::Vertices, xs::Points, nmin, nmax, n_points, reltol, abstol)
    n = length(xs)
    ys = zeros(n)
    ∂ys = zeros(n)

    neigh = VoronoiGraph.neighbors(σ)

    for i in 1:n
        ix = [i; neigh[i]]
        j = 0
        e = MCError()
        while true
            j += 1
            y, ∂y, V, A = VoronoiGraph.mc_integrate(f, 1, xs[ix], 1, n_points, RaycastBruteforce())

            push!(e, y)
            if V == Inf
                @show mean(e)
                break
            end
            # if isnan(mean(e))
            #     error()
            #     @show e, y
            #     break
            # end
            if e.n >= nmax || (e.n > nmin && converged(e, reltol=reltol, abstol=abstol))
                @show e.n, intervall(e), mean(e)
                break
            end
            yield()
        end
        ys[i] = mean(e)
    end
    return ys
end
