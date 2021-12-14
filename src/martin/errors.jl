
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
    L = L_2(v,P,vs)
    LL = sqrt(sum(L[inner])) ## TODO: change names to Ls, L. Kept for compatibility


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
    for (i,j,a) in spinds(areas)
        i == j && continue
        #a = areas[i,j]
        #a == q || throw(Exception)
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

L_2(e::NamedTuple) = sqrt(sum(L_2(e.v, e.P, e.us, 100, 10)[e.inner]))

function L_2(verts, P, us, nmc=100, nmc2=10)
    kernel(x, i) = abs2(u(x) - us[i])
    L = VoronoiGraph.mc_integrate_i(kernel, verts, P, nmc, nmc2)[1]
    return L
end
