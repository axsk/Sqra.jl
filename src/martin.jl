using Sqra
using IterativeSolvers
using LinearAlgebra
using ForwardDiff: jacobian, gradient
using Plots

u(x) = abs(norm(x) - 1/2)^2 * (norm(x) < 1/2)
#k(x) = exp(-norm(x)^(-alpha))
k(x) = 1.
f(y) = tr(jacobian(x->k(x) * gradient(u, x), y))  # f = ∇⋅(k∇u)

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
    return Base.@locals
end

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

    # boundary condition
    for i in 1:length(xs)
        if !(bndinner < norm(xs[i]) < bndouter)
            QQ[i, :] .= 0
            QQ[i, i] = 1
            b[i] = us[i]
        end
    end

    # solve linear system
    _, hist = IterativeSolvers.gmres!(vs, QQ, b; maxiter=10000, log=true)
    !hist.isconverged && @warn "Solver did not converge"

    @show norm(QQ * vs - b)

    plot(us)
    plot!(vs)

    plotstep = cld(length(us), 500)
    scatter(X[1,1:plotstep:end], X[2,1:plotstep:end], marker_z=(vs-us)[1:plotstep:end])

    @show norm((us-vs) / length(us))

    plot((us), labels="R_T u");
    plot!((vs), label="u_T")
    xlabel!("|x|") |> display

    return merge(kwargs, Base.@locals)
end

function errorstats(;
    v=nothing, P=nothing,
    Q = nothing,
    V = nothing,
    us = nothing,
    vs = nothing,
    kwargs...)

    inner = norm.(i[:xs]) .< .5

    r, R = excentricity(v, P)
    plot(R./r)
    plot!(R)|>display
    @show h = maximum(R)
    H = H_Tk(vs - us, Q, V)

    I = I_2(vs, v, P)
    plot(I[inner], yaxis=:log, label="I2")
    plot!(H[inner], label="H")
end


##

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



""" ‖v‖²ₕₜₖ = ∑_σ m_σ h_σ k_σ |∂_σ v|²
 = Q_ij * m_i * beta """
function H_Tk(v, Q, m)
    H = zeros(length(v))
    beta = 1
    for (i,j,q) in spinds(Q)
        H[i] = q * m[i] * beta * (v[i] - v[j]) ^ 2
    end
    return H
end

using Continuables
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

hessu(y) = jacobian(x->gradient(u,x), y)

function I_2(v, verts, P, nmc=100, nmc2=10)
    n = length(P)
    d = length(P[1])

    I = zeros(n)

    r, R = excentricity(verts, P)

    du = mc_integrate(x -> sum(abs2, hessu(x)), verts, P, nmc, nmc2)[1]
    neigh = VoronoiGraph.neighbors(verts)

    for K in 1:length(P)
        I[K] = R[K]^2 * (R[K]/r[K])^(d+1) * du[K] * length(neigh[K])
    end

    return I
end
