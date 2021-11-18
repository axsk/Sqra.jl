using SparseArrays
using LinearAlgebra

""" Convenience wrapper for the SQRA,
implicitly computing the connectivity for the N-D Array `u` based on a regular grid """
function sqra(u::Matrix, beta::Real)
    A = grid_adjacency(size(u))
    sqra(vec(u), A, beta)
end

""" Compute the Square-Root approximation to the generator of Langevin diffusion
for a given potential vector `u` and connectivity matrix `A` at coldness `beta`.
Any desired rate scaling ϕ can be factored into `A`.
"""
function sqra(u::Vector, A::SparseMatrixCSC, beta::Real)
    I, J, a = findnz(A)
    v = -beta / 2 .* u
    q = similar(u, size(a))
    for n in 1:length(q)
        q[n] = exp(v[J[n]] - v[I[n]]) * a[n]
    end
    Q = sparse(I, J, q, size(A)...)
    Q = fixdiagonal(Q)
end

function grid_adjacency(dims::NTuple{N, Int} where N)
    dims = reverse(dims) # somehow we have to take the krons backwards, dont know why
    k = []
    for i in 1:length(dims)
        x = [spdiagm(ones(s)) for s in dims] # identity in all dimensions
        x[i] = spdiagm(-1 => ones(dims[i]-1), 1=>ones(dims[i]-1)) # neighbour matrix in dimension i
        push!(k, kron(x...))
    end
    sum(k)
end

function fixdiagonal(Q)
	Q = Q - spdiagm(diag(Q)) # remove diagonal
    Q = Q - spdiagm(sum(Q, dims=2)|>vec) # rowsum 0
	return Q
end

# deprecated (?)
# since underdetermined systems are not a problem with iterative solvers
function prune_Q(Q, lim)
	pinds = zeros(Bool, size(Q, 1))

	# keep only small outbound rates
	pinds[-lim .< diag(Q) .< 0] .= 1    # TODO: handle .= 0 below

	noutbound = size(Q,1)-sum(pinds)
	#println("pruned $noutbound large outbound rates / unconnecteds")

	# prune unconnceted cells
	while true
		QQ = Q[pinds, pinds]
		QQ = QQ - Diagonal(QQ)

		rem = (sum(QQ, dims=1)|>vec .== 0)
		pinds[findall(pinds)[rem]] .= 0

	 	(sum(rem) == 0) && break
	end

	nunconn = size(Q,1) - sum(noutbound) - sum(pinds)
	#println("pruned $nunconn states without incoming rates")


	Q[pinds, pinds]

	Q = Q[pinds, pinds]
	Q = fixdiagonal(Q)
	return Q, pinds
end

#=
using PyCall
@pyimport cmdtools

function test_compare()
    u = rand(2,3)
    q1 = sqra(u, 1) |> Matrix
    q2 = cmdtools.estimation.sqra.SQRA(u, 1).Q.todense()
    q1, q2 # note that these wont be equal because python uses other flattening scheme
end
=#

"""
    sqra_voronoi(u, beta, xs; nmc=1000)
Compute the voronoi diagram, approximate the volumes with `nmc` samples each and return the SQRA
"""
function sqra_voronoi(u, beta, xs; nmc=0)
    @assert length(u) == size(xs, 2)
    v, P = VoronoiGraph.voronoi(xs)
    if nmc > 0
        A, V = VoronoiGraph.mc_volumes(v, P, nmc)
    else
        A, V = VoronoiGraph.volumes(v, P)
    end
    C = sqra_weights(A, V, P)
    return sqra(u, C, beta)
end

"""
    sqra_weights(A, Vs, P)

Given the boundary areas, volumes and centers, compute the SQRA weight matrix C
``C_ij = A_ij / (V_i * h_ij)``
"""
function sqra_weights(A, Vs, P)
    C = Diagonal(1 ./ Vs) * A
    divide_by_h!(C, P)

    # for unbounded boundary cells, set inbound rates to 0
    for (i, v) in enumerate(Vs)
        v < Inf && continue
        C[:, i] .= 0
    end

    return C
end

""" divide the area of each boundary area by the distance its generators """
function divide_by_h!(A, P)
    rows = rowvals(A)
    for j in 1:size(A, 2)
        for i in nzrange(A, j)
            i = rows[i]
            A[i,j] /= norm(P[i] - P[j])
        end
    end
end
