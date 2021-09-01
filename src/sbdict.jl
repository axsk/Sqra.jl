using StaticArrays, SparseArrays, DataStructures


struct SparseBoxesDict{D}
	level::Int
	boundary::Matrix{Float64}
    dict::D
end

function SparseBoxesDict(x::Matrix, level::Int, boundary::Matrix=autoboundary(x))
    carts = cartesiancoords(x, level, boundary)
    d = dict(carts)
    SparseBoxesDict(level, boundary,d)
end

boxes(b::SparseBoxesDict) = keys(b.dict)
boxmatrix(b::SparseBoxesDict) = reduce(hcat, collect(boxes(b)))
inds(b::SparseBoxesDict) = values(b.dict)


function merge(a, b, offset)
    @assert a.level == b.level
	@assert a.boundary == b.boundary

    d = copy(a.dict)

    for (k,v) in b.dict
        vv=get!(d, k, Int[])
        append!(vv, v .+ offset)
    end


    SparseBoxesDict(a.level, a.boundary, d)
end

adjacency(d::SparseBoxesDict) = neighbours(d.dict)

dict(x::Matrix{Int}) = dict(x, SVector{size(x,1), Int})

function dict(x::Matrix{Int}, ::Type{T}) where {T}
	d = SortedDict{T, Vector{Int}}()
	for (i, col) in enumerate(eachcol(x))
		c = T(col)
		vs = get!(d, c, Int[])
		push!(vs, i)
	end
	d
end

function neighbours(dict::AbstractDict)
	I = Int[]
	J = Int[]
	ids = SortedDict((k=>i) for (i,k) in (enumerate(keys(dict))))
    D = length(first(keys(dict)))
	v = zeros(Int,D)
	for (k,i) in ids
		copy!(v, k)
		for dim in 1:D
			v[dim] += 1
			j = get(ids, v, 0)
			if j > 0
				push!(I, i)
				push!(J, j)
			end
			v[dim] -= 1
		end
	end
	A = sparse(I, J, ones(Int, length(I)), length(ids), length(ids))
	A + A'
end

#=
using Sqra
function performance2(n=1_000_000)
	s=run(Simulation(nsteps=n))
	x=s.x
	#@profile begin

	@time begin
		b=SparseBoxes(s.x, 6)
		A = Sqra.adjacency(b)
	end
    dicts = [SortedDict, Dict]
    x = Sqra.cartesiancoords(s.x, 6)

    @time begin
        d = dict(x)
        B = neighbours(d)
    end

	A,B

end
=#