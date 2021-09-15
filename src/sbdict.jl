using StaticArrays, SparseArrays, DataStructures

struct SparseBoxesDict{D}
	level::Int
	boundary::Matrix{Float64}
    dict::D
end

function Base.show(io::IO, sb::SparseBoxesDict)
	Base.print(io, "SparseBoxes of level $(sb.level) with $(length(sb.dict)) elements")
end

function SparseBoxesDict(x::Matrix, level::Int, boundary::Matrix=autoboundary(x))
    carts = cartesiancoords(x, level, boundary)
    d = sbdict(carts, level)
    SparseBoxesDict(level, boundary,d)
end

boxes(b::SparseBoxesDict) = keys(b.dict)
boxmatrix(b::SparseBoxesDict) = reduce(hcat, collect(boxes(b)))
inds(b::SparseBoxesDict) = values(b.dict)
level(b::SparseBoxesDict) = b.level


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

sbdict(x::Matrix{Int}, level) = sbdict(x, level, SVector{size(x,1), Int})

function sbdict(x::Matrix{Int}, level, ::Type{T}) where {T}
	d = SortedDict{T, Vector{UInt32}}()
	for (i, col) in enumerate(eachcol(x))
		c = T(col)
		all(1 .<= c .<= level) || continue
		vs = get!(d, c, [])
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
