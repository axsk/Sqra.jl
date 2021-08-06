using FileIO

struct PermaDict{T}
	d::T
	prefix::String
end

PermaDict(d=Dict(), prefix="cache/") = PermaDict(d, prefix)

function Base.empty!(d::PermaDict)
	empty!(d.d)
end

function Base.get!(f, d::PermaDict, k)
	if haskey(d.d, k)
		println("found cached entry")
		v = d.d[k]
	else
		fn = d.prefix * string(mhash(k)) * ".jld2"
		if isfile(fn)
			println("found saved entry")
			v = load(fn, "output")
		else
			v = f()
			save(fn, "input", k, "output", v)
			println("saved new entry")
		end
		d.d[k] = v
	end
	return v
end

# workaround mutable structs
# alternatively use AutoHashEquals.jl
function mhash(x::T, y = UInt64(0)) where T
	if fieldcount(T) > 0
		mapfoldr(z->getfield(x,z), mhash, fieldnames(T), init=y)
	else
		hash(x, y)
	end
end
