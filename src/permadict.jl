using FileIO
using Logging

struct PermaDict{T}
	d::T
	prefix::String
end

mutable struct PermaSettings
	save::Bool
end

const ENABLE_PERMADICT = PermaSettings(false)

PermaDict(prefix="cache/") = PermaDict(Dict(), prefix)

function Base.empty!(d::PermaDict)
	empty!(d.d)
end

function Base.get!(f, d::PermaDict, k)
	if ENABLE_PERMADICT.save == false
		return f()
	end
	if haskey(d.d, k)
		@info("found cached entry")
		v = d.d[k]
	else
		fn = d.prefix * string(mhash(k)) * ".jld2"
		if isfile(fn)
			@info("found saved entry $fn")
			v = load(fn, "output")
		else
			v = f()
			save(fn, "input", k, "output", v)
			@info("saved new entry $fn")
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
