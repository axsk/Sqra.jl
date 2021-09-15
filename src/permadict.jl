using FileIO
using Logging

struct PermaDict
	prefix::String
end

mutable struct PermaSettings
	write::Bool
	read::Bool
end

const PERMADICT = PermaSettings(false, false)

PermaDict() = PermaDict("cache/")

function Base.empty!(d::PermaDict)

end

function CACHE!(read=true, write=true)
	PERMADICT.write = write
	PERMADICT.read = read
	read, write
end



function with_perma(f, read=true, write=true)
	r = PERMADICT.read
	w = PERMADICT.write
	CACHE!(read, write)
	x=f()
	CACHE!(r, w)
	return x
end

macro cache(exp)
	dump(exp)
end

function Base.get!(f, d::PermaDict, k)
	fn = d.prefix * string(mhash(k)) * ".jld2"
	if PERMADICT.read && isfile(fn)
		#@info("reading $fn")
		v = load(fn, "output")
	else
		e = @elapsed v = f()
		if PERMADICT.write
			@info("writing $fn")
			save(fn, #"input", k,
				"output", v, "elapsed", e)
		end
	end

	return v
end

#=
function Base.get!(f, d::PermaDict, k, settings=PERMADICT)
	if settings.read && haskey(d.d, k)
		@info("found cached entry")
		v = d.d[k]
	else
		fn = d.prefix * string(mhash(k)) * ".jld2"
		if settings.read && isfile(fn)
			@info("found saved entry $fn")
			v = load(fn, "output")
		else
			v = f()
			if settings.save
				save(fn, "input", k, "output", v)
				@info("saved new entry $fn")
			end
		end
		d.d[k] = v
	end
	return v
end
=#

# workaround mutable structs
# alternatively use AutoHashEquals.jl
function mhash(x::T, y = UInt64(0)) where T
	if fieldcount(T) > 0
		mapfoldr(z->getfield(x,z), mhash, fieldnames(T), init=y)
	else
		hash(x, y)
	end
end
