using Base: Float64

using SparseArrays

# mean squared error of two scalar functions (not densities!) on two respective
# sparse box distributions integrated over their common support
function sp_mse(x1, x2, cart1, cart2, res1, res2)
	v = sb_overlap(cart1, cart2, res1, res2)
	I,J,V = findnz(v)
	e = 0
	for i in 1:length(I)
		e += (x1[I[i]] - x2[J[i]])^2 * V[i]
	end
	e
end

sb_overlap(a, b, k, l) = sbv_linear(a,b,k,l)

##

""" calculate the 1-d overlap between regular discretizations of k and j subdivisions """
function calc_dists(k,l)
	x = zeros(k,l)
	for i in 1:k, j in 1:l
		l1, r1 = 1/k*(i-1), 1/k*i
		l2, r2 = 1/l*(j-1), 1/l*j
		ll = max(l1, l2)
		r = min(r1, r2)
		x[i,j] = ll<r ? r-ll :
			(l1<r2 ? Inf : -Inf)
	end
	return x
end


### Main routine for overlap computation

sbv_linear_nocache(args...) = sbv_linear(args...; cachetype=NoCache)

""" linear scheme to the recursive operation, traversing all j and matching to an i on the coarsest level """
function sbv_linear(b1, b2, k, l, cachetype=IndexCache)
	@assert size(b1, 1) == size(b2, 1)

    dists = calc_dists(k, l)
    p1 = sortperm(collect(eachcol(b1)))
    p2 = sortperm(collect(eachcol(b2)))

	b1 = b1[:, p1]
	bb = b2[:, p2]  # we keep this for col major access in gonext
	b2 = collect(bb')' # use lazy transpose to get faster row-major vectors in gonext
	# bb = b2  # in memory sensitive situations we can discard the col major 


	d, n = size(b1)
	c = cachetype(d)

	I = Int[]
	J = Int[]
	V = Float64[]

	for i = 1:n
		j = start(c, b1[:, i])
		firstmatch = search!(i, b1, b2, dists, I, J, V, j, bb)
		update!(c, b1[:,i], firstmatch)
	end

	V = sparse(I,J,V,size(b1, 2), size(b2, 2))

	return V[invperm(p1), invperm(p2)]
end


function search!(i, b1, b2, overlap, I, J, V, j, bb)
	D, M = size(b2)
	O = zeros(D)

	firstmatch = zeros(Int, D)
	lm = 0  # depth of match

	l = 1  # current dimesion/level

	while j <= M
		o = overlap[b1[l,i], b2[l,j]]
		if o == -Inf
			l,j = gonext(l, j, b2, bb)
		elseif o == Inf
			if lm < l  # only for cache updates
				lm = l
				firstmatch[l:end] .= j
			end
			l = l - 1
			if l == 0
				break
			end
			l, j = gonext(l, j, b2, bb)
		else
			if lm < l  # only for cache updates
				lm = l
				firstmatch[l:end] .= j
			end
			O[l] = o
			if l == D
				push!(I, i)
				push!(J, j)
				push!(V, prod(O))
				l, j = gonext(l, j, b2, bb)
			else
				l = l + 1
			end
		end
	end
	return firstmatch
end


# go from position j to the next with a different entry in dimensions[1:l]
function gonext(l, j, b, bb)
	n = size(b, 2)
	@inbounds @views for i in j+1:n
		if b[l,i] != b[l,j]  # search difference to the right
			if bb[1:l-1, i] == bb[1:l-1, j]  # check that everything above is the same
				return l, i
			else
				return gonext(l-1,j, b, bb)  # otherwise search difference a level higher
			end
		end
	end
	return l, n+1
end


### Caching for faster traversal, remembers the starting j for increasing i

struct IndexCache
	box::Vector{Int}
	ind::Vector{Int}
end

IndexCache(d) = IndexCache(ones(d), ones(d))

function start(c::IndexCache, b)
	i = findfirst(c.box .!= b)
	j = isnothing(i) ? c.ind[end] : c.ind[i]
	return j
end

function update!(c::IndexCache, b, firstmatch)
	i = findfirst(c.box .!= b)
	if !isnothing(i)
		c.box[i:end] = b[i:end]
		c.ind[i:end] = firstmatch[i:end]
	end
end

struct NoCache end

NoCache(d) = NoCache()
start(::NoCache, _) = 1
update!(::NoCache, _, _) = nothing




## Tests

sbv_old(b1,b2,k,l) = sparse(spdistances(k,l,b1,b2)..., size(b1,2), size(b2,2))
sbv_methods = [sbv_linear, sbv_linear_nocache, sbv_old]

using Random

subsample(n) = shuffle(1:n)[1:rand(1:n)]
randboxes(n, k, d) = reduce(hcat, unique(eachcol(rand(1:k, d, n))))

function test_23(method=sbv_methods[1])
	k = 2
	l = 3
	b1 = [1 1 2 2; 1 2 1 2]
	b2 = [1 1 1 2 2 2 3 3 3; 1 2 3 1 2 3 1 2 3]

	truth = [4 2 0 2 1 0 0 0 0
	         0 2 4 0 1 2 0 0 0
			 0 0 0 2 1 0 4 2 0
			 0 0 0 0 1 2 0 2 4] / 36

	v = method(b1, b2, k, l)

	@assert isapprox(v, truth)

	for i=1:100
		i1 = subsample(size(b1,1))
		i2 = subsample(size(b2,1))
		v = method(b1[:,i1], b2[:,i2], k, l)
		@assert isapprox(v, truth[i1,i2])
	end
end

function test_compare(n=100,m=100,k=10,l=5,d=4; method=sbv_linear, ref=sbv_old)
	b1 = randboxes(n, k, d)
	b2 = randboxes(m, l, d)

	v1 = method(b1,b2,k,l)
	v2 = ref(b1,b2,k,l)

	@assert isapprox(v1,v2)
end

benchmarkdata() = test_data(10000,10000,8,8,6)

function test_data(n=100,m=100,k=10,l=5,d=4)
	b1 = randboxes(n, k, d)
	b2 = randboxes(m, l, d)
	b1, b2, k, l
end

function test_sbv()
	for method in sbv_methods
		test_23(method)
	end
	for i=1:100
		test_compare(rand(10:1000), rand(10:1000), rand(1:15), rand(1:15), rand(1:8))
	end
end


# returns the common volume of two sparse discretizations of a unit volume
# compare all cells withanother, compute the boundaries and resp. distances in each dimension and take the product
function spdistances(res1, res2, cart1, cart2)
	w1 = 1/res1 / 2 # half width of the box
	w2 = 1/res2 / 2
	c1 = cart1 / res1 .- w1
	c2 = cart2 / res2 .- w2
	dims, n = size(cart1)
	dims, m = size(cart2)
	I = Int[]
	J = Int[]
	V = Float64[]
	for i in 1:n, j in 1:m

		vol = 1.
		for d in 1:dims
			left  = max(c1[d, i] - w1, c2[d, j] - w2)
			right = min(c1[d, i] + w1, c2[d, j] + w2)
			s = right - left
			if s > 0
				vol = vol * s
			else
				vol = 0.
				#break
			end
		end

		if vol > 0
			push!(I, i)
			push!(J, j)
			push!(V, vol)
		end
	end
	#return sparse(I, J, V, n, m)
	return I, J, V
end


# history
# v1 - the old bruteforce (correct)
# v2 - wrong with frank
# v3 - recursive
# v4 - linear (with 3 subsequent cache versions) (now correct)

#= wrong recursive version

""" recursive overlap calculation """
function sbv_rec(b1, b2, k, l)
    dists = calc_dists(k, l)
    p1 = sortperm(collect(eachcol(b1)))
    p2 = sortperm(collect(eachcol(b2)))

 	V = spzeros(size(b1, 2), size(b2, 2))

	b1 = b1[:, p1]
	b2 = b2[:, p2]

	D, n = size(b1)
	D, m = size(b2)

	for i = 1:n
		deepen(i, b1, b2, V, dists)
	end

	return V[invperm(p1), invperm(p2)]
end


""" initialization to the recursion """
function deepen(i, b1, b2, V, dists)
	D, m = size(b2)
	deepen(1, i, 1, m, b1, b2, D, 1, V, dists)
end

""" check if compare b2[d,jstart:jend] has overlap with b1[d, i].
if so call itself recursively on the interval of same d-th coordinate and d=d+1,
finally (on match on lowest level) update V with the volume """
function deepen(d, i, jstart, jend, b1, b2, maxd, v, V, dists)
	#@show d, jstart, jend
	di = b1[d, i]
	K = size(dists, 2)

	for x in 1:K
		l = dists[di, x]
		if l == -Inf
			continue
		elseif l == Inf
			break
		else
			#@show d,x,jstart, jend
			found, jstart, jend = findint(b2, d, x, jstart, jend)
			if found
				if d < maxd
					deepen(d+1, i, jstart, jend, b1, b2, maxd, v*l, V, dists)
				else
					@assert jstart==jend
					V[i, jstart:jend] .= v
				end
			end
		end
	end
end

""" find s/e such that b[d,s,e] == x searching only between jstart:jend"""
function findint(b, d, x, jstart, jend)
	found = false
	s = jstart
	e = jend

	for j in jstart:jend
		v = b[d,j]
		if v == x
			if !found
				s = j
				found = true
			end
		elseif v > x
			e = j - 1
			break
		end
	end

	return found, s, e
end
=#
