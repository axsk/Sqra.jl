using Base: Float64

using SparseArrays

# mean squared error of two scalar functions (not densities!) on two respective
# sparse box distributions integrated over their common support
function sp_mse(x1, x2, cart1, cart2, res1, res2)
	I,J,V = spdistances(res1, res2, cart1, cart2)
	e = 0
	for i in 1:length(I)
		e += (x1[I[i]] - x2[J[i]])^2 * V[i]
	end
	e
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

function test_spdistances()
	@assert spdistances(2,2,[1 2; ], [1 2; ]) == [.5 0; 0 .5]
end
#=
function spdistances_vec(res1, res2, cart1, cart2)
	c1 = cart1 / res1
	c2 = cart2 / res2
	w1 = 1/res1 / 2 # half width of the box
	w2 = 1/res2 / 2
	left = max.(c1 .- w1, c2 .- w2)
	right = min.(c1 .+ w1, c2 .+ w2)
	d = max.(right-left, 0)
	prod(d, dims=1)
end
=#

using SparseArrays

sbv_old(b1,b2,k,l) = sparse(spdistances(k,l,b1,b2)..., size(b1,2), size(b2,2))

function test_spb_volume(;d=3, n=10, m=10, k=3, l=3, method=spb_volume)
	b1 = unique(rand(1:k, d, n), dims=2)
	b2 = unique(rand(1:l, d, m), dims=2)

	method(b1, b2, k, l)
end


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

function calc_dists(k, l)
	x = zeros(k,l)
	for i in 1:k
		for j in 1:l
			if i == j
				x[i,j] = 1/k
			elseif i < j
				x[i,j] = Inf
			elseif i > j
				x[i,j] = -Inf
			end
		end
	end
	x
end

function deepen(i, b1, b2, V, dists)
	D, m = size(b2)
	deepen(1, i, 1, m, b1, b2, D, 1, V, dists)
end

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



function sbv_linear(b1, b2, k, l)
    dists = calc_dists(k, l)
    p1 = sortperm(collect(eachcol(b1)))
    p2 = sortperm(collect(eachcol(b2)))

 	V = spzeros(size(b1, 2), size(b2, 2))

	b1 = b1[:, p1]
	b2 = b2[:, p2]

	n = size(b1, 2)
	for i = 1:n
		search!(i, b1, b2, dists, V)
	end

	return V[invperm(p1), invperm(p2)]
end


function search!(i, b1, b2, overlap, V)
	D, M = size(b2)
	O = zeros(D)

	l = 1  # current dimesion/level
	j = 1

	while j <= M
		o = overlap[b1[l,i], b2[l,j]]
		if o == -Inf
			l,j = gonext(l, j, b2)
		elseif o == Inf
			l = l - 1
			if l == 0
				break
			end
			l, j = gonext(l, j, b2)
		else
			O[l] = o
			if l == D
				V[i,j] = prod(O)
				l, j = gonext(l, j, b2)
			else
				l = l + 1
			end
		end
	end
end

function updatecache()
end

# go from position [l,j] to the next
function gonext(l, j, b)
	n = size(b,2)

	found = false

	for i in j+1:n, d in 1:l
		if b[d, i] != b[d,j]
			l = d
			j = i
			found = true
			break
		end
	end

	if found == false
		j = n + 1
	end

	return l, j
end
