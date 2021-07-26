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
