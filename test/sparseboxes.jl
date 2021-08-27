
using Profile, Random, BenchmarkTools, Test
using Sqra: sparseboxpick, sparseboxpick_old

function benchbox(n=100000, k=10, d=6)
	Random.seed!(1)
	points = rand(d, n)
	boundary = autoboundary(points)
	boxify(points, k, boundary)
	Profile.init(10000000, 0.0001)
	#Profile.clear
	@btime boxify($points, $k, $boundary)
	@profile for i=1:10
		boxify(points, k, boundary)
	end
	Profile.print(mincount=10, maxdepth=14)
end


function benchboth(n=10000, k=10, d=6)
	points = rand(d, n)
	boundary = autoboundary(points)
	u = rand(n)

	begin
		A, i = sparseboxpick_old(points, k, u, boundary)
		p = sortperm(collect(eachcol(points)))
		a = points[:,p]
	end

	begin
		A, i = sparseboxpick(points, k, u, boundary)
	end
end

# compase old and new function
function test_sparseboxpick(n=10000, k=10, d=6)
	p = rand(d, n)
	u = rand(n)
	A, i = sparseboxpick(p, k, u)
	B, j = sparseboxpick_old(p, k, u)

	p1 = sortperm(i)
	p2 = sortperm(j)

	@assert i[p1] == j[p2]
	@assert A[p1,p1] == B[p2,p2]

	return  A, B, i, j
end

function test_spboxes()
    x = [0 0.5 0
	 0 0   1]
	cartesians, A = spboxes(x, 2)
	@assert all(A .== @show (pairwise(Cityblock(), cartesians) .== 1))
end

function test_merge(n=5, d=2, l=2)
	x = rand(d,2*n)
	bnd = [0 1; 0 1]
	b1 = SparseBoxes(x[:,1:n], l, bnd)
	b2 = SparseBoxes(x[:,(1:n) .+ n], l, bnd)
	b  = SparseBoxes(x, l, bnd)

	bm = Sqra.merge(b1,b2, n)
	@test b.boxes == bm.boxes
	@test b.inds  == bm.inds
end

test_merge()

@test (test_sparseboxpick(); true)
