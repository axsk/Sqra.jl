"methods to compare the approximations of the committor
between different discretizations"


"given voronoi generators v_i and sparseboxes x_j compute
the l2 difference between u1(v_i) and u2(v_j) summed over v_i
where x_j is the representative of the box v_i lies in"
function error_voronoi_sb(x, sb::SparseBoxesDict, u1, u2)
	c = cartesiancoords(x, sb.level)
	k = keys(sb.dict) |> collect
	err = 0.
	for (i, coord) in enumerate(eachcol(c))
		j = findfirst(x->x==coord, k)
		err += sum(abs2, u1[i] - u2[j])
	end
	sqrt(err) / size(c, 2)
end

using NearestNeighbors

"given two voronoi tesselations, compute the nonsymmetric l2 difference between
u1(v1_i) and u2(v2_j) over v1_i, where v2_j is the closest point to v1_i"
function error_voronoi_voronoi(x, y, u1, u2)
	tree = KDTree(y)
	idxs, _ = nn(tree, x)
	err = sum(abs2, u1 .- u2[idxs])
	sqrt(err) / size(x, 2)
end

"symmetric average error of two voronoi tesselations"
function error_voronoi_symmetric(x, y, u1, u2)
	e1 = error_voronoi_voronoi(x, y, u1, u2)
	e2 = error_voronoi_voronoi(y, x, u2, u1)
	return (e1+e2)/2
end


struct isVoronoi end
struct isSparseBoxes end

function classify_experiment(e)
	if haskey(e, :sb)
		return isSparseBoxes()
	elseif haskey(e, :npick)
		return isVoronoi()
	else
		Base.error()
	end
end


# first argument is considered to be the truth

# approx
error(_::isVoronoi, _::isVoronoi, e1, e2) =
	error_voronoi_voronoi(e1.x, e2.x, e1.cmt, e2.cmt)

# exact
error(_::isSparseBoxes, _::isSparseBoxes, e1, e2) =
	sp_mse(e1.cmt, e2.cmt, e1.sb, e2.sb)

# approx
error(::isVoronoi, _::isSparseBoxes, e1, e2) =
	error_voronoi_sb(e1.x, e2.sb, e1.cmt, e2.cmt)

# approx
error(::isSparseBoxes, ::isVoronoi, e1, e2) =
	error_voronoi_voronoi(e1.x[:,e1.picks], e2.x, e1.cmt, e2.cmt)


function error(e1,e2)
	t1 = classify_experiment(e1)
	t2 = classify_experiment(e2)
	error(t1,t2,e1,e2)
end
