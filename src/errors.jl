function voronoi_sb_l2(x, sb::SparseBoxesDict, u1, u2)
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

function error_voronoi_voronoi(x, y, u1, u2)
	tree = KDTree(y)
	idxs, _ = nn(tree, x)
	err = sum(abs2, u1 .- u2[idxs])
	sqrt(err) / size(x, 2)
end

function error_voronoi_symmetric(x, y, u1, u2)
	e1 = error_voronoi_voronoi(x, y, u1, u2)
	e2 = error_voronoi_voronoi(y, x, u2, u1)
	return (e1+e2)/2
end
