using Distances

function picking(X, n)
	@assert size(X, 2) >= n
	d = zeros(size(X, 2), n)

	qs = [1]
	pairwise!(@view(d[:, 1:1]), SqEuclidean(), X, X[:,1:1])
	mins = d[:, 1]

	@views @showprogress 1 "Picking " for i in 1:n-1
		mins .= min.(mins, d[:, i])
		q = argmax(mins)
		pairwise!((d[:, i+1:i+1]), SqEuclidean(), X, X[:,q:q])
		push!(qs, q)
	end

	return X[:, qs], qs, sqrt.(d)
end
