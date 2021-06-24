using Distances

function picking(X, n)
	d = zeros(size(X, 2), n)

	qs = [1]
	pairwise!(@view(d[:, [1]]), SqEuclidean(), X, X[:,[1]])
	mins = d[:, 1]

	@views for i in 1:n-1
		mins .= min.(mins, d[:, i])
		q = argmax(mins)
		pairwise!(d[:, [i+1]], SqEuclidean(), X, X[:,[q]])
		push!(qs, q)
	end

	return X[:, qs], qs, d
end
