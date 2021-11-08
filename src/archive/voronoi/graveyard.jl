## experimental code

function orthonorm(x)
	v = deepcopy(x)
	k = length(x)
	for i in 1:k
		v[i] = x[i]# .- x[k]
		for j in 1:(i-1)
			v[i] = v[i] .- dot(v[i], normalize(v[j])) .* normalize(v[j])# / norm(v[j])^2
		end
		#v[i] = normalize(v[i])
	end
	v
end

function uniquefaces(s)
	sigs = []
	rs = []
	for x in s
		sig, r = x
		if !(sig in sigs)
			push!(sigs, sig)
			push!(rs, r)
		end
	end
	return sigs, rs
end
