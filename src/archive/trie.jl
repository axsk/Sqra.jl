struct Node
    val::Union{Dict{Int, Node}, Vector{Int}}

    function Node(val)
        new(val)
    end
end

NNode(isleaf::Bool) = isleaf ? Node(Int[]) : Node(Dict{Int, Node}())

function Base.get!(n::Node, i::Int, isleaf::Bool)
    newnode() = NNode(isleaf)
    get!(newnode, n.val, i)
end

function Base.push!(n::Node, v::Int)
    push!(n.val, v)
end

function trie(x::Matrix{Float64}, l)
    dims, N = size(x)
    trie = NNode(false)
    for i in 1:N
        n = trie
        for d in 1:dims
            v = x[d,i]
            (0 <= v <= 1) || break
            a = ceil(Int, v * l)
            n = get!(n, a, d == dims)
            if d == dims
                push!(n, i)
            end
        end
    end
    return trie
end

function test(n=100,d=6,l=5)
    x = rand(d, n)
    x, trie(x, l)
end

isleaf(t::Node) = isa(t.val, Vector)


function traverse(n::Node, prefix=Int[])
    prefix, n
    if isleaf(n)
        return [prefix], [n.val]
    end

    ccat = ((p1,v1), (p2,v2)) -> ([p1;p2], [v1;v2])
    reducer(a,b) = vcat.(a,b)

    mapper((k,v)) = traverse(v, [prefix; k])
    iterable = [(k,v) for (k,v) in n.val]

    #@show iterable

    mapreduce(mapper, reducer,  iterable)#; init= (Vector{Int}[],Vector{Int}[]))
end

function traverse2(t, D, n)
	x = Vector{Int}[]
	y = Vector{Int}[]
	coords = zeros(Int, D)

	i=0

	nodeiter(n) = Iterators.Stateful(n.val)
	iters=[nodeiter(t)]

	while !isempty(iters)
		d=length(iters)
		iter=iters[end]
		if isempty(iter)
			pop!(iters)
		else
			k, n = popfirst!(iter)
			coords[d] = k
			if d < D
				push!(iters, nodeiter(n))
			else
				push!(x, copy(coords))
				push!(y, n.val)
			end
		end
	end
	x, y
end




function performance(n=10_000_000,d=6,l=5; x=rand(d,n ))
    @time begin
        t = trie(x, l)
        traverse(t)
    end

    x .= round.(x.*l)

    @time begin
        sortperm(collect(eachcol(x)))
    end

    ()
end


using Sqra
function performance2(n=1_000_000)
	s=run(Simulation(nsteps=n))
	x=s.x
	@profile begin

	@time begin
		b=SparseBoxes(s.x, 6)
		A = Sqra.adjacency(b)
	end

	@time begin
		x = Sqra.cartesiancoords(s.x, 6)
		d = dict(x)
		B = neighbours(d,6)
	end
	A,B

	end
end
