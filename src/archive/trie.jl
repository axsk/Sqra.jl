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

    mapreduce(mapper, reducer,  iterable; init= (Vector{Int}[],Vector{Int}[])) 
end

function performance(n=10_000_000,d=6,l=5)
    x = rand(d, n)
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

for (k,v) in node