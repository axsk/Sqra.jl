# Another attempt at writing a trie,
# see trie.jl

struct Tree
    depth
    width
    root
end

struct Node
    children
end

struct Leaf
    inds
end

function Node(t::Tree, d::Int)
    if d < t.depth
        @show Node(Vector{Any}(undef, 6))
    else
        @show Node(Int[])
    end
end


function get(t::Tree, n::Node, i::Int, d::Int)
    if !isassigned(n.children, i)
        m = Node(t, d+1)
        n.children[i] = m
        return m
    else
        return n.children[i]
    end
end

function push!(n::Node, i)
    Base.push!(n.children, i)
end

function boxify(x::Vector{Vector{Int}})
    n = Node()
    for i in 1:length(x)
        addnode(x, i)
    end
end

function addnode(tree, x, i)
    n = tree.root
    dims = tree.depth
    for d in 1:dims - 1
        n = get(tree, n, x[d], d)
    end
    push!(n, i)
end
