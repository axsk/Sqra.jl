using Flux
import Zygote
import Zygote.@showgrad

function mwe1(n=1)
    a = ones(1) * 0
    b = ones(1) * .5
    data = ones(1,n)

    function loss(x)
        y, pb = Zygote.pullback(x) do x
            @showgrad aa = sum(x->(x^2), x .- a, dims=1)
            @showgrad bb = sum(x->(x^2), x .- b, dims=1)
            @showgrad(aa = aa .+ 0)
            @showgrad(bb = aa .+ 0)
            @showgrad r = ones(1,n) - @showgrad(aa)
            @showgrad s = r - bb
        end
        dfdx,  = pb(ones(1,n))
        dfdx
    end

    @show l1 = sum(loss(data))

    #z = rand(2)

    l2, pb = Zygote.pullback(rand(3)) do z
        @show sum(loss(data)) # inside do block
    end
    @show l2

    @assert l1 == l2
end

#mwe1()

function mwe()
    a = 0
    b = .5
    data = 1.0

    function loss()
        y, pb = Zygote.pullback(data) do x
            @showgrad aa = abs2(x - a)
            @showgrad bb = abs2(x - b)
            @showgrad r = 1 - aa - bb
        end
        @showgrad dfdx,  = pb(data)
        dfdx
    end

    l1 = sum(loss())
    @show l1

    l2, pb = Zygote.pullback() do
        @showgrad sum(loss())
    end

    @show l1, l2
    @assert l1 == l2
end

function losses(c::CommittorVariational, f, x::Matrix)
    #f = boundaryfixture(c.boundary, f)
    y, pb = Flux.pullback(x) do x
        bnd = c.boundary
        #a = exp.(bnd.e * sum(abs2, x .- bnd.a, dims=1)) # set a with bnd = 1
        #b = exp.(bnd.e * sum(abs2, x .- bnd.b, dims=1)) # set b with bnd = 0
        #a = exp.(bnd.e * sum(abs2, x .- bnd.a, dims=1))
        @show bnd.a
        a = sum(abs2, x .- bnd.a, dims=1)
        b = sum(abs2, x .- bnd.b, dims=1)
        #a = .1
        r = 1 .- (a .+ b)# + b)
        1 .* r .+ a
        @show size(r)
        r

    end
    #@show sum(y)
    dx, = pb(ones(1,size(x,2)))
    @show sum(dx)
    l2 = sum(abs2, dx, dims=1)
    #@show sum(l2)
    l2
    #r = c.reweight(x)
    #@show size(l2), size(r)
    #l2 .* r'
end

using ForwardDiff

function losses_f(c::CommittorVariational, f, x::Matrix)
    f = boundaryfixture(c.boundary, f)
    df = map(eachcol(x)) do x
        ForwardDiff.gradient(x->f(x)[1], x) end
    df = hcat(df...)
    l2 = sum(abs2, df, dims=1)
    r = c.reweight(x)
    l2 .* r'
end