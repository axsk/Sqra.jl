function scatter(xs::Vector{S}; kwargs...) where S <: SVector
    Plots.scatter(collect(eachrow(reduce(hcat, xs)[1:2,:]))...; kwargs...)
end

methodmarker = Dict(:uniform => :rect, :normal => :circle, :grad => :dtriangle, :hess => :utriangle)

function plot_h_i(b; kwargs...)
    ih = plot()
    #plot!(ih, legend=false)
    xaxis!(L"I_2", :log)
    yaxis!(L"H_{T,\kappa}",:log)
    for e in b
        scatter!(ih, [e.I], [e.H], marker_z=log10(e.N), marker=methodmarker[e.method],
        group = [e.method], label=nothing; kwargs...)
    end
    for e in b[1,:,end]
        scatter!(ih, [e.I], [e.H], marker_z=log10(e.N), marker=methodmarker[e.method],
        group = [e.method], label=String(e.method); kwargs...)
    end
    mn, mx = extrema(filter(!isnan,[e.I for e in b]))
    yn, yx = [1, mx/mn] .* minimum(filter(!isnan, [e.H for e in b]))
    plot!(ih, [mn, mx], [yn, yx], label=nothing)
    yticks!([yn, yx])
    xticks!([mn, mx])
    ih
end

function plot_n_h(b)
    st = plot()
    xaxis!(L"\# K")
    yaxis!(L"\Vert u_\mathcal{T}-\mathcal{R}_{\mathcal{T}}u\Vert_{H_{\mathcal{T},\kappa}}")
    yaxis!(:log)
    xaxis!(:log)
    plot!(legend=false)

    for e in b
        scatter!(st, [e.N], [e.H])
    end
    st
end

function plot_sqn_h(b)
    d = first(b).D
    st = plot()
    xaxis!(L"N^{-\frac{1}{d}}")
    yaxis!(L"H_{\mathcal{T},\kappa}")
    yaxis!(:log)
    xaxis!(:log)
    plot!(legend=false)
    n = [e.N ^ (-1/d) for e in b] |> vec
    h = [e.H for e in b] |> vec
    marker = [methodmarker[e.method] for e in b] |> vec
    scatter!(st, n, h, marker=marker)

    mn, mx = extrema(filter(!isnan,n))
    yn, yx = [1, mx/mn] .* minimum(filter(!isnan, h))
    plot!(st, [mn, mx], [yn, yx], label=nothing)
    yticks!([yn, yx])
    xticks!([mn, mx])
    #for e in b
    #    scatter!(st, [e.N ^ (-1/d)], [e.H])
    #end
    st
end

function plot_n_l(b; kwargs...)
    b = vec(b)
    N = map(x->x.N ^ -(1/4), b)
    L = map(x->sqrt(sum(x.L[x.inner])), b)
    yt= vcat(extrema(filter(isfinite,L))...)
    method = (x->x.method).(b)
    #@show length(L)
    Plots.scatter(N, L, xticks=vcat(extrema(N)...), yticks=yt, xaxis=:log, yaxis=:log, group = method; kwargs...)
    Plots.xaxis!(L"N^{-\frac{1}{4}}")
    Plots.yaxis!(L"L_2")
end

function plot_u(e::NamedTuple; kwargs...)
    scatter(e.xs, marker_z=e.us, xlims=[-1,1], ylims=(-1,1); kwargs...)
end

function plot_uv(e::NamedTuple; kwargs...)
    scatter(e.xs, marker_z=e.us-e.vs, xlims=[-1,1], ylims=(-1,1); kwargs...)
end

function plotbatch(b)
    hi = plot_h_i(b)
    nh = plot_n_h(b)

    l = @layout[a; b]
    plot(hi, nh, layout = l, size=(800,1000))
end

@userplot SqnH

@recipe function f(es::SqnH)
    #@show es
    es,  = es.args
    @show typeof(es)
    data = reduce(hcat, [e.N, e.H] for e in es)
    N = data[1,:]
    H = data[2,:]
    seriestype := scatter

    #d = first(b).D
    #st = plot()
    #xaxis!(L"n^{-1/d}")
    #yaxis!(L"H_{\mathcal{T},\kappa}")
    #yaxis!(:log)
    #xaxis!(:log)
    #plot!(legend=false)
    #for e in b
    #    scatter!(st, [e.N ^ (-1/d)], [e.H])
    #end
    N, H
end

function samplingplot(n=300; kwargs...)
    for method in [:uniform, :normal, :grad, :hess]
        e = experiment(n, 2, method, 1)
        p = plot_u(e, title=String(method); kwargs...) |> display
        savefig("sampling$(String(method)).png")
    end
end
