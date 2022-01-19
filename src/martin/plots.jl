function scatter(xs::Vector{S}; kwargs...) where S <: SVector
    Plots.scatter(collect(eachrow(reduce(hcat, xs)[1:2,:]))...; kwargs...)
end

methodlabels = Dict(:legacy => "normal")
methodmarker = Dict(:uniform => :rect, :normal => :x, :grad => :dtriangle, :hess => :utriangle, :legacy => :circle)

function plot_h_i(b; kwargs...)
    ih = plot()
    #plot!(ih, legend=false)
    xaxis!(L"B", :log)
    yaxis!(L"H",:log)
    for e in b
        marker = get(methodmarker, e.method, :x)
        nin = length(e.inner)
        scatter!(ih, [e.I], [e.H], marker_z=log10(nin), marker=marker,
        group = [e.method], label=nothing; kwargs...)
    end
    for e in b[1,:,end]
        marker = get(methodmarker, e.method, :x)
        nin = length(e.inner)
        lab = get(methodlabels, e.method, String(e.method))
        scatter!(ih, [e.I], [e.H], marker_z=log10(nin), marker=marker,
        group = [e.method], label=lab; kwargs...)
    end
    mn, mx = extrema(filter(!isnan,[e.I for e in b]))
    yn, yx = [1, mx/mn] .* minimum(filter(!isnan, [e.H for e in b]))
    plot!(ih, [mn, mx], [yn, yx], label=nothing, color=:navy)
    plot!(legend_position=:bottomright)
    yticks!([yn, yx])
    xticks!([mn, mx])
    ih
end

function plot_n_l(b; kwargs...)
    b = vec(b)
    N = map(x->length(x.inner) ^ -(1/4), b)
    L = map(x->sqrt(sum(x.L[x.inner])), b)
    yt= vcat(extrema(filter(isfinite,L))...)

    mn, mx = extrema(filter(!isnan, N))
    yn, yx = [1, mx/mn] .* minimum(filter(!isnan, L))


    method = (x->x.method).(b)

    marker = (x->methodmarker[x]).(method)
    replace!(method, :legacy=>:normal)

    Plots.scatter(N, L, xticks=vcat(extrema(N)...), yticks=yt, xaxis=:log, yaxis=:log, group = method, marker=marker; kwargs...)
    Plots.plot!([mn, mx], [yn, yx], label=nothing, color=:navy)
    plot!(legend_position=:bottomright)
    Plots.xaxis!(L"N^{-\frac{1}{4}}")
    Plots.yaxis!(L"L_2")
end

function plot_u(e::NamedTuple; kwargs...)
    lim = 1.5
    scatter(e.xs, marker_z=e.us, xlims=[-lim,lim], ylims=(-lim,lim); kwargs...)
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

function samplingplot(n=300; seed=1, kwargs...)
    for method in [:uniform, :legacy, :grad, :hess]
        e = experiment(n, 2, method, seed)
        label = get(methodlabels, method, String(method))
        p = plot_u(e, title=label, legend=false; kwargs...) |> display

        savefig("sampling$(label).png")
    end
end


function paperplots(b=qbatch(D=4, seeds=1:2, ns=[128000, 64000,32000,16000,8000,4000,2000,1000], methods=[:legacy, :uniform, :normal, :grad, :hess]))
    plot_h_i(b[1:1,[1,2,4,5],1:7]);
    xticks!([10^.98, 10^1.95]);
    yticks!([10^-.7, 10^-1.67]);
    plot!() |> display
    savefig("ih.pdf")

    plot_n_l(b[1:1,[1,2,4,5],1:7])
    xticks!([10^-1.25, 10^-.65])
    yticks!([10^-1, 10^-1.8])
    plot!() |> display
    savefig("nl.pdf")
end
