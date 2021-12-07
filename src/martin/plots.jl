
function plot_h_i(b)
    ih = plot()
    plot!(ih, legend=false)
    xaxis!(L"I_2", :log)
    yaxis!(L"H_{T,\kappa}",:log)
    for e in b
        scatter!(ih, [e.I], [e.H], marker_z=log10(e.N))
    end
    mn, mx = extrema(filter(!isnan,[e.I for e in b]))
    yn, yx = [1, mx/mn] .* minimum(e.H for e in b)
    plot!(ih, [mn, mx], [yn, yx])
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
    st = plot()
    xaxis!(L"n^{1/4}")
    yaxis!(L"H_{\mathcal{T},\kappa}")
    yaxis!(:log)
    xaxis!(:log)
    plot!(legend=false)
    for e in b
        scatter!(st, [e.N ^ (1/4)], [e.H])
    end
    st
end

function plot_u(e::NamedTuple)
    scatter(e.xs, marker_z=e.vs, xlims=[-1,1], ylims=(-1,1))
end

function plot_uv(e::NamedTuple)
    scatter(e.xs, marker_z=e.us-e.vs, xlims=[-1,1], ylims=(-1,1))
end

function plotbatch(b)
    hi = plot_h_i(b)
    nh = plot_n_h(b)

    l = @layout[a; b]
    plot(hi, nh, layout = l, size=(800,1000))
end
