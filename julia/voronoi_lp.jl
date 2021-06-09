using JuMP
#import Tulip
import Clp


function voronoi_neighbourhood(g, tolerance=1e-6)
    n, d = size(g)
    b = vec(sum(abs2, g, dims=2))
    A = hcat(2*g, -ones(n))
    adjacency = zeros(Bool, n,n)
    v = zeros(n,n)

    for i in 1:n
        e_i = zeros(n)
        e_i[i] = 1
        for j in i+1:n
            #m = Model(Tulip.Optimizer)
            m = Model(Clp.Optimizer)
            set_optimizer_attribute(m, "LogLevel", 0)
            #set_optimizer_attribute(m, "Algorithm", 4)
            @variable(m, x[1:(d+1)])
            @objective(m, Min, b[i] - A[i,:]' * x)
            @constraint(m, b+e_i - A*x .>= 0)
            @constraint(m, b[j] - A[j,:]' * x == 0)
            optimize!(m)
            v[i,j] = objective_value(m)

            #@show termination_status(m), v
            #if v < -.5
            #    adjacency[i,j] = 1
            #end
        end
    end
    v = v + v'
    adj = v .< -tolerance
    return adj, v
end


function testruns(
    ns = [10, 50, 100, 200, 300, 400, 500],
    ds = [2,3,6])
    neig = zeros(length(ns), length(ds))
    time = copy(neig)
    try
        for (i,n) in enumerate(ns)
            for (j,d) in enumerate(ds)
                @show n, d
                g = rand(n, d)
                time[i,j] = @elapsed A = @time voronoi_neighbourhood(g)[1]
                @show neig[i,j] = sum(A) / n
            end
        end
    catch
    end
    neig, time
end