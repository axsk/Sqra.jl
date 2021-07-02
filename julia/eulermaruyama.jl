using ProgressMeter

function eulermarujamatrajectories(x0::Matrix, potential::Function, sigma::Real, dt::Real, steps::Integer; branches::Integer=1, maxdelta=Inf, progressbar=true)
    dim, samples = size(x0)
    xts = similar(x0, dim, steps+1, branches, samples)
    p = Progress(samples*branches*steps; dt=1, desc="Euler Maruyama simulation", enabled=progressbar)
    for s in 1:samples
        for b in 1:branches
            x = x0[:, s]
            xts[:, 1, b, s] = x
            for t in 2:steps+1
                g = Zygote.gradient(potential, x)[1]
                if sum(abs2, g * dt) > maxdelta
                    x = eulermarujamatrajectories(reshape(x, length(x), 1), potential, sigma, dt/10, 10, maxdelta=maxdelta, progressbar=false)[:, end, 1, 1]
                else
                    x .+= -g * dt .+ sigma * randn(dim) * sqrt(dt)
                end
                xts[:, t, b, s] = x
		next!(p)
            end
        end
    end
    xts
end
