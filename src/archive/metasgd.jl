# my own take on learning the learning rate

using Zygote

loss(t,d) = sum((t .- d).^2) / length(d)
dloss(t,d) = Zygote.gradient(t->loss(t,d),t)[1]

using Plots
function adaptalpha(ls, alphas, lambda, beta)
    lambda == 0 && return alphas[end]
    length(alphas)<3 && return alphas[end]/1.1



    #dd = (ls[end] - ls[end-2]) / (alphas[end] - alphas[end-1])
    #@show da =  da * beta + dd * (1-beta)

    #a,b = linearregression(alphas, diff(ls), beta)
    #da = a
    #alpha = alphas[end] - da * lambda

    last(x) = x[end-min(length(x)-1, 100):end]

    alpha = quadraticmin(last(alphas), last(diff(ls)))

    min(max(alpha, alphas[end]/1.1), alphas[end]*1.1) * (1 + rand() * 0.1 - rand() * 0.1)
end

function learn(x0=5., n=20, a0=0.01; lambda=0, data=randn(100), batch=1, beta=0.1)

    xs = [x0]
    as = [a0]
    ls = [loss(x0, rand(data, batch))]
    da = 0

    local x

    for i in 1:n
        d = rand(data, batch)
        x = xs[end]
        a = as[end]

        x -= a * dloss(x, d)
        l = loss(x,d)

        push!(xs, x)
        push!(ls, l)

        a = adaptalpha(log.(ls), as, lambda, beta)

        push!(as, a)
    end

    loss(x, data) - loss(0, data), xs, ls, as
end

learn()

function linearregression(x,y,decay=0)
    xx = ones(2, length(x))
    xx[1,:] = x

    d = exp.(-collect(length(x):-1:1) * decay)

    scatter(x,y, markersize=d*10)
    a,b = (xx' .* d) \ (y .* d)
    plot!(x->x*a + b) |> display
    sleep(0.1)
    a,b
end

using Polynomials

function quadraticmin(x,y)
    @show size(x), size(y)
    p = fit(x, y, 2)
    scatter(x,y)
    plot!(p) |> display
    roots(derivative(p))[1]
end


# first attempt at metalearning from `age` previous alphas and losses

function metalearn(losses, alphas, age, opt=Descent(1))
    n = min(age, length(losses), length(alphas))
    if n == 0
        return 1/2
    elseif  n == 1
        return 1.
    else
        learnrate_fd = losses[end-n+1:end] - losses[end-n:end-1]
        alphas = alphas[end-n+1:end]
        X = ones(n, 2)
        X[:, 1] = alphas
        coeff = X \ learnrate_fd
        stochgrad = coeff
        slope = coeff[1]
        @show g = [slope]
        alpha = [alphas[end]]
        Flux.Optimise.update!(opt,alpha, g)
        alpha[1] * (1 + (rand()/100))
    end
end
