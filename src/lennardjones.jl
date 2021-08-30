export Simulation, run


const x0gen =  [0.19920158482463968
0.13789462153196408
-0.1709575705426315
0.0784533378749835
0.06778720715969005
-0.2112155752270007]


### Lennard Jones specifics

function lennard_jones_harmonic(x; sigma=1/4, epsilon=1, harm=1)
    x = reshape(x, 2, 3)
    u = 0.
    @views for i in 1:3
        for j in i+1:3
            r = sigma^2 / sum(abs2, (x[:,i] .- x[:,j]))
            u += 4*epsilon * (r^6 - r^3)
        end
    end
	u += sum(abs2, x) * harm
    return u
end

function ljloop(x, sigma=1/4, epsilon=1., harm=1.)
    u = 0.
    @inbounds for i = 0:2
        for j in i+1:2
            r = (x[1+2*i] - x[1+2*j]) ^ 2 + (x[2+2*i] - x[2+2*j])^2
            r = sigma^2 / r
            u += 4*epsilon * (r^6 - r^3)
        end
    end
    u += sum(abs2, x) * harm
    return u
end

classify(coords::Matrix) = mapslices(classify, coords, dims=1) |> vec

function classify(coords::Vector)
    ab = coords[3:4] - coords[1:2]
    ac = coords[5:6] - coords[1:2]

    angle = acos(min(dot(ab, ac) / norm(ab) / norm(ac), 1))
    offset = angle - pi/3  # offset to 60 degree
    if (abs(offset) < pi/12)  # +- 15 degrees
        return sign(ab[1]*ac[2] - ab[2]*ac[1])
    else
        return 0
    end
end

normalform(x::Matrix) = mapslices(normalform, x, dims=1)

" shift the first particle to 0 and rotate the second onto the x axis"
function normalform(x)
    x = reshape(x, 2, div(length(x),2))
    x = x .- x[:,1]

    one = [1,0]
    b   = normalize(x[:,2])
    B   = [b[1] -b[2]
           b[2]  b[1]]
    E   = [1 0
           0 1]
    A   =  E / B
    reshape(A * x, length(x))
end


@with_kw struct Simulation
	x0 = x0gen
	epsilon = 1
    r0 = 1/3
    harm = 1
    sigma = 1/2
    dt=0.001
    nsteps=100000
    maxdelta=0.1
	seed = 1
	x=nothing
	u=nothing
end

using Setfield

function permute(s::Simulation)
	xp = s.x
	xp = hcat(xp, xp[[4,3,2,1,6,5],:])
	s = @set s.x = xp
	s = @set s.u = vcat(s.u, s.u)
	s = @set s.nsteps = s.nsteps*2
end

function run_parallel(sim::Simulation; seeds=1:Threads.nthreads())
	copies=length(seeds)
	results = Array{Simulation}(undef, copies)
	batch = cld(sim.nsteps, copies)

	x = Array{Float64}(undef, length(sim.x0), batch * copies)
	u = Array{Float64}(undef, batch * copies)

	Threads.@threads for i in 1:copies
		results[i] = run(Simulation(sim, nsteps = batch, seed = seeds[i]))

		x[:, (i-1)*batch + 1 : i*batch] = results[i].x[:, 1:batch]
		u[(i-1)*batch + 1 : i*batch] = results[i].u[1:batch]
	end

	return Sqra.Simulation(sim, x = x, u = u)
end

@memoize PermaDict("cache/sim_") function run(params::Simulation)
	@unpack_Simulation params

	Random.seed!(seed)
	potential(x) = lennard_jones_harmonic(x; epsilon=epsilon, sigma=r0, harm=harm)

	x, u = eulermaruyama(x0 |> vec, potential, sigma, dt, nsteps, maxdelta=maxdelta,
		progressbar=Threads.threadid() == 1)
	#u = mapslices(potential, x, dims=1) |> vec

	@pack_Simulation
end

function extend(s::Simulation, n)
	e = Simulation(s, x0 = s.x[:, end], nsteps=n)
	e = run(e)
	Simulation(s, x=hcat(s.x, e.x[:, 2:end]), u = vcat(s.u, e.u[2:end]), nsteps=s.nsteps+n)
end



@userplot CloudPlot

@recipe function f(s::Simulation)
	CloudPlot((s.x, ))
end

""" plotting recipe for plotting multiple cluster states """
@recipe function f(p::CloudPlot; select=:all, normalize=true, com=nothing, triangles=false)
	x,  = p.args
	legend --> false
	colorbar --> true
	aspect_ratio --> :equal
	seriescolor --> :roma
	#seriescolor --> :hawaii
	seriesalpha --> 0.3

	#preprocessing
	n = size(x,2)
	if isa(select, Integer)
		i = 1:cld(size(x,2), select):size(x,2)
	elseif isa(select, AbstractRange) || isa(select, Vector)
		i = select
	else
		i = 1:n
	end
	x = x[:,i]
	normalize && (x = normalform(x))

	if isnothing(com)
		z = [1]
	else
		z = com[i]
	end

	if triangles
		@series begin
			seriestype := :path
			line_z := (z')
			n = x
			xs = [n[1:2:end,:]; n[[1],:]]
			ys = [n[2:2:end,:]; n[[2],:]]
			xs, ys
		end
	end

	@series begin
		seriestype := :scatter
		marker_z := repeat(z, inner=(1,3))

		x[1:2:end,:]', x[2:2:end,:]'
 	end

end
