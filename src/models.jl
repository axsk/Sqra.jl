export TripleWell, LJCluster


abstract type Model end

EMSimulation(m::Model) = EMSimulation()
potential(m::Model, x) = Base.error("undefined potential for model $m")
potential(m::Model) = x->potential(m, x)
x0default(m::Model) = Base.error("undefined x0 for model $m")
beta(m::Model) = sigma_to_beta(m.sigma)
sigma(m::Model) = m.sigma

classify(m::Model, x::Matrix) = mapslices(x->classify(m, x), x, dims=1) |> vec

### Triple Well

@with_kw struct TripleWell <: Model
	sigma = 1.
	box = [-3. 3; -2 2]
end

EMSimulation(m::TripleWell) = EMSimulation()

x0default(m::TripleWell) = [0., 0.]

potential(m::TripleWell, x) = triplewell(x)
dim(m::TripleWell) = 2
boundingbox(m::TripleWell) = m.box


function classify(m::TripleWell, x::Matrix)
	A = sum(abs2, x .- [ 1., 0], dims=1) .< 0.25
	B = sum(abs2, x .- [-1., 0], dims=1) .< 0.25
	vec(A - B)
end

function triplewell(x::AbstractVector)
    x, y = x
    V =  (3/4 * exp(-x^2 - (y-1/3)^2)
        - 3/4 * exp(-x^2 - (y-5/3)^2)
        - 5/4 * exp(-(x-1)^2 - y^2)
        - 5/4 * exp(-(x+1)^2 - y^2)
        + 1/20 * x^4 + 1/20 * (y-1/3)^4)
end

### LJCluster

@with_kw struct LJCluster <: Model
	epsilon = 1
    r0 = 1/3
    harm = 1
    sigma = 1/2
	box = [-ones(6) ones(6)] .* 0.8
end

EMSimulation(m::LJCluster) = EMSimulation(dt=0.001, maxdelta=0.1)

x0default(::LJCluster) = [0.19920158482463968, 0.13789462153196408, -0.1709575705426315, 0.0784533378749835, 0.06778720715969005, -0.2112155752270007]
potential(m::LJCluster, x) = lennard_jones_harmonic(x; sigma=m.sigma, epsilon=m.epsilon, harm=m.harm)

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

@assert begin
	x = rand(6)
	lennard_jones_harmonic(x) == ljloop(x)
end


function classify(m::LJCluster, coords::Vector)
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


@userplot CloudPlot


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
