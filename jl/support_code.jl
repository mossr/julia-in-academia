# Any imports you need go here.
using PGFPlots
using LinearAlgebra # for dot: ⋅
using SignalTemporalLogic
using Parameters
using BSON
using TikzPictures
using Distributions
using GridInterpolations
using IntervalArithmetic
using LazySets
using Random
import ForwardDiff: gradient

# include("../output/all_algorithm_blocks.jl")

pushPGFPlotsPreamble("\\definecolor{pastelMagenta}{HTML}{FF48CF}")
pushPGFPlotsPreamble("\\definecolor{pastelPurple}{HTML}{8770FE}")
pushPGFPlotsPreamble("\\definecolor{pastelBlue}{RGB}{0,114,178}")
pushPGFPlotsPreamble("\\definecolor{pastelSkyBlue}{RGB}{86,180,233}")
pushPGFPlotsPreamble("\\definecolor{pastelGreen}{RGB}{0,158,115}")
pushPGFPlotsPreamble("\\definecolor{pastelOrange}{RGB}{230,159,0}")
pushPGFPlotsPreamble("\\definecolor{pastelRed}{HTML}{F5615C}")

# Any other supporting code you need goes here.
abstract type Agent end
abstract type Environment end
abstract type Sensor end

struct System
    agent::Agent
    env::Environment
    sensor::Sensor
end

function step(sys::System, s)
    o = sys.sensor(s)
    a = sys.agent(o)
    s′ = sys.env(s, a)
    return (; o, a, s′)
end

function rollout(sys::System; d)
    s = rand(Ps(sys.env))
    τ = []
    for t in 1:d
        o, a, s′ = step(sys, s)
        push!(τ, (; s, o, a))
        s = s′
    end
    return τ
end

@with_kw struct GridWorld <: Environment
    size = (10, 10)                          # dimensions of the grid
    terminal_states = [[5,5],[7,8]]          # goal and obstacle states
    directions = [[0,1],[0,-1],[-1,0],[1,0]] # up, down, left, right
    tprob = 0.7                              # probability do not slip
end

function get_rectangle(lb, ub; color="black", alpha=1.0, linewidth="1pt")
    return "\\draw[$(color), opacity=$(alpha), line width=$(linewidth)] ($(string(lb[1])),$(string(lb[2]))) rectangle ($(string(ub[1])),$(string(ub[2])));"
end

function get_filled_rectangle(lb, ub, color; alpha=1.0, draw="black")
    return "\\filldraw[fill=$(color), draw=$draw, opacity=$alpha] ($(string(lb[1])),$(string(lb[2]))) rectangle ($(string(ub[1])),$(string(ub[2])));"
end

function plot_grid!(ax, size; cell_width=1, color="black!80")
    lbsx = collect(range(0, step=cell_width, length=size[1]))
    lbsy = collect(range(0, step=cell_width, length=size[2]))
    for lbx in lbsx
        for lby in lbsy
            push!(ax, Plots.Command(get_rectangle([lbx, lby], [lbx, lby] .+ cell_width, color=color)))
        end
    end
end

function color_cell!(ax, location, color; cell_width=1, draw="black!80")
    lbs, ubs = (location .- 1) .* cell_width, location .* cell_width
    push!(ax, Plots.Command(get_filled_rectangle(lbs, ubs, color, draw=draw)))
end

function plot_grid_world(env::GridWorld; good_cells=[[7, 8]], bad_cells=[[5, 5]], checkpoint_cells=[], color="black!80")
    ax = Axis(xmin=0, xmax=10, ymin=0, ymax=10)
    ax.style = "xticklabels={,,}, yticklabels={,,}, ticks=none"
    ax.axisEqualImage = true
    plot_grid!(ax, env.size, color=color)
    for bad_cell in bad_cells
        color_cell!(ax, bad_cell, "pastelRed")
    end
    for good_cell in good_cells
        color_cell!(ax, good_cell, "pastelGreen")
    end
    for checkpoint_cell in checkpoint_cells
        color_cell!(ax, checkpoint_cell, "pastelBlue")
    end
    return ax
end

function plot_gw_trajectory!(ax, sample; alpha=1.0, color="black", fill="black", linewidth="1pt", rad="0.75mm", dashed=false, include_marks=false, mark_color=color)
    states = [step.s for step in sample]
    diffs = [states[i+1] - states[i] for i in 1:length(states)-1]
    traj_inds = findall([diff != [0, 0] for diff in diffs])
    traj_inds = [traj_inds; traj_inds[end]+1]
    xs = [s[1] - 0.5 for s in states[traj_inds]]
    ys = [s[2] - 0.5 for s in states[traj_inds]]
    line_style = dashed ? "dashed" : "solid"
    if include_marks
        push!(ax, Plots.Linear(xs[1:2], ys[1:2], style="$line_style, $color, mark=none, line width=$linewidth"))
        push!(ax, Plots.Linear([xs[1]], [ys[1]], style="solid, $color, mark=square*, mark options={$mark_color, fill=$fill, scale=0.75}, line width=$linewidth"))
        push!(ax, Plots.Linear(xs[2:end], ys[2:end], style="rounded corners=$rad, $line_style, $color, mark=*, mark options={$mark_color, fill=$fill, scale=0.75}, line width=$linewidth"))
    else
        push!(ax, Plots.Linear(xs, ys, style="rounded corners=$rad, $line_style, mark=none, $color, opacity=$alpha, line width=$linewidth"))
    end
    push!(ax, get_circle([xs[1], ys[1]], color=fill, alpha=alpha))
    push!(ax, get_circle([xs[end], ys[end]], color=fill, alpha=alpha))
end

get_circle(center; color="black", alpha=1.0) = Plots.Command("\\fill[$(color), opacity=$(alpha)] ($(center[1]), $(center[2])) circle (0.1);")
get_circle(center, radius; color="black", alpha=1.0, draw="none") = Plots.Command("\\fill[$(color), opacity=$(alpha), draw=$draw] ($(center[1]), $(center[2])) circle ($(radius));")

struct SetCategorical{S}
	elements::Vector{S} # Set elements (could be repeated)
	distr::Categorical # Categorical distribution over set elements
	function SetCategorical(elements::AbstractVector{S}) where S
		weights = ones(length(elements))
		return new{S}(elements, Categorical(normalize(weights, 1)))
	end
	function SetCategorical(
			elements::AbstractVector{S},
			weights::AbstractVector{Float64}
		) where S
		ℓ₁ = norm(weights,1)
		if ℓ₁ < 1e-6 || isinf(ℓ₁)
			return SetCategorical(elements)
		end
		distr = Categorical(normalize(weights, 1))
		return new{S}(elements, distr)
	end
	function SetCategorical(elements::AbstractDict{S,Float64}) where S
		return SetCategorical(collect(keys(elements)), collect(values(elements)))
	end
end
Distributions.rand(D::SetCategorical) = D.elements[rand(D.distr)]
Distributions.rand(D::SetCategorical, n::Int) = D.elements[rand(D.distr, n)]
function Distributions.pdf(D::SetCategorical, x)
	sum(e == x ? w : 0.0 for (e,w) in zip(D.elements, D.distr.p))
end
Distributions.logpdf(D::SetCategorical, x) = log(pdf(D, x))
function Distributions.fit(D::SetCategorical, samples)
	inds = [findfirst(map(e->s == e, D.elements)) for s in samples]
	distr = fit(typeof(D.distr), inds)
	return SetCategorical(D.elements, distr.p)
end
Distributions.fit(D::SetCategorical, samples, w::Missing) = fit(D, samples)

@with_kw struct ContinuumWorld <: Environment
    size = [10, 10]                          # dimensions
    terminal_centers = [[4.5,4.5],[6.5,7.5]] # obstacle and goal centers
    terminal_radii = [0.5, 0.5]              # radius of obstacle and goal
    directions = [[0,1],[0,-1],[-1,0],[1,0]] # up, down, left, right
    Σ = 0.5 * I(2)
end

Ds(env::ContinuumWorld, s, a) = MvNormal(zeros(2), env.Σ)
(env::ContinuumWorld)(s, a) = env(s, a, rand(Ds(env, s, a)))
function (env::ContinuumWorld)(s, a, x)
    is_terminal = [norm(s .- c) ≤ r 
            for (c, r) in zip(env.terminal_centers, env.terminal_radii)]
    if any(is_terminal)
        return s
    else
        dir = normalize(env.directions[a] .+ x)
        return clamp.(s .+ dir, [0, 0], env.size)
    end
end
Ps(env::ContinuumWorld) = SetCategorical([[0.5, 0.5]])

struct InterpAgent <: Agent
    grid # grid of discrete states using GridInteroplations.jl
    Q    # corresponding state-action values
end
(c::InterpAgent)(s) = argmax(interpolate(c.grid, q, s) for q in c.Q)

struct IdealSensor <: Sensor end
(sensor::IdealSensor)(s) = s

function plot_cw(env::ContinuumWorld;
    color="black!92!white",
    good_centers=[[6.5, 7.5]], good_radii=[0.5],
    bad_centers=[[4.5, 4.5]], bad_radii=[0.5])
    ax = Axis(xmin=0, xmax=env.size[1], ymin=0, ymax=env.size[2])
    ax.style = "axis background/.style={fill=$color}, xticklabels={,,}, yticklabels={,,}, ticks=none"
    ax.axisEqualImage = true
    for (c, r) in zip(bad_centers, bad_radii)
        push!(ax, get_circle(c, r, color="pastelRed"))
    end
    for (c, r) in zip(good_centers, good_radii)
        push!(ax, get_circle(c, r, color="pastelGreen"))
    end
    return ax
end

function plot_cw_trajectory!(ax, sample; alpha=1.0, color="black", linewidth="1pt", rad="0.75mm", dashed=false, circles=true)
    states = [step.s for step in sample]
    diffs = [states[i+1] - states[i] for i in 1:length(states)-1]
    traj_inds = findall([diff != [0, 0] for diff in diffs])
    traj_inds = [traj_inds; traj_inds[end]+1]
    xs = [s[1] for s in states[traj_inds]]
    ys = [s[2] for s in states[traj_inds]]
    if dashed
        push!(ax, Plots.Linear(xs, ys, style="rounded corners=$rad, dashed, mark=none, $color, opacity=$alpha, line width=$linewidth"))
    else
        push!(ax, Plots.Linear(xs, ys, style="rounded corners=$rad, solid, mark=none, $color, opacity=$alpha, line width=$linewidth"))
    end
    if circles
        push!(ax, get_circle([xs[1], ys[1]], color=color, alpha=alpha))
        push!(ax, get_circle([xs[end], ys[end]], color=color, alpha=alpha))
    end
end

@with_kw struct CollisionAvoidance <: Environment
	ddh_max::Float64 = 1.0                # maximum vertical acceleration
    𝒜::Vector{Float64} = [-5.0, 0.0, 5.0] # vertical rate commands
	Ds::Sampleable = Normal()             # vertical rate noise
end

Ds(env::CollisionAvoidance, s, a) = env.Ds
(env::CollisionAvoidance)(s, a) = env(s, a, rand(Ds(env, s, a)))
function (env::CollisionAvoidance)(s, a, x)
	a = env.𝒜[a]
	h, dh, a_prev, τ = s
	h = h + dh
    if a != 0.0
        if abs(a - dh) < env.ddh_max
            dh += a
        else
            dh += sign(a - dh) * env.ddh_max
        end
    end
    a_prev = a
    τ = max(τ - 1.0, -1.0)
	return [h, dh + x, a_prev, τ]
end
Ps(env::CollisionAvoidance) = product_distribution(Uniform(-100, 100), 
                                    Uniform(-10, 10), 
                                    DiscreteNonParametric([0], [1.0]), 
                                    DiscreteNonParametric([40], [1.0]))

function cas_axis(; collision_threshold=50, xmin=0, xmax=40, ymin=-400, ymax=400, color="black!92!white")
    ax = Axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, style="x dir = reverse")
    push!(ax, Plots.Command(get_filled_rectangle([xmin, -collision_threshold], [1, collision_threshold], "pastelRed", alpha=0.5, draw="white")))
    ax.xlabel = L"$t_\text{col}$ (s)"
    ax.ylabel = L"$h$ (m)"
    ax.style *= ", axis background/.style={fill=$color}"
    return ax
end

function plot_cas_traj!(ax, sample; color="black", alpha=1.0, linewidth="1pt")
    states = [step.s for step in sample]
    τs = [s[4] for s in states]
    hs = [s[1] for s in states]
    push!(ax, Plots.Linear(τs, hs, style="$color, solid, line width=$linewidth, mark=none, opacity=$alpha"))
end

function plot_interval(f, input_interval, output_interval, xmin, xmax, ymin, ymax; color="pastelBlue", cinput=color, fxmax=xmax, alpha=0.3, draw="none", linewidth="2pt", lwdash="1.5pt")
    xlo, xhi = inf(input_interval), sup(input_interval)
    ylo, yhi = inf(output_interval), sup(output_interval)
    ax = Axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, width="5cm", height="5cm")
    push!(ax, Plots.Linear(f, (xmin, fxmax), style="solid, gray, line width=$linewidth"))
    xbar_hw = 0.03 * (ymax - ymin)
    ybar_hw = 0.03 * (xmax - xmin)
    push!(ax, Plots.Command("\\draw[$cinput, line width=2pt] (axis cs: $(xlo), $(ymin-xbar_hw)) -- ($(xlo), $(ymin+ybar_hw))"))
    push!(ax, Plots.Command("\\draw[$cinput, line width=2pt] (axis cs: $(xhi), $(ymin-xbar_hw)) -- ($(xhi), $(ymin+xbar_hw))"))
    push!(ax, Plots.Command("\\draw[$cinput, line width=2pt] (axis cs: $(xlo), $(ymin)) -- ($(xhi), $(ymin))"))
    push!(ax, Plots.Command("\\draw[$color, line width=2pt] (axis cs: $(xmin-ybar_hw), $(ylo)) -- ($(xmin+ybar_hw), $(ylo))"))
    push!(ax, Plots.Command("\\draw[$color, line width=2pt] (axis cs: $(xmin-ybar_hw), $(yhi)) -- ($(xmin+ybar_hw), $(yhi))"))
    push!(ax, Plots.Command("\\draw[$color, line width=2pt] (axis cs: $(xmin), $(ylo)) -- ($(xmin), $(yhi))"))
    push!(ax, Plots.Command(get_filled_rectangle([xmin, ylo], [xmax, yhi], color, alpha=alpha, draw=draw)))
    push!(ax, Plots.Command("\\draw[dashed, $cinput, line width=$lwdash] (axis cs: $(xlo), $(ymin)) -- ($(xlo), $(f(xlo)))"))
    push!(ax, Plots.Command("\\draw[dashed, $cinput, line width=$lwdash] (axis cs: $(xhi), $(ymin)) -- ($(xhi), $(f(xhi)))"))
    ax.style = "clip=false"
    return ax
end

abstract type Agent end
abstract type Environment end
abstract type Sensor end

struct System
    agent::Agent
    env::Environment
    sensor::Sensor
end
####################

#################### introduction 2
function step(sys::System, s)
    o = sys.sensor(s)
    a = sys.agent(o)
    s′ = sys.env(s, a)
    return (; o, a, s′)
end

function rollout(sys::System; d)
    s = rand(Ps(sys.env))
    τ = []
    for t in 1:d
        o, a, s′ = step(sys, s)
        push!(τ, (; s, o, a))
        s = s′
    end
    return τ
end

struct Disturbance
    xa # agent disturbance
    xs # environment disturbance
    xo # sensor disturbance
end

struct DisturbanceDistribution
    Da # agent disturbance distribution
    Ds # environment disturbance distribution
    Do # sensor disturbance distribution
end

function step(sys::System, s, D::DisturbanceDistribution)
    xo = rand(D.Do(s))
    o = sys.sensor(s, xo)
    xa = rand(D.Da(o))
    a = sys.agent(o, xa)
    xs = rand(D.Ds(s, a))
    s′ = sys.env(s, a, xs)
    x = Disturbance(xa, xs, xo)
    return (; o, a, s′, x)
end

struct ProportionalController <: Agent
    k
end
(c::ProportionalController)(s, a=missing) = c.k' * s
Πo(agent::ProportionalController) = agent.α'

@with_kw struct InvertedPendulum <: Environment
    m::Float64 = 1.0     # mass of the pendulum
    l::Float64 = 1.0     # length of the pendulum
    g::Float64 = 10.0    # acceleration due to gravity
    dt::Float64 = 0.05   # time step
    ω_max::Float64 = 8.0 # maximum angular velocity
    a_max::Float64 = 2.0 # maximum torque
end

function (env::InvertedPendulum)(s, a, xs=missing)
    θ, ω = s[1], s[2]
    dt, g, m, l = env.dt, env.g, env.m, env.l
    ω = ω + (3g / (2 * l) * sin(θ) + 3 * a / (m * l^2)) * dt
    θ = θ + ω * dt
    return [θ, ω]
end

Ps(env::InvertedPendulum) = Product([Uniform(-π / 16, π / 16), 
                                     Uniform(-1.0, 1.0)])

struct AdditiveNoiseSensor <: Sensor
	Do # noise distribution
end

(sensor::AdditiveNoiseSensor)(s) = sensor(s, rand(Do(sensor, s)))
(sensor::AdditiveNoiseSensor)(s, x) = s + x
Do(sensor::AdditiveNoiseSensor, s) = sensor.Do
Os(sensor::AdditiveNoiseSensor) = I

abstract type ReachabilityAlgorithm end

struct NaturalInclusion <: ReachabilityAlgorithm
    h # time horizon
end

function r(sys, x)
    s, 𝐱 = extract(sys.env, x)
    τ = rollout(sys, s, 𝐱)
    return τ[end].s
end

to_hyperrectangle(𝐈) = Hyperrectangle(low=[i.lo for i in 𝐈], 
                                      high=[i.hi for i in 𝐈])

function reachable(alg::NaturalInclusion, sys)
    𝐈′s = []
    for d in 1:alg.h
        𝐈 = intervals(sys, d)
        push!(𝐈′s, r(sys, 𝐈))
    end
    return UnionSetArray([to_hyperrectangle(𝐈′) for 𝐈′ in 𝐈′s])
end

function step(sys::System, s, x)
    o = sys.sensor(s, x.xo)
    a = sys.agent(o, x.xa)
    s′ = sys.env(s, a, x.xs)
    return (; o, a, s′)
end

function rollout(sys::System, s, 𝐱; d=length(𝐱))
    τ = []
    for t in 1:d
        x = 𝐱[t]
        o, a, s′ = step(sys, s, x)
        push!(τ, (; s, o, a, x))
        s = s′
    end
    return τ
end

function inv_pendulum_state_space(; failure_threshold = π/4, xmin=-1.2, xmax=1.2, ymin=-1.2, ymax=1.2)
    ax = Axis(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
    push!(ax, Plots.Command(get_filled_rectangle([xmin, ymin], [-failure_threshold, ymax], "pastelRed", alpha=0.5)))
    push!(ax, Plots.Command(get_filled_rectangle([failure_threshold, ymin], [xmax, ymax], "pastelRed", alpha=0.5)))
    ax.xlabel = L"$\theta$ (rad)"
    ax.ylabel = L"$\omega$ (rad/s)"
    return ax
end

function plot_polytope!(ax, vertices; fill_color="pastelSkyBlue", fill_alpha=1.0, draw_color="black", draw_alpha=1.0, linestyle="solid", linewidth="0.5pt")
    comm = "\\filldraw[$linestyle, fill=$fill_color, draw=$draw_color, fill opacity=$fill_alpha, draw opacity=$draw_alpha, line width=$linewidth] "
    for vert in vertices
        comm *= "($(vert[1]), $(vert[2])) -- "
    end
    comm *= "cycle;"
    push!(ax, Plots.Command(comm))
end

function plot_polytope!(ax, set::ConvexSet; fill_color="pastelSkyBlue", fill_alpha=1.0, draw_color="black", draw_alpha=1.0, linestyle="solid", linewidth="0.5pt")
    vertices = LazySets.vertices_list(set)
    comm = "\\filldraw[$linestyle, fill=$fill_color, draw=$draw_color, fill opacity=$fill_alpha, draw opacity=$draw_alpha, line width=$linewidth] "
    for vert in vertices
        comm *= "($(vert[1]), $(vert[2])) -- "
    end
    comm *= "cycle;"
    push!(ax, Plots.Command(comm))
end

function plot_polytope!(ax, set::Hyperrectangle; fill_color="pastelSkyBlue", fill_alpha=1.0, draw_color="black", draw_alpha=1.0, linestyle="solid", linewidth="0.5pt")
    vertices = LazySets.vertices_list(set)
    vertices = [vertices[1], vertices[2], vertices[4], vertices[3]]
    comm = "\\filldraw[$linestyle, fill=$fill_color, draw=$draw_color, fill opacity=$fill_alpha, draw opacity=$draw_alpha, line width=$linewidth] "
    for vert in vertices
        comm *= "($(vert[1]), $(vert[2])) -- "
    end
    comm *= "cycle;"
    push!(ax, Plots.Command(comm))
end
