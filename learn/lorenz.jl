using GLMakie

# Params and integrator (explicit RK4; stable and smooth)
σ, ρ, β = 10.0, 28.0, 8/3
dt = 0.0075

rk4(u) = begin
    x,y,z = u
    function f(x,y,z)
        dx = σ*(y - x)
        dy = x*(ρ - z) - y
        dz = x*y - β*z
        (dx, dy, dz)
    end
    k1 = f(x,y,z)
    k2 = f(x + 0.5dt*k1[1], y + 0.5dt*k1[2], z + 0.5dt*k1[3])
    k3 = f(x + 0.5dt*k2[1], y + 0.5dt*k2[2], z + 0.5dt*k2[3])
    k4 = f(x + dt*k3[1], y + dt*k3[2], z + dt*k3[3])
    (
        x + dt*(k1[1] + 2k2[1] + 2k3[1] + k4[1])/6,
        y + dt*(k1[2] + 2k2[2] + 2k3[2] + k4[2])/6,
        z + dt*(k1[3] + 2k2[3] + 2k3[3] + k4[3])/6
    )
end

# Observable state and trail (bounded length for speed)
u0  = (1.0, 1.0, 1.0)
trail_len = 4000
trail = Observable(Point3f0[])

# prefill to avoid reallocs
resize!(trail[], trail_len)
for i in eachindex(trail[]); trail[][i] = Point3f0(NaN,NaN,NaN); end
notify(trail)

fig = Figure(resolution=(900, 700))
ax  = Axis3(fig[1,1], aspect=:data, azimuth=0.2, elevation=0.5,
            title="Lorenz attractor (GPU, RK4, fading trail)")
ax.showgrid = (false, false, false)

# Fading trail via per-vertex alpha
colors = Observable(fill(RGBAf0(0.1, 0.7, 1.0, 0.0), trail_len))
lineplot = lines!(ax, trail; color=colors, linewidth=2)
headplot = scatter!(ax, Point3f0(0), markersize=12, color=:yellow)

xlims!(ax, -30, 30); ylims!(ax, -30, 30); zlims!(ax, 0, 50)

# Animation
u = u0
θ = 0.0
record(fig, "lorenz.gif", 1:1200) do i
    # advance several small steps per frame for smoothness
    for _ in 1:4
        u = rk4(u)
    end

    # shift and append
    tr = trail[]
    # drop first, append new point
    @inbounds begin
        for k in 1:trail_len-1
            tr[k] = tr[k+1]
        end
        tr[end] = Point3f0(u[1], u[2], u[3])
    end
    notify(trail)

    # recompute fading colors (oldest transparent -> newest opaque)
    cs = colors[]
    @inbounds for k in 1:trail_len
        a = Float32(k)/trail_len
        cs[k] = RGBAf0(0.15 + 0.6a, 0.4 + 0.5a, 1.0 - 0.7a, a^1.5)
    end
    notify(colors)

    headplot[1] = tr[end]  # move bright head marker

    # slow orbiting camera
    θ += 0.01
    az = 0.4 + 0.3*sin(θ)
    el = 0.3 + 0.2*cos(0.7θ)
    ax.azimuth[] = az
    ax.elevation[] = el
end