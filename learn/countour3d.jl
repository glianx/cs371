using GLMakie

# grid ranges (use LinRange or colon ranges)
x = LinRange(1, 10, 30)
y = LinRange(1, 10, 30)
z = LinRange(1, 10, 30)

# scalar field on the grid
f(x, y, z) = x^2 + y^2 + z^2
vol = [f(ix, iy, iz) for ix in x, iy in y, iz in z]

# figure + axis
fig = Figure(resolution = (800, 800))
ax = Axis3(fig[1,1]; perspectiveness = 0.5, azimuth = 2.19,
           elevation = 0.57, aspect = (1, 1, 1))

# correct call on stable
contour3d!(ax, x, y, z, vol; levels=10, colormap=:viridis, transparency=true)

display(fig)  # force window to show if REPL