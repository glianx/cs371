using GLMakie

x = range(-3, 3, length=100)
y = range(-3, 3, length=100)
z = [sin(√(xi^2 + yi^2)) / √(xi^2 + yi^2) for xi in x, yi in y]

surface(x, y, z, colormap=:viridis)