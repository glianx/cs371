using Pkg; Pkg.add("Plots")
using Plots
gr()   # GR backend supports 3D

x = y = -2:0.1:2
z = [sin(sqrt(xi^2 + yi^2)) for xi in x, yi in y]

surface(x, y, z, xlabel="x", ylabel="y", zlabel="z")
savefig("surface.png")