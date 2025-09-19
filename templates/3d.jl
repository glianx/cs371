using GLMakie

x = -10:0.1:10
y = -10:0.1:10

z = [xx^2 - yy^2 for xx in x, yy in y]

fig = Figure()
ax = Axis3(fig[1,1])

surface!(x,y,z)
fig