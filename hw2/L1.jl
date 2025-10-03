using GLMakie
xx = -10:0.1:10
yy = -10:0.1:10
zz = [abs(x - y) + abs(1 - x) + abs(x + y) for x in xx, y in yy]

fig = Figure()
ax = Axis3(fig[1,1])

surface!(xx,yy,zz)
fig