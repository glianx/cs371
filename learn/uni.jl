using UnicodePlots

x = 1:10
y = x .^ -1
z = sin.(x)
@time p = lineplot(x, y)
@time q = lineplot(x, z)
display(p)
display(q)