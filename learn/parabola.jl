using Plots

x = 1:100
y = sin.(x)

plot(x, y, label = "y = sin(x)", xlabel = "x", ylabel = "y")
savefig("hello_plot.png")