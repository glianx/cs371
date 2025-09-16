import matplotlib.pyplot as plt

fig, ax = plt.subplots()
xs = [x for x in range(1,10)]
ys = [x ** 2 for x in xs]

ax.plot(xs,ys)
plt.show()