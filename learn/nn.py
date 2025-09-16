import matplotlib.pyplot as plt
import numpy as np
import random as rnd

n = 100
xs = [rnd.random() for _ in range(n)]
ys = [rnd.random() for _ in range(n)]
s  = [100 * rnd.random() for _ in range(n)]

# fig, ax = plt.subplots()
# plt.scatter(xs, ys, s)
# plt.show()

fig = plt.figure()
ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
ax.scatter(xs, ys)
plt.show()