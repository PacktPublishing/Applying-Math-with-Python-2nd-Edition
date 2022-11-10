import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-5, 5)
Y = np.linspace(-5, 5)

x, y = np.meshgrid(X, Y)

z = np.exp(-((x - 2.)**2 + (y - 3.)**2)/4)  - np.exp(-((x + 3.)**2 + (y + 2.)**2)/3)

from mpl_toolkits import mplot3d

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.plot_surface(x, y, z, cmap="gray")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.set_title("Graph of the function f(x, y)")

plt.show()  # paused here


fig, ax = plt.subplots()
ax.contour(x, y, z, cmap="gray")
ax.set_title("Contours of f(x, y)")
ax.set_xlabel("x")
ax.set_ylabel("y")


plt.show()
