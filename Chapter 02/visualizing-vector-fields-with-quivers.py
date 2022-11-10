import numpy as np
import matplotlib.pyplot as plt


def f(x, y): 
    v = x**2 + y**2
    return np.exp(-2*v)*(x + y), np.exp(-2*v)*(x - y)


t = np.linspace(-1., 1.)
x, y = np.meshgrid(t, t)

dx, dy = f(x, y)


fig, ax = plt.subplots()
ax.quiver(x, y, dx, dy)

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Quiver plot of a vector field")

plt.show()


