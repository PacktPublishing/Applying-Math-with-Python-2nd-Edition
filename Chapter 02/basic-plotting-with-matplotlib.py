import numpy as np
import matplotlib.pyplot as plt

# set up
def f(x):
    return x*(x-2)*np.exp(3 - x)

def g(x):
    return x**2

def h(x):
    return 1 - x


x = np.linspace(-0.5, 3.0) # 50 values between -0.5 and 3.0


y1 = f(x)  # evaluate f on the x points
y2 = g(x)  # evaluate g on the x points
y3 = h(x)  # evaluate h on the x points


fig, ax = plt.subplots()

ax.plot(x, y1, "k")  # black solid line style


ax.plot(x, y2, "k--")  # black dashed line style
ax.plot(x, y3, "k.-")  # black dot-dashed line style


ax.set_title("Plot of the functions f, g, and h")
ax.set_xlabel("x")
ax.set_ylabel("y")


ax.legend(["f", "g", "h"])

ax.text(0.4, 2.0, "Intersection")

plt.show()