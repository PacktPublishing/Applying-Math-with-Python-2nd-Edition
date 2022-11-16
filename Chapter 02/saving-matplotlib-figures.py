import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1, 5, 0.1)
y = x*x

fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("Graph of $y=x^2$")
ax.set_xlabel("$x$")
ax.set_ylabel("$y$")
fig.savefig("savingfigs.png", dpi=300)
plt.show()
