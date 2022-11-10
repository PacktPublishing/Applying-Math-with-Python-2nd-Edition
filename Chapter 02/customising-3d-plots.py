import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

t = np.linspace(-5, 5)
x, y = np.meshgrid(t, t)
z = np.exp(-((x-2.)**2 + (y-3.)**2)/4) - np.exp(-((x+3.)**2 + (y+2)**2)/3)


fig = plt.figure()

ax = fig.add_subplot(projection="3d", proj_type="ortho")

ax.plot_surface(x, y, z, cmap="gray", vmin=-1.2, vmax=1.2)
ax.set_title("Customized 3D surface plot")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")


plt.show()

