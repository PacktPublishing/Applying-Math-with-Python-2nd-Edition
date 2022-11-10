import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


x = np.array([ 0.19, -0.82, 0.8 , 0.95, 0.46, 0.71,
     -0.86, -0.55,   0.75,-0.98, 0.55, -0.17, -0.89,
         -0.4 , 0.48, -0.09, 1., -0.03, -0.87, -0.43])
y = np.array([-0.25, -0.71, -0.88, 0.55, -0.88, 0.23,
      0.18,-0.06, 0.95, 0.04, -0.59, -0.21, 0.14, 0.94,
          0.51, 0.47, 0.79, 0.33, -0.85, 0.19])
z = np.array([-0.04, 0.44, -0.53, 0.4, -0.31, 0.13,
      -0.12, 0.03, 0.53, -0.03, -0.25, 0.03, -0.1 ,
          -0.29, 0.19, -0.03, 0.58, -0.01, 0.55, -0.06])


fig = plt.figure(tight_layout=True)
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_trisurf(x, y, z, cmap="gray")

ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.set_title("Approximate surface")

ax2 = fig.add_subplot(1, 2, 2)
ax2.tricontour(x, y, z, cmap="gray")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("Approximate contours")

plt.show()
