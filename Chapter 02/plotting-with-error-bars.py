import numpy as np
import matplotlib.pyplot as plt

measurement_id = np.arange(1, 11)
measurements = np.array([2.3, 1.9, 4.4, 1.5, 3.0, 3.3, 2.9, 2.6, 4.1, 3.6])
err = np.array([0.1]*10)

fig, ax = plt.subplots()

ax.errorbar(measurement_id, measurements, yerr=err, fmt="kx", capsize=2.0)

ax.set_title("Plot of measurements and their estimated error")
ax.set_xlabel("Measurement ID")
ax.set_ylabel("Measurement (cm)")


ax.set_xticks(measurement_id)

plt.show()
