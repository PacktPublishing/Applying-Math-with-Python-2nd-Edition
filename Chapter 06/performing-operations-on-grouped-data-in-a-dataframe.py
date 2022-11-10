import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

rng = np.random.default_rng(12345)
from matplotlib.rcsetup import cycler
plt.rc("axes", prop_cycle=cycler(c=["k"]*3, ls=["-", "--", "-."]))

labels1 = rng.choice(["A", "B", "C"], size=50)
labels2 = rng.choice([1, 2], size=50)
data = rng.normal(0.0, 2.0, size=50)

df = pd.DataFrame({"label1": labels1, "label2": labels2, "data": data})

df["first_group"] = df.groupby("label1")["data"].cumsum()
print(df.head())


grouped = df.groupby(["label1", "label2"])
df["second_group"] = grouped["data"].transform(
    lambda d: d.rolling(2, min_periods=1).mean())

print(df.head())

print(df[df["label1"] == "C"].head())


fig, ax = plt.subplots()
df.groupby("label1")["first_group"].plot(ax=ax)
ax.set(title="Grouped data cumulative sums", xlabel="Index", ylabel="value")
ax.legend()


plt.show()
