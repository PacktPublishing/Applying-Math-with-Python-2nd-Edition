import geopandas
import geoplot
import matplotlib.pyplot as plt

world = geopandas.read_file(
        geopandas.datasets.get_path("naturalearth_lowres")
)

cities = geopandas.read_file(
        geopandas.datasets.get_path("naturalearth_cities")
)

fig, ax = plt.subplots()
geoplot.polyplot(world, ax=ax, alpha=0.7)


geoplot.pointplot(cities, ax=ax, fc="k", marker="2")
ax.axis((-180, 180, -90, 90))


plt.show()
