# %%
# Loading extension for reloading editable packages (pip install -e .)
# %load_ext autoreload

# %%
# Reloading editable packages.
# %autoreload
# from lab1.main import get_results

import bokeh
# %%
import datashader as ds  # noqa
import holoviews as hv
import panel as pn
from packaging.version import Version

min_versions = dict(ds="0.15.1", bokeh="3.2.0", hv="1.16.2", pn="1.2.0")

for lib, ver in min_versions.items():
    v = globals()[lib].__version__
    if Version(v) < Version(ver):
        print("Error: expected {}={}, got {}".format(lib, ver, v))

# %%
hv.extension("bokeh", "matplotlib")

# %%
import pathlib

try:
    import pandas as pd

    columns = ["depth", "id", "latitude", "longitude", "mag", "place", "time", "type"]
    path = pathlib.Path("../../data/earthquakes-projected.parq")
    df = pd.read_parquet(path, columns=columns, engine="fastparquet")
    df.head()
except RuntimeError as e:
    print("The data cannot be read: %s" % e)

# %%
print(df.shape)
df.head()

# %%
small_df = df.sample(frac=0.01)
print(small_df.shape)
small_df.head()

# %%
# %matplotlib inline

# %%
small_df.plot.scatter(x="longitude", y="latitude")

# %%
import hvplot.pandas

small_df.hvplot.scatter(x="longitude", y="latitude")
