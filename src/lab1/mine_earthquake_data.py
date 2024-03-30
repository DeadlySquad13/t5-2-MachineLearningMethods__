import calendar
import datetime as dt
import os
from pathlib import Path

import pandas as pd
import requests

data_path = Path("data")
earthquakes_data_path = data_path / "earthquakes"
earthquakes_per_month_path = earthquakes_data_path / "per-month"

URL = "https://earthquake.usgs.gov/fdsnws/event/1/query.csv?starttime={start}&endtime={end}&minmagnitude=2.0&orderby=time"


def get_file_name(year, month):
    return earthquakes_per_month_path / "{yr}_{m}.csv".format(yr=year, m=month)


for year in range(2000, 2019):
    for month in range(1, 13):
        data_chunk_file_name = get_file_name(year, month)

        if os.path.isfile(data_chunk_file_name):
            continue
        _, ed = calendar.monthrange(year, month)
        start = dt.datetime(year, month, 1)
        end = dt.datetime(year, month, ed, 23, 59, 59)
        with open(
            data_chunk_file_name,
            "w",
            encoding="utf-8",
        ) as f:
            url_for_current_data_chunk = URL.format(start=start, end=end)
            f.write(requests.get(url_for_current_data_chunk).content.decode("utf-8"))

dfs = []
for year in range(2000, 2019):
    for month in range(1, 13):
        data_chunk_file_name = get_file_name(year, month)

        if not os.path.isfile(data_chunk_file_name):
            print(f"Not data chunk found for year: {year}, month: {month}")
            continue
        df = pd.read_csv(data_chunk_file_name, dtype={"nst": "float64"})
        dfs.append(df)
df = pd.concat(dfs, sort=True)
df.to_parquet(earthquakes_data_path / "earthquakes.parq", "fastparquet")

# Reprojected, cleaned and gzip (not snappy)

from holoviews.util.transform import lon_lat_to_easting_northing  # noqa

df = pd.read_parquet(earthquakes_data_path / "earthquakes.parq")

cleaned_df = df.copy()
cleaned_df["mag"] = df.mag.where(df.mag > 0)
cleaned_df = cleaned_df.reset_index()

x, y = lon_lat_to_easting_northing(cleaned_df.longitude, cleaned_df.latitude)
cleaned_projected = cleaned_df.join(
    [pd.DataFrame({"easting": x}), pd.DataFrame({"northing": y})]
)

cleaned_projected.to_parquet(
    earthquakes_data_path / "earthquakes-projected.parq",
    "fastparquet",
    compression="gzip",
    file_scheme="simple",
)
