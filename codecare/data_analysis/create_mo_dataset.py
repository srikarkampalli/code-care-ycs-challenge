import pandas as pd  # reading CSVs
from sklearn.metrics.pairwise import (
    haversine_distances,
)  # for matching the nodes to hospitals
import numpy as np  # helps in haversine with radians

# read the codecare and the Missouri nodes dataframe
codecare_df = pd.read_csv("../../data/codecare_data.csv")
mo_nodes_df = pd.read_csv("../../data/mo_nodes.csv")

# Filter for only MO data
codecare_df = codecare_df[
    codecare_df["state"].str.contains("MO", regex=False, na=False, case=False)
]

# Rename lat and lon columns for clarity
mo_nodes_df = mo_nodes_df.rename(columns={" la": "latitude", " lo": "longitude"})

# Run Haversine
codecare_coords = np.radians(codecare_df[["latitude", "longitude"]].values)
mo_nodes_coords = np.radians(mo_nodes_df[["latitude", "longitude"]].values)

dist_matrix = haversine_distances(codecare_coords, mo_nodes_coords)
dist_matrix = dist_matrix * 6371

closest_indices = dist_matrix.argmin(axis=1)
closest_distances = dist_matrix.min(axis=1)

# Set a threshold distance (e.g., 1 km)
threshold_km = 100
mask = closest_distances <= threshold_km

# Merge matching points
df1_matches = codecare_df[mask].reset_index(drop=True)
df2_matches = mo_nodes_df.iloc[closest_indices[mask]].reset_index(drop=True)
merged_matches = pd.concat([df1_matches, df2_matches.add_suffix("_mo_nodes")], axis=1)

# Get non-matching points
df1_nonmatches = codecare_df[~mask]
df2_nonmatches = mo_nodes_df.drop(closest_indices[mask])


# Union: all unique points
final_df = pd.concat(
    [merged_matches, df1_nonmatches, df2_nonmatches], ignore_index=True
)

# Choose which essential columns to keep
COLS_TO_KEEP = [
    "date",
    "state",
    "hospital_name",
    "address",
    "city",
    "hospital_subtype",
    "risk_score",
    "risk_class",
    "# index_mo_nodes",
]

# Adjust and drop any missing values
final_df = final_df[COLS_TO_KEEP]

final_df = final_df.dropna()

final_df["# index_mo_nodes"] = final_df["# index_mo_nodes"].astype(int)

final_df = final_df.rename(
    columns={"# index_mo_nodes": "# index"}
)  # rename index for clarity

final_df.to_csv("../../data/mo_nodes_with_risk.csv")  # export
