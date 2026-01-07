import pandas as pd  # csv reading

# Merge risk with MO nodes
nodes_with_risk = pd.read_csv("../../data/mo_nodes_with_risk.csv")
nodes = pd.read_csv("../../data/mo_nodes.csv")

merged_df = nodes.merge(nodes_with_risk, on="# index", how="left")
merged_df["risk_score"] = merged_df["risk_score"].fillna(0)

merged_df.to_csv("../../data/mo_data.csv")
