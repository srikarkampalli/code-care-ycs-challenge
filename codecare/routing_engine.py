import pandas as pd  # for reading the csv
import networkx as nx  # for rendering the graph
from sklearn.metrics.pairwise import haversine_distances  # heuristic
import numpy as np  # radians in haversine


# reads the nodes and edges from two paths
def read_data(path1, path2):
    """
    This function, based on two file paths, reads them as DataFrames and exports them.
    """
    nodes = pd.read_csv(path1)
    edges = pd.read_csv(path2)

    # Assuming the car goes at 50 MPH constantly, this returns hours
    edges[" travel_time"] = (
        (edges[" distance"] / 1609) / 50
    ) * 3600  # changes travel time by converting meters into miles

    print("Data loaded!")
    return nodes, edges


def construct_graph(nodes: pd.DataFrame, edges: pd.DataFrame):
    """
    This function will create the NetworkX graph from the node and edge DataFrames.
    """
    # Create graph from edges
    G = nx.from_pandas_edgelist(
        edges,
        source="# source",
        target=" target",
        edge_attr=[" travel_time", " distance"],
        create_using=nx.Graph(),
    )

    # Add node attributes, handle duplicates by grouping (keep first)
    nodes_unique = nodes.drop_duplicates(subset="# index")
    attr_dict = nodes_unique.set_index("# index")[
        [
            "risk_score",
            "latitude",
            "longitude",
            "hospital_name",
            "address",
            "city",
            "hospital_subtype",
            "risk_class",
        ]
    ].to_dict("index")
    nx.set_node_attributes(G, attr_dict)

    print("Graph constructed!")

    return G


def astar_shortest_path(G, start_node, end_node, weight=" travel_time"):
    """
    Using the NetworkX A* implementation (which is easier to implement that your own), find the shortest path given a graph.
    """
    # Precompute lat/lon in radians for all nodes
    lat_lon_rad = {
        n: np.radians([data["latitude"], data["longitude"]])
        for n, data in G.nodes(data=True)
    }

    # Heuristic function for A*
    def heuristic(u, v):
        coord_u = lat_lon_rad[u].reshape(1, -1)
        coord_v = lat_lon_rad[v].reshape(1, -1)
        # haversine_distances returns radians; multiply by Earth radius
        return haversine_distances(coord_u, coord_v)[0][0] * 6371  # km

    # Run A*
    path = nx.astar_path(G, start_node, end_node, heuristic=heuristic, weight=weight)

    # Calculate total cost
    total_cost = sum(G[u][v][weight] for u, v in zip(path[:-1], path[1:]))

    print("Path and total cost calculated!")

    return path, total_cost


# Based off of the individual A* algorithm
def astar_many_nodes(G, pairs):
    """
    This function builds off of the initial A* function by adapting it to multiple nodes.
    """
    # Create of cumulative path list
    all_paths = []

    # Same thing but with the time
    cumulative_cost = 0

    # Runs A* individually for each and updates local vars
    for start, end in pairs:
        path, cost = astar_shortest_path(G, start, end)
        all_paths.extend(path)
        cumulative_cost += cost

    return all_paths, cumulative_cost
