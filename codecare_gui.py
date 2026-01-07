# Import CustomTkinter for GUI, folium and webbrowserfor the map, and the functions from the routing engine
import customtkinter as ctk
import folium
import webbrowser
from codecare.routing_engine import read_data, construct_graph, astar_many_nodes

# Load in the graph
nodes, edges = read_data(path1="data/mo_data.csv", path2="data/mo_edges.csv")
G = construct_graph(nodes, edges)

# Create a lookup table to get IDs from hospital names
hospital_lookup = {}

for node, data in G.nodes(data=True):
    name = data.get("hospital_name")
    if not name:
        continue

    hospital_lookup.setdefault(name, []).append(node)


# This function is meant to draw the edges on the map
def draw_route_on_map(m, path, color="green"):
    if len(path) < 2:
        print("Path too short to draw!")
        return

    coords = [
        (float(G.nodes[n]["longitude"]), float(G.nodes[n]["latitude"])) for n in path
    ]
    folium.PolyLine(coords, color=color, weight=5, opacity=0.8, tooltip="Route").add_to(
        m
    )
    m.fit_bounds(coords)


# Utilizes the A* algorithm from the routing engine module and generates routes
def generate_routes(hospital_list):
    # Sort the hospital list based on risk score
    hospital_list.sort(
        key=lambda name: G.nodes[hospital_lookup[name][0]]["risk_score"],
        reverse=True,
    )

    # Create a list for the multiple A* algortihm
    indices_list = []
    for hospital in hospital_list:
        indices_list.append(hospital_lookup[hospital][0])

    indices_list = list(zip(indices_list, indices_list[1:]))

    # Run the multuple A* algorithm
    path, cumulative_cost = astar_many_nodes(G, indices_list)
    cumulative_cost = cumulative_cost / 3600  # convert seconds into hours
    cumulative_cost_label.configure(text=f"Total time: {cumulative_cost:.2f} hours")

    # Create Folium map
    route_map = folium.Map(location=[38.5, -92.5], zoom_start=7)

    # Draw hospital markers
    for name in hospital_list:
        node_id = hospital_lookup[name][0]
        print(node_id)
        data = G.nodes[node_id]
        folium.Marker(
            [data["longitude"], data["latitude"]],
            popup=f"<b>{name} | Risk Class is {int(data['risk_class'])} | Risk Score of {data['risk_score']:.2f}</b>",
            icon=folium.Icon(color="red"),
        ).add_to(route_map)

    # Draw route lines
    draw_route_on_map(m=route_map, path=path, color="green")

    # Save and open map
    route_map.save("hospital_routes.html")
    webbrowser.open("hospital_routes.html")


# Create a list of hospitals
hospitals = nodes["hospital_name"].dropna().tolist()

# This list will be used for the hospitals that will be used in the path calculations
hospital_nodes = []

# Set the inital appearance of the video
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# Create the window with the title
app = ctk.CTk()
app.geometry("500x300")
app.title("Missouri Hospital Explorer: CodeCare")

# Display a title text on the window
title = ctk.CTkLabel(
    app, text="CodeCare: Select a Hospital in Missouri", font=("Roboto", 20)
)
title.pack(pady=20)

# Multi-select box on the GUI
combobox = ctk.CTkComboBox(
    master=app,
    values=hospitals,
    width=200,
    corner_radius=5,
)
combobox.pack(pady=20)

# Button for adding the hospitals to the list
add_button = ctk.CTkButton(
    app,
    text="Add to List",
    width=100,
    command=lambda: (
        hospital_nodes.append(combobox.get())
        if combobox.get() not in hospital_nodes
        else None
    ),
)
add_button.pack(pady=15)

# This button, when clicked on, will generate the paths and create the webpage
generate_button = ctk.CTkButton(
    app, text="Generate Map", width=100, command=lambda: generate_routes(hospital_nodes)
)
generate_button.pack(pady=15)

# This label will tell the total time for the trip
cumulative_cost_label = ctk.CTkLabel(
    app, text="Total time: 0 hours", font=("Roboto", 16)
)
cumulative_cost_label.pack(pady=10)

# Run the app!
app.mainloop()
