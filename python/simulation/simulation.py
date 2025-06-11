import networkx as nx
import matplotlib.pyplot as plt
import random

# Parameters
n = 5                      # Number of nodes
initial_capacity = 10      # Starting capacity
time_steps = 50            # Total simulation steps
generation_rate = 10     # Capacity increase every 10 steps
swap_rate = 2 
consumption_rates = {}
edge_with_consumption_non_zero_rate = 4

def create_graph(n, initial_capacity):
    """
    Create a directed cycle graph with specified parameters.
    
    Each node is connected to the next in a cycle.
    Every 10 time steps, the capacity on each edge increases by `generation_rate`.
    A subset of edges is randomly selected to have non-zero consumption rates.

    Returns:
        G (nx.DiGraph): Directed graph with initialized capacities.
        consumption_rates (dict): Mapping from edge to consumption rate.
    """
    # Create directed cycle graph
    G = nx.DiGraph()
    for i in range(n):
        G.add_edge(i, (i + 1) % n, capacity=initial_capacity)

    # Select a few edges to have non-zero consumption rates
    all_edges = list(G.edges())
    random_edges = random.sample(all_edges, edge_with_consumption_non_zero_rate)

    consumption_rates = {}
    
    for edge in all_edges:
        if edge in random_edges:
            consumption_rates[edge] = random.randint(1, 5)
        else:
            consumption_rates[edge] = 0

    return G, consumption_rates

# Visualization function
def draw_graph(G, step):
    plt.figure(figsize=(6, 4))
    pos = nx.circular_layout(G)
    edge_labels = nx.get_edge_attributes(G, 'capacity')
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title(f"Step {step}: Directed Cycle Graph with Capacities")
    plt.show()

# Simulation loop
def simulate(G, consumption_rates, time_steps=50, generation_rate=10, swap_rate=2):
    edge_states = {(u, v): G[u][v]['capacity'] for u, v in G.edges()}
    nodes = list(G.nodes())
    
    for t in range(1, time_steps + 1):
        event_type = random.choices(
            ["generate", "consume", "swap"],
            weights=[sum(generation_rate for _ in G.edges()),
                     sum(consumption_rates.values()),
                     swap_rate * len(nodes)],
            k=1
        )[0]

        if event_type == "generate":
            edge = random.choice(list(G.edges()))
            edge_states[edge] += 1
            print(f"[Time {t}] Generated on {edge} → New count: {edge_states[edge]}")

        elif event_type == "consume":
            edge = random.choices(list(G.edges()), weights=[consumption_rates[e] for e in G.edges()])[0]
            if edge_states[edge] > 0:
                edge_states[edge] -= 1
                print(f"[Time {t}] Consumed on {edge} → New count: {edge_states[edge]}")
            else:
                print(f"[Time {t}] Tried to consume on {edge} but count is 0")

        elif event_type == "swap":
            x = random.choice(nodes)
            neighbors = list(G.successors(x))
            if len(neighbors) < 2:
                continue

            y, z = random.sample(neighbors, 2)
            c_xy = edge_states.get((x, y), 0)
            c_xz = edge_states.get((x, z), 0)
            c_yz = edge_states.get((y, z), 0)

            if c_xy > c_yz + 1 and c_xz > c_yz + 1:
                edge_states[(x, y)] -= 1
                edge_states[(x, z)] -= 1
                edge_states[(y, z)] = edge_states.get((y, z), 0) + 1
                print(f"[Time {t}] Swap by {x}: ({x},{y}) + ({x},{z}) → ({y},{z})")
            else:
                print(f"[Time {t}] {x} attempted swap ({y},{z}) but not preferable")

        # Capacity refresh every 10 steps
        if t % 10 == 0:
            for edge in G.edges():
                edge_states[edge] += generation_rate
            print(f"[Time {t}] Capacity boost: +{generation_rate} on all edges")


def main():
    G, consumption_rates = create_graph(n, generation_rate)

    for step in range(0, time_steps + 1):
        if step % 10 == 0 and step != 0:
            for u, v in G.edges():
                G[u][v]['capacity'] += generation_rate
    
    print("Graph edges with capacities:")
    for u, v, attr in G.edges(data=True):
        print(f"Edge ({u} -> {v}): Capacity = {attr['capacity']}")
    
    print("\nConsumption rates:")
    for edge, rate in consumption_rates.items():
        print(f"{edge}: Consumption Rate = {rate}")

    simulate(G, consumption_rates, time_steps=50, generation_rate=10, swap_rate=2)

if __name__ == "__main__":
    main()