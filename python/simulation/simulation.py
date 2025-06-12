import itertools
import networkx as nx
import matplotlib.pyplot as plt
import random

# Parameters
n = 5                      # Number of nodes
initial_capacity = 10      # Starting capacity
time_steps = 50            # Total simulation steps
generation_rate = 10     # Capacity increase every 10 steps
swap_rate = 3 
consumption_rates = {}
edge_with_consumption_non_zero_rate = 4
swap_count = 0
swap_count_consumption = 0

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
    edge_states = {(min(u, v), max(u, v)): G[u][v]['capacity'] for u, v in G.edges()}
    nodes = list(G.nodes())
    swap_count = 0
    swap_count_consumption = 0
    
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
            u, v = edge
            edge_states[(min(u,v), max(u, v))] += 1
            print(f"[Time {t}] Generated on {edge} → New count: {edge_states[(min(u,v), max(u, v))]}")

        elif event_type == "consume":
            u, v = random.sample(nodes, 2)
            edge = (min(u, v), max(u, v))

            if edge in edge_states and edge_states[edge] > 0:
                edge_states[edge] -= 1
                print(f"[Time {t}] Consumed on {edge} → New count: {edge_states[(min(u,v), max(u, v))]}")
            else:
                try:
                    path = nx.shortest_path(G, source=u, target=v)
                    # Try to consume along the full path
                    can_consume = all(
                        edge_states.get((min(path[i], path[i+1]), max(path[i], path[i+1])), 0) > 0
                        for i in range(len(path) - 1)
                        )
                    if can_consume:
                        for i in range(len(path) - 1):
                            a, b = path[i], path[i+1]
                            swap_count_consumption+=1
                            edge_states[(min(a, b), max(a, b))] -= 1
                            print(f"[Time {t}] Indirect consumption from {u} to {v} via path {path}")
                    else:
                        print(f"[Time {t}] Cannot consume: not enough Bell-pairs along path {path}")
                except nx.NetworkXNoPath:
                    print(f"[Time {t}] No path exists from {u} to {v} for indirect consumption")

        elif event_type == "swap":
            x = random.choice(nodes)
            neighbors = set(G.predecessors(x)).union(set(G.successors(x)))
            print(neighbors)
            if len(neighbors) < 2:
                continue

            preferable_swaps = []
            
            for y, z in itertools.combinations(neighbors, 2):
                c_xy = edge_states.get((min(x, y), max(x, y)), 0)
                c_xz = edge_states.get((min(x, z), max(x,z)), 0)
                c_yz = edge_states.get((min(y, z), max(y,z) ), 0)
                print(c_xy, c_xz, c_yz)
                
                # Check if this is a preferable swap
                if c_xy > c_yz + 1 and c_xz > c_yz + 1:
                    preferable_swaps.append(((y, z), c_yz))


            if preferable_swaps:
                # Choose the (y, z) with the smallest c(y,z)
                (y, z), _ = min(preferable_swaps, key=lambda item: item[1])
                edge_states[(min(x,y), max(x,y))] -= 1
                edge_states[(min(x,z), max(x,z))] -= 1
                edge_states[(min(y,z), max(y,z))] = edge_states.get((min(y,z), max(y,z)), 0) + 1
                swap_count +=1
                print(f"[Time {t}] Swap by {x}: ({x},{y}) + ({x},{z}) → ({y},{z}) [Preferable with min count]")
            else:
                print(f"[Time {t}] {x} found no preferable swap")
    
    print(swap_count, swap_count_consumption)

def main():
    G, consumption_rates = create_graph(n, generation_rate)

    
    print("Graph edges with capacities:")
    for u, v, attr in G.edges(data=True):
        print(f"Edge ({u} -> {v}): Capacity = {attr['capacity']}")
    
    print("\nConsumption rates:")
    for edge, rate in consumption_rates.items():
        print(f"{edge}: Consumption Rate = {rate}")

    simulate(G, consumption_rates, time_steps=50, generation_rate=10, swap_rate=2)

    print(swap_count, swap_count_consumption)

if __name__ == "__main__":
    main()