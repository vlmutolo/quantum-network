import itertools
import networkx as nx
import matplotlib.pyplot as plt
import random

# Parameters
n = 10                     # Number of nodes
initial_capacity = 10      # Starting capacity
time_steps = 50            # Total simulation steps
generation_rate = 10     # Capacity increase every 10 steps
swap_rate = 3 
consumption_rates = {}
edges_with_consumption_non_zero_rate = 15
swap_count = 0
swap_count_consumption = 0

def create_graph(n, initial_capacity):
    """
    Create a cycle graph with specified parameters.
    
    Each node is connected to the next in a cycle.
    Every 10 time steps, the capacity on each edge increases by `generation_rate`.
    A subset of edges is randomly selected to have non-zero consumption rates.

    Returns:
        G (nx.DiGraph): Graph with initialized capacities.
        consumption_rates (dict): Mapping from edge to consumption rate.
    """
    G = nx.Graph()
    for i in range(n):
        G.add_edge(i, (i + 1) % n, capacity=initial_capacity)

    # Select a few edges to have non-zero consumption rates
    nodes = list(range(n))
    all_possible_edges = [(u, v) for u, v in itertools.combinations(nodes, 2)]
    random_edges = random.sample(all_possible_edges, edges_with_consumption_non_zero_rate)

    consumption_rates = {}
    
    for edge in all_possible_edges:
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
def simulate(G, consumption_rates, time_steps, generation_rate, swap_rate):
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
            u, v = random.choice(list(G.edges()))
            edge = (min(u, v), max(u, v))

            edge_states[edge] += 1
            print(f"[Time {t}] Generated on {edge} → New count: {edge_states[edge]}")

        elif event_type == "consume":
            u, v = random.sample(nodes, 2)
            edge = (min(u, v), max(u, v))

            if edge in edge_states and edge_states[edge] > 0:
                edge_states[edge] -= 1
                print(f"[Time {t}] Consumed on {edge} → New count: {edge_states[edge]}")

        elif event_type == "swap":
            x = random.choice(nodes)
            neighbors = list(G.neighbors(x))
            print(neighbors)
            if len(neighbors) < 2:
                continue

            preferable_swaps = []
            
            for y, z in itertools.combinations(neighbors, 2):
                c_xy = edge_states.get((min(x, y), max(x, y)), 0)
                c_xz = edge_states.get((min(x, z), max(x, z)), 0)
                c_yz = edge_states.get((min(y, z), max(y, z) ), 0)
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

    simulate(G, consumption_rates, time_steps=10000, generation_rate=10, swap_rate=4)

    print(swap_count, swap_count_consumption)

if __name__ == "__main__":
    main()