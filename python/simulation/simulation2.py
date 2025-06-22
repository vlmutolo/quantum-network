import itertools
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import math
import pandas as pd

# Parameters
n = 50                     # Number of nodes
initial_capacity = 20      # Starting capacity
time_steps = 50            # Total simulation steps
generation_rate = 20      # Capacity increase every 10 steps
swap_rate = 300
consumption_rates = {}
distillation_pairs = 2
edges_with_consumption_non_zero_rate = 50

def generate_recursive_array(D, N=50):
    f = [0] * (N)
    f[0] = 0  # Start with f[0]
    f[1] = D
    for n in range(2, N):
        left = f[math.floor(n / 2)]
        right = f[math.ceil(n / 2)]
        f[n] = D * (f[left] + f[right])
    return f

def create_overlay_on_grid_graph(G):
    rows, cols = 10, 5
    initial_capacity = 1

    # Add all nodes
    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j))

    # List potential wrap-around edges
    edges = []
    for i in range(rows):
        for j in range(cols):
            current = (i, j)
            right = (i, (j + 1) % cols)
            down = ((i + 1) % rows, j)
            edges.append((current, right))
            edges.append((current, down))

    # Shuffle edges for randomness
    random.shuffle(edges)

    # Add edges until graph is connected
    for u, v in edges:
        G.add_edge(u, v, capacity=initial_capacity)
        if nx.is_connected(G):
            # Optionally stop early once connected
            break

    # Add a few more random edges for redundancy, but not all
    extra_edges_to_add = int(0.2 * len(edges))  # Add 20% more edges
    added = 0
    for u, v in edges:
        if not G.has_edge(u, v):
            G.add_edge(u, v, capacity=initial_capacity)
            added += 1
        if added >= extra_edges_to_add:
            break
    return G

def create_graph(n, initial_capacity, graph_type):
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
    if graph_type == "cycle":
        for i in range(n):
            G.add_edge(i, (i + 1) % n, capacity=initial_capacity)
    elif graph_type == "line":
        for i in range(n - 1):
            G.add_edge(i, i + 1, capacity=initial_capacity)
    elif graph_type == "wrap_grid":
        rows = 10
        cols = 5
        for i in range(rows):
            for j in range(cols):
                current = (i, j)
                # Wrap-around neighbors
                right = (i, (j + 1) % cols)
                down = ((i + 1) % rows, j)
            
                # Add edges with wrap-around
                G.add_edge(current, right, capacity=initial_capacity)
                G.add_edge(current, down, capacity=initial_capacity)
    elif graph_type == "wrap_grid_overlay":
        G = create_overlay_on_grid_graph(G)
    elif graph_type == "non_wrap_grid":
        rows = 10
        cols = 5
        for i in range(rows):
            for j in range(cols):
                current = (i, j)
                # Right neighbor (if not on the last column)
                if j + 1 < cols:
                    G.add_edge(current, (i, j + 1), capacity=initial_capacity)
                # Down neighbor (if not on the last row)
                if i + 1 < rows:
                    G.add_edge(current, (i + 1, j), capacity=initial_capacity)
        

    # Select a few edges to have non-zero consumption rates
    nodes = list(range(n))
    all_possible_edges = [(u, v) for u, v in itertools.combinations(nodes, 2)]
    random_edges = random.sample(all_possible_edges, edges_with_consumption_non_zero_rate)

    consumption_rates = {}
    
    for edge in all_possible_edges:
        if edge in random_edges:
            consumption_rates[edge] = random.randint(1, 3)
        else:
            consumption_rates[edge] = 0

    return G, consumption_rates

# Simulation loop
def simulate(G, consumption_rates, time_steps, generation_rate, swap_rate):
    edge_states = {(min(u, v), max(u, v)): G[u][v]['capacity'] for u, v in G.edges()}
    nodes = list(G.nodes())

    swap_count_over_time = []
    corrected_consumption_sum_over_time = []
    consumption_sum_over_time = []
    failed_consumption_count_over_time = []
    successful_consumption_count_over_time = []
    total_bell_pairs_in_graph_over_time = []

    swap_overlay = []
    swap_count = 0
    swap_count_consumption = 0
    failed_consumption_count = 0
    successful_consumption_count = 0
    total_bell_pairs_in_graph = initial_capacity * G.number_of_edges()

    distillation_sum_multiplier = generate_recursive_array(distillation_pairs)
    
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
            total_bell_pairs_in_graph += 1

        elif event_type == "consume":
            u, v = random.sample(nodes, 2)
            edge = (min(u, v), max(u, v))

            if edge in edge_states and edge_states[edge] > distillation_pairs-1:
                edge_states[edge] -= distillation_pairs
                #print(f"[Time {t}] Consumed on {edge} → New count: {edge_states[edge]}")
                path_length = nx.shortest_path_length(G, source=u, target=v)
                swap_count_consumption += distillation_sum_multiplier[path_length - 1] 

                successful_consumption_count+=1
                total_bell_pairs_in_graph = total_bell_pairs_in_graph - distillation_pairs
            else:
                #print(f"failed, no path: {edge}")
                failed_consumption_count+=1 

        elif event_type == "swap":
            x = random.choice(nodes)
            neighbors = get_neighbours(edge_states, x)
                    
            if len(neighbors) < 2:
                continue

            preferable_swaps = get_preferable_swaps(neighbors, edge_states, x)

            if preferable_swaps:
                # Choose the (y, z) with the smallest c(y,z)
                (y, z), _ = min(preferable_swaps, key=lambda item: item[1])
                edge_states[(min(x,y), max(x,y))] -= distillation_pairs
                edge_states[(min(x,z), max(x,z))] -= distillation_pairs
                edge_states[(min(y,z), max(y,z))] = edge_states.get((min(y,z), max(y,z)), 0) + 1
                swap_count += 1
                total_bell_pairs_in_graph = total_bell_pairs_in_graph - 2 * distillation_pairs + 1
                # print(f"[Time {t}] Swap by {x}: ({x},{y}) + ({x},{z}) → ({y},{z}) [Preferable with min count]")

        if (t % 200 == 0):
            swap_count_over_time.append(swap_count)

            correction_factor = 0
            if edge in edge_states and edge_states[edge] > distillation_pairs-1:
                #print(f"[Time {t}] Consumed on {edge} → New count: {edge_states[edge]}")
                path_length = nx.shortest_path_length(G, source=u, target=v)
                correction_factor += path_length - 1

            corrected_consumption_sum_over_time.append(swap_count_consumption + correction_factor)
            consumption_sum_over_time.append(swap_count_consumption)
            failed_consumption_count_over_time.append(failed_consumption_count)
            successful_consumption_count_over_time.append(successful_consumption_count)
            total_bell_pairs_in_graph_over_time.append(total_bell_pairs_in_graph)

            if (swap_count_consumption > 0):
                swap_overlay.append(swap_count/swap_count_consumption)
                if (is_stabilized(swap_overlay)):
                    print(t)
                    break
    
    print(swap_count, swap_count_consumption, successful_consumption_count, failed_consumption_count)
    return (swap_count_over_time,
consumption_sum_over_time,
corrected_consumption_sum_over_time,
failed_consumption_count_over_time,
successful_consumption_count_over_time,
total_bell_pairs_in_graph_over_time)

def get_neighbours(edge_states, x):
    neighbors = set()
    for u, v in edge_states:
        if u == x:
            if (edge_states[u, v] != 0):
                neighbors.add(v)
        elif v == x:
            if (edge_states[u, v] != 0):
               neighbors.add(u)
    return neighbors

def get_preferable_swaps(neighbors, edge_states, x):
    preferable_swaps = []
            
    for y, z in itertools.combinations(neighbors, 2):
        c_xy = edge_states.get((min(x, y), max(x, y)), 0)
        c_xz = edge_states.get((min(x, z), max(x, z)), 0)
        c_yz = edge_states.get((min(y, z), max(y, z) ), 0)
        #print(x, y, z, c_xy, c_xz, c_yz)
                
        # Check if this is a preferable swap
        if c_xy > c_yz + distillation_pairs and c_xz > c_yz + distillation_pairs:
            preferable_swaps.append(((y, z), c_yz))
    return preferable_swaps



def main():
    graph_types = ["cycle", "wrap_grid_overlay"]

    for graph in graph_types: 
        G, consumption_rates = create_graph(n, generation_rate, "cycle")
    
        print("Graph edges with capacities:")
        for u, v, attr in G.edges(data=True):
            print(f"Edge ({u} -> {v}): Capacity = {attr['capacity']}")
    
    (swap_count_over_time,
     consumption_sum_over_time,
corrected_consumption_sum_over_time,
     failed_consumption_count_over_time,
     successful_consumption_count_over_time,
     total_bell_pairs_in_graph_over_time) = simulate(
        G,
        consumption_rates,
        time_steps=2000000,
        generation_rate=generation_rate,
        swap_rate=swap_rate)
    plot_metrics(swap_count_over_time,
                 consumption_sum_over_time,
                 corrected_consumption_sum_over_time,
                 failed_consumption_count_over_time,
                 successful_consumption_count_over_time,
                 total_bell_pairs_in_graph_over_time)
       

def is_stabilized(arr, num_last=10, tolerance=0.05):
    if len(arr) < max(2, num_last):
        print("Not enough data points to check stabilization.")
        return False
    last_values = arr[-num_last:]
    diffs = np.abs(np.diff(last_values))
    max_diff = np.max(diffs)
    return max_diff < tolerance

def plot_metrics(swap_counts,
                 consumption_costs,
                 corrected_consumption_sum_over_time,
                 failed_consumption_count_over_time,
                 successful_consumption_count_over_time,
                 total_bell_pairs_in_graph_over_time):
    time_steps = list(range(1, len(swap_counts) + 1))
    successes = successful_consumption_count_over_time
    failures = failed_consumption_count_over_time

    swap_count_rate = [s / c if (c) > 0 else 0 for s, c in zip(swap_counts, consumption_costs)]

    k = is_stabilized(swap_count_rate)
    print(k)
    corrected_swap_count_rate = [s / c if (c) > 0 else 0 for s, c in zip(swap_counts, corrected_consumption_sum_over_time)]

    failure_rate = [f / (s + f) if (s + f) > 0 else 0 for s, f in zip(successes, failures)]

    data_dict = {
        "time_steps": time_steps,
        "total_swap_count_over_time": swap_counts,
        "optimal_direct_graph_path_sum": consumption_costs,
        "optimal_direct_graph_path_sum_corrected": corrected_consumption_sum_over_time,
        "successful_consumptions": successes,
        "failed_consumptions": failures,
        "total_bell_pairs_over_time": total_bell_pairs_in_graph_over_time,
        "failure_rate": failure_rate
    }

    df = pd.DataFrame(data_dict)
    df.to_csv("metrics_output.csv", index=False)


    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    
    # --- First plot: Swap-related metrics ---
    axes[0].plot(time_steps, swap_counts, label='Swap overhead', marker='o')
    axes[0].set_xlabel("1 Time Step = 200 ticks")
    axes[0].set_ylabel("Swap overhead")
    axes[0].set_title(f"{n} nodes swap overhead")
    axes[0].legend()
    axes[0].set_yscale('log') 
    axes[0].grid(True)
    
    # --- Second plot: Consumption success/failure ---
    axes[1].plot(time_steps, failure_rate, label='failure rate of consumption request', marker='x')
    axes[1].set_xlabel("1 Time Step = 200 ticks")
    axes[1].set_ylabel("failure rate")
    axes[1].set_yscale('log') 
    axes[1].set_title("Successful vs Failed Consumption Over Time")
    axes[1].legend()
    axes[1].grid(True)

     # --- Second plot: Consumption success/failure ---
    axes[2].plot(time_steps[50:], total_bell_pairs_in_graph_over_time[50:], label='total bell pairs in the system', marker='x')
    axes[2].set_xlabel("1 Time Step = 200 ticks")
    axes[2].set_ylabel("Count")
    axes[2].set_title("total bell pairs in the system")
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()