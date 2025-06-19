import itertools
import networkx as nx
import matplotlib.pyplot as plt
import random
import math

# Parameters
n = 50                     # Number of nodes
initial_capacity = 20      # Starting capacity
time_steps = 50            # Total simulation steps
generation_rate = 20      # Capacity increase every 10 steps
swap_rate = 200
decoherence_rate = 0
consumption_rates = {}
distillation_pairs = 5
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
    graph_type = "wrap_grid"
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
    elif graph_type == "non_wrap_grid":
        rows = 10
        cols = 5
        for i in range(rows):
            for j in range(cols):
                current = (i, j)
                # Right neighbor (if not on the last column)
                if j + 1 < cols:
                    G.add_edge(current, (i, j + 1), capacity=capacity)
                # Down neighbor (if not on the last row)
                if i + 1 < rows:
                    G.add_edge(current, (i + 1, j), capacity=capacity)
        

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

    swap_count = 0
    swap_count_consumption = 0
    failed_consumption_count = 0
    successful_consumption_count = 0
    total_bell_pairs_in_graph = initial_capacity * G.number_of_edges()

    

    distillation_sum_multiplier = generate_recursive_array(distillation_pairs)
    
    for t in range(1, time_steps + 1):
        event_type = random.choices(
            ["generate", "consume", "swap", "decoherence"],
            weights=[sum(generation_rate for _ in G.edges()),
                     sum(consumption_rates.values()),
                     swap_rate * len(nodes),
                     sum(decoherence_rate for _ in G.edges())],
            k=1
        )[0]

        if event_type == "generate":
            u, v = random.choice(list(G.edges()))
            edge = (min(u, v), max(u, v))
            edge_states[edge] += 1
            total_bell_pairs_in_graph += 1

        elif event_type == "decoherence":
            valid_edges = [edge for edge, cap in edge_states.items() if cap > 1]
            if (valid_edges):
              random_edge = random.choice(valid_edges)
              edge_states[random_edge] -= 1
              total_bell_pairs_in_graph -= 1

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
    G, consumption_rates = create_graph(n, generation_rate)
    
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
        time_steps=200000,
        generation_rate=generation_rate,
        swap_rate=swap_rate)
    plot_metrics(swap_count_over_time,
                 consumption_sum_over_time,
                 corrected_consumption_sum_over_time,
                 failed_consumption_count_over_time,
                 successful_consumption_count_over_time,
                 total_bell_pairs_in_graph_over_time)
    
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
    corrected_swap_count_rate = [s / c if (c) > 0 else 0 for s, c in zip(swap_counts, corrected_consumption_sum_over_time)]

    failure_rate = [f / (s + f) if (s + f) > 0 else 0 for s, f in zip(successes, failures)]

    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    
    # --- First plot: Swap-related metrics ---
    axes[0].plot(time_steps[50:], swap_count_rate[50:], label='Swap overhead', marker='o')
    axes[0].plot(time_steps[50:], corrected_swap_count_rate[50:], label='Corrected Swap Overhead', marker='x')
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