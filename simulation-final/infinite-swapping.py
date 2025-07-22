from collections import deque
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import math
import pandas as pd
import multiprocessing
from multiprocessing import Pool, cpu_count


# Parameters
n = 25
initial_capacity = 20
generation_rate = 40
edges_with_consumption_non_zero_rate = 35

def norm_edge(u, v):
    return (u, v) if u <= v else (v, u)

def generate_recursive_array(D, N=50):
    f = [0] * N
    if N > 1:
        f[1] = D
    for n in range(2, N):
        f[n] = D * (f[n // 2] + f[(n + 1) // 2])
    return f

def generate_worst_recursive_array(D, N=50):
    f = [0] * N
    if N > 1:
        f[1] = D
    for n in range(2, N):
        f[n] = D * (f[n-1] + f[1])
    return f

def create_overlay_on_non_wrap_grid_graph(G, initial_capacity):
    rows, cols = 5, 5

    # Add all nodes
    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j))

    # List potential non-wrap-around edges (right and down only if in bounds)
    edges = []
    for i in range(rows):
        for j in range(cols):
            current = (i, j)
            if j + 1 < cols:  # right neighbor
                right = (i, j + 1)
                edges.append((current, right))
            if i + 1 < rows:  # down neighbor
                down = (i + 1, j)
                edges.append((current, down))

    # Shuffle edges for randomness
    random.shuffle(edges)

    # Add edges until graph is connected
    for u, v in edges:
        if not G.has_edge(u, v):
            G.add_edge(u, v, capacity=initial_capacity)
        if nx.is_connected(G):
            break

    # Add a few more random edges for redundancy, but not all
    extra_edges_to_add = int(0 * len(edges))  # Add 20% more edges
    added = 0
    for u, v in edges:
        if not G.has_edge(u, v):
            G.add_edge(u, v, capacity=initial_capacity)
            added += 1
        if added >= extra_edges_to_add:
            break

    return G

def create_overlay_on_grid_graph(G, initial_capacity):
    rows, cols = 5, 5

    # Add all nodes to the graph
    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j))

    # Create all potential wraparound grid edges
    edges = []
    for i in range(rows):
        for j in range(cols):
            current = (i, j)
            right = (i, (j + 1) % cols)
            down = ((i + 1) % rows, j)
            edges.append((current, right))
            edges.append((current, down))

    # Shuffle edges to randomize the order of addition
    random.shuffle(edges)

    # Add edges until the graph becomes connected
    for u, v in edges:
        if not G.has_edge(u, v):
            G.add_edge(u, v, capacity=initial_capacity)
        if nx.is_connected(G):
            break  # Stop early once connected

    # Add a few extra edges to increase connectivity (e.g., 20% more)
    remaining_edges = [e for e in edges if not G.has_edge(*e)]
    extra_edges_to_add = int(0 * len(edges))  # 20% more edges
    random.shuffle(remaining_edges)

    for u, v in remaining_edges[:extra_edges_to_add]:
        G.add_edge(u, v, capacity=initial_capacity)

    return G

def create_graph(n, initial_capacity, graph_type):
    G = nx.Graph()
    if graph_type == "cycle":
        for i in range(n):
            G.add_edge(i, (i + 1) % n, capacity=initial_capacity)
        pos =  nx.circular_layout(G)

    elif graph_type == "wrap_grid_overlay":
        G = create_overlay_on_grid_graph(G, initial_capacity)
        pos = {(i, j): (j, -i) for i in range(5) for j in range(5)}  # y = -i to plot top-to-bottom

    # Draw graph
    plt.figure(figsize=(8, 12))
    nx.draw(G, pos, with_labels=True, node_size=300,node_color='lightblue',edge_color='gray',font_size=8)

    plt.title("Random Overlay on 10x5 Grid Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    nodes = [(i, j) for i in range(5) for j in range(5)] if "grid" in graph_type else list(range(n))
    all_possible_edges = [norm_edge(u, v) for u, v in itertools.combinations(nodes, 2)]
    random_edges = random.sample(all_possible_edges, edges_with_consumption_non_zero_rate)
    consumption_rates = {edge: random.randint(1, 6) for edge in random_edges}

    return G, consumption_rates

def is_stabilized_overlay(arr, num_last=15, tolerance=0.2, distillation_pairs = 1):
    if (distillation_pairs == 1):
        tolerance = 0.1
    elif (distillation_pairs <= 2):
        num_last = 8
        tolerance = 0.7
    elif (distillation_pairs <= 6):
        num_last = 8
        tolerance = 1

    if len(arr) < max(2, num_last):
        return False
    
    diffs = np.abs(np.diff(arr[-num_last:]))

    return np.max(diffs) < tolerance

def is_stabilized_efficiency(arr, num_last=15, tolerance=0.5, distillation_pairs = 1):
    if (distillation_pairs == 1):
        tolerance = 0.1
    elif (distillation_pairs <= 2):
        tolerance = 3
    elif (distillation_pairs <= 3):
        num_last = 7
        tolerance = 150
    elif (distillation_pairs <= 4):
        num_last = 7
        tolerance = 1800
    elif (distillation_pairs <= 5):
        num_last = 7
        tolerance = 18000
    if len(arr) < max(2, num_last):
        return False
    diffs = np.abs(np.diff(arr[-num_last:]))

    return np.max(diffs) < tolerance

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

def get_preferable_swaps(neighbors, edge_states, x, distillation_pairs):
    swaps = []
    for y, z in itertools.combinations(neighbors, 2):
        c_xy = edge_states.get(norm_edge(x, y), 0)
        c_xz = edge_states.get(norm_edge(x, z), 0)
        c_yz = edge_states.get(norm_edge(y, z), 0)

        if (c_xy >= distillation_pairs and
         c_xz >= distillation_pairs and
         c_xy > c_yz + distillation_pairs and 
         c_xz > c_yz + distillation_pairs):
            swaps.append(((y, z), c_yz))
    return swaps

def attempt_preferable_swap(x, neighbors, edge_states, distillation_pairs, swap_count, total_bell_pairs):
    swaps = get_preferable_swaps(neighbors, edge_states, x, distillation_pairs)
    if not swaps:
        return False, swap_count, total_bell_pairs
    
    swaps_sorted = sorted(swaps, key=lambda item: item[1])
    min_c_yz = swaps_sorted[0][1]
    changed = False
    delta = 2
    swap_made = 0

    for (y, z), c_yz in swaps_sorted:
        if c_yz > min_c_yz + delta:
            break  # Avoid overswapping

        if swap_made >= 1:
            break  # limit to 2 swaps

        edge_xy, edge_xz, edge_yz = norm_edge(x, y), norm_edge(x, z), norm_edge(y, z)

        if edge_states[edge_xy] >= distillation_pairs and edge_states[edge_xz] >= distillation_pairs:
            edge_states[edge_xy] -= distillation_pairs
            edge_states[edge_xz] -= distillation_pairs
            edge_states[edge_yz] = edge_states.get(edge_yz, 0) + 1

            total_bell_pairs -= 2 * distillation_pairs
            total_bell_pairs += 1
            swap_count += 1
            changed = True
    
    #for (y, z), _ in sorted(swaps, key=lambda item: item[1]):
    #    edge_xy, edge_xz, edge_yz = norm_edge(x, y), norm_edge(x, z), norm_edge(y, z)

    #    if edge_states[edge_xy] >= distillation_pairs and edge_states[edge_xz] >= distillation_pairs:
    #        edge_states[edge_xy] -= distillation_pairs
    #        edge_states[edge_xz] -= distillation_pairs
    #        edge_states[edge_yz] = edge_states.get(edge_yz, 0) + 1

    #        return True, swap_count + 1, total_bell_pairs - 2 * distillation_pairs + 1
        
    return changed, swap_count, total_bell_pairs

def run_simulation(args):
    G, consumption_rates, time_steps, generation_rate, swap_rate, distillation_pairs, infinite_swapping_enabled = args
    return simulate_consumption_sequence(G.copy(), consumption_rates.copy(), time_steps, generation_rate, swap_rate, distillation_pairs)


def simulate_consumption_sequence(G, consumption_rates, time_steps, generation_rate, swap_rate, distillation_pairs):
    edge_states = {norm_edge(u, v): G[u][v]['capacity'] for u, v in G.edges()}
    nodes = list(G.nodes())

    distillation_sum_multiplier = generate_recursive_array(distillation_pairs)
    distillation_sum_multiplier_worst = generate_worst_recursive_array(distillation_pairs)
    consumption_edges = list(consumption_rates.keys())
    consumption_weights = list(consumption_rates.values())
    total_swap_count = failed = succeeded = 0
    total_bell_pairs = initial_capacity * G.number_of_edges()
    optimal_swap_count = 0
    worst_optimal_swap_count = 0
    poisson_mean = 3

    swap_overlay = []
    swap_efficiency = []
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))

    consumption_queue = deque()
    for i in range(1, 50000):
        u, v = random.choices(consumption_edges, weights=consumption_weights)[0]
        edge = norm_edge(u, v)
        consumption_queue.append(edge)

    event = "generate"
    for t in range(1, time_steps + 1):
        if event == "generate":
            edge = norm_edge(*random.choice(list(G.edges())))
            if (distillation_pairs == 1):
               bell_pairs_generated = np.random.poisson(10)
            elif (distillation_pairs <= 2): 
                bell_pairs_generated = np.random.poisson(30)
            elif (distillation_pairs <= 3): 
                bell_pairs_generated = np.random.poisson(200)
            elif (distillation_pairs <= 4): 
                bell_pairs_generated = np.random.poisson(650)
            elif (distillation_pairs <= 5): 
                bell_pairs_generated = np.random.poisson(850)
            edge_states[edge] += bell_pairs_generated
            total_bell_pairs += bell_pairs_generated

            while True:
                changed = False
                for x in random.sample(nodes, len(nodes)):
                    neighbors = get_neighbours(edge_states, x)
                    local_changed, total_swap_count, total_bell_pairs = attempt_preferable_swap(
                        x, neighbors, edge_states, distillation_pairs, total_swap_count, total_bell_pairs)
                    if local_changed:
                        changed = True
                if not changed:
                        break  # No swaps occurred in this pass — done

        while consumption_queue:
            if len(consumption_queue) == 1:
                # Preemptively refill before consuming the last item
                for _ in range(50000):
                    u_new, v_new = random.choices(consumption_edges, weights=consumption_weights)[0]
                    consumption_queue.append((u_new, v_new))

            u, v = consumption_queue[0]  # Peek front
            edge = norm_edge(u, v)
            #print(u, v, edge, edge_states.get(edge, 0), shortest_paths[u][v])
            if edge_states.get(edge, 0) >= distillation_pairs:
                consumption_queue.popleft()  # Only remove if we can consume
                edge_states[edge] -= distillation_pairs
                path_length = shortest_paths[u][v]
                optimal_swap_count += distillation_sum_multiplier[path_length - 1]
                worst_optimal_swap_count += distillation_sum_multiplier_worst[path_length - 1]
                succeeded += 1
                total_bell_pairs -= distillation_pairs
            else:
                break 

        if t % 100 == 0:
            if (t % 1000 == 0):
                print (t, swap_overlay[-7:], swap_efficiency[-7:], succeeded)
            if optimal_swap_count > 0 and succeeded >0 :
                swap_overlay.append(total_swap_count / optimal_swap_count)
                swap_efficiency.append(total_swap_count/succeeded)
                if is_stabilized_overlay(swap_overlay, 12, 0.5, distillation_pairs) and is_stabilized_efficiency(swap_efficiency, 12, 0.5, distillation_pairs):
                    print(f"Result Stabilised, at swap overhead: {swap_overlay[-1]} and swap efficiency: {swap_efficiency[-1]}")
                    break
                
    return total_swap_count, optimal_swap_count, worst_optimal_swap_count, succeeded, total_bell_pairs

def main():
    graph_types = ["wrap_grid_overlay"]
    for graph_type in graph_types:
        G, consumption_rates = create_graph(n, initial_capacity, graph_type)
        
        print(f"Simulating distillation effect for {graph_type}")
        distillations = list(range(1, 6))
        sim_args = [(G, consumption_rates, 5_000_000, generation_rate, 200, d, True) for d in distillations]

        with Pool(processes=min(1, len(sim_args))) as pool:
            results = pool.map(run_simulation, sim_args)

        print(results)
        with open("results.txt", "a") as f:
            for d, r in zip(distillations, results):
                f.write(f"Distillation pairs: {d}\n")
                f.write(f"  Total swaps: {r[0]}\n")
                f.write(f"  Optimal direct path sum: {r[1]}\n")
                f.write(f"  Worst direct path sum: {r[2]}\n")
                f.write(f"  Total pairs consumed: {r[3]}\n")
                f.write(f"  Total bell pairs: {r[4]}\n")
                f.write("\n")  # Add a blank line between entries
        df = pd.DataFrame({
            "distillation pairs": distillations,
            "total swaps": [r[0] for r in results],
            "optimal direct path sum": [r[1] for r in results],
            "worst direct path sum": [r[2] for r in results],
            "total pairs consumed": [r[3] for r in results],
            "total bell pairs": [r[4] for r in results]
        })

        df.to_csv(f"{graph_type}_distillation_metrics_output-1.csv", index=False)

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()