import itertools
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import math
import pandas as pd

# Parameters
n = 50
initial_capacity = 20
generation_rate = 40
edges_with_consumption_non_zero_rate = 50

def norm_edge(u, v):
    return (u, v) if u <= v else (v, u)

def generate_recursive_array(D, N=50):
    f = [0] * N
    if N > 1:
        f[1] = D
    for n in range(2, N):
        f[n] = D * (f[n // 2] + f[(n + 1) // 2])
    return f

def create_overlay_on_grid_graph(G, initial_capacity, wrap=True):
    rows, cols = 10, 5
    for i in range(rows):
        for j in range(cols):
            G.add_node((i, j))
    edges = []
    for i in range(rows):
        for j in range(cols):
            current = (i, j)
            right = (i, (j + 1) % cols) if wrap else (i, j + 1)
            down = ((i + 1) % rows, j) if wrap else (i + 1, j)
            if wrap or j + 1 < cols:
                edges.append((current, right))
            if wrap or i + 1 < rows:
                edges.append((current, down))
    random.shuffle(edges)
    for u, v in edges:
        G.add_edge(u, v, capacity=initial_capacity)
        if nx.is_connected(G):
            break
    extra_edges_to_add = int(0.2 * len(edges))
    added = 0
    for u, v in edges:
        if not G.has_edge(u, v):
            G.add_edge(u, v, capacity=initial_capacity)
            added += 1
        if added >= extra_edges_to_add:
            break
    return G

def create_graph(n, initial_capacity, graph_type):
    G = nx.Graph()
    if graph_type == "cycle":
        for i in range(n):
            G.add_edge(i, (i + 1) % n, capacity=initial_capacity)

    elif graph_type == "line":
        for i in range(n - 1):
            G.add_edge(i, i + 1, capacity=initial_capacity)

    elif graph_type in ["wrap_grid", "wrap_grid_overlay"]:
        G = create_overlay_on_grid_graph(G, initial_capacity, wrap=True)

    elif graph_type in ["non_wrap_grid", "non_wrap_grid_overlay"]:
        G = create_overlay_on_grid_graph(G, initial_capacity, wrap=False)

    nodes = [(i, j) for i in range(10) for j in range(5)] if "grid" in graph_type else list(range(n))
    all_possible_edges = [norm_edge(u, v) for u, v in itertools.combinations(nodes, 2)]
    random_edges = random.sample(all_possible_edges, edges_with_consumption_non_zero_rate)
    consumption_rates = {edge: random.randint(1, 3) for edge in random_edges}
    return G, consumption_rates

def is_stabilized(arr, num_last=20, tolerance=0.005):
    if len(arr) < max(2, num_last):
        return False
    diffs = np.abs(np.diff(arr[-num_last:]))
    return np.max(diffs) < tolerance

def get_neighbours(edge_states, x):
    return {v if u == x else u for u, v in edge_states if x in (u, v) and edge_states[(u, v)] > 0}

def get_preferable_swaps(neighbors, edge_states, x, distillation_pairs):
    swaps = []
    for y, z in itertools.combinations(neighbors, 2):
        c_xy = edge_states.get(norm_edge(x, y), 0)
        c_xz = edge_states.get(norm_edge(x, z), 0)
        c_yz = edge_states.get(norm_edge(y, z), 0)
        if c_xy >= distillation_pairs and c_xz >= distillation_pairs and c_xy > c_yz + distillation_pairs and c_xz > c_yz + distillation_pairs:
            swaps.append(((y, z), c_yz))
    return swaps

def attempt_preferable_swap(x, neighbors, edge_states, distillation_pairs, swap_count, total_bell_pairs):
    for (y, z), _ in sorted(get_preferable_swaps(neighbors, edge_states, x, distillation_pairs), key=lambda item: item[1]):
        edge_xy, edge_xz, edge_yz = norm_edge(x, y), norm_edge(x, z), norm_edge(y, z)
        if edge_states[edge_xy] >= distillation_pairs and edge_states[edge_xz] >= distillation_pairs:
            edge_states[edge_xy] -= distillation_pairs
            edge_states[edge_xz] -= distillation_pairs
            edge_states[edge_yz] = edge_states.get(edge_yz, 0) + 1
            return True, swap_count + 1, total_bell_pairs - 2 * distillation_pairs + 1
    return False, swap_count, total_bell_pairs

def simulate(G, consumption_rates, time_steps, generation_rate, swap_rate, distillation_pairs, infinite_swapping):
    edge_states = {norm_edge(u, v): G[u][v]['capacity'] for u, v in G.edges()}
    nodes = list(G.nodes())

    distillation_sum_multiplier = generate_recursive_array(distillation_pairs)
    consumption_edges = list(consumption_rates.keys())
    consumption_weights = list(consumption_rates.values())
    total_swap_count = failed = succeeded = 0
    total_bell_pairs = initial_capacity * G.number_of_edges()
    optimal_swap_count = 0
    swap_overlay = []
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))

    for t in range(1, time_steps + 1):
        event = random.choices(
            ["generate", "consume", "swap"],
            weights=[len(G.edges()) * generation_rate,
                     sum(consumption_weights),
                     swap_rate * len(nodes)])[0]
        if event == "generate":
            edge = norm_edge(*random.choice(list(G.edges())))
            edge_states[edge] += 1
            total_bell_pairs += 1
        elif event == "consume":
            u, v = random.choices(consumption_edges, weights=consumption_weights)[0]
            edge = norm_edge(u, v)
            if edge_states.get(edge, 0) >= distillation_pairs:
                edge_states[edge] -= distillation_pairs
                path_length = shortest_paths[u][v]
                optimal_swap_count += distillation_sum_multiplier[path_length - 1]
                succeeded += 1
                total_bell_pairs -= distillation_pairs
            else:
                failed += 1
        elif event == "swap":
            if not infinite_swapping:
                x = random.choice(nodes)
                neighbors = get_neighbours(edge_states, x)
                _, total_swap_count, total_bell_pairs = attempt_preferable_swap(
                    x, neighbors, edge_states, distillation_pairs, total_swap_count, total_bell_pairs)
            else:
                for x in random.sample(nodes, len(nodes)):
                    neighbors = get_neighbours(edge_states, x)
                    changed, total_swap_count, total_bell_pairs = attempt_preferable_swap(
                        x, neighbors, edge_states, distillation_pairs, total_swap_count, total_bell_pairs)
                    if changed:
                        break
                    
        if t % 20000 == 0:
            if (t % 100000 == 0):
                print (t, swap_overlay[-5:])
            if optimal_swap_count > 0:
                swap_overlay.append(total_swap_count / optimal_swap_count)
                if is_stabilized(swap_overlay):
                    break
    correction = 0
    for edge, count in edge_states.items():
        if count >= distillation_pairs:
            u, v = edge
            correction += distillation_sum_multiplier[shortest_paths[u][v] - 1]
    return total_swap_count, optimal_swap_count, optimal_swap_count + correction, failed, succeeded, total_bell_pairs

def main():
    graph_types = ["cycle", "wrap_grid_overlay", "non_wrap_grid_overlay"]
    for graph_type in graph_types:
        
        G, consumption_rates = create_graph(n, initial_capacity, graph_type)
        '''
        print(f"Simulating swap overhead for {graph_type}")
        swap_rates = list(range(150, 190, 50))
        
        results = [
            simulate(G, consumption_rates, 2_000_000, generation_rate, rate, 1, False) for rate in swap_rates]
        df = pd.DataFrame({
            "swap rate": swap_rates,
            "total swaps": [r[0] for r in results],
            "optimal": [r[1] for r in results],
            "corrected": [r[2] for r in results],
            "failed": [r[3] for r in results],
            "succeeded": [r[4] for r in results],
            "final bell pairs": [r[5] for r in results]
        })
        df.to_csv(f"{graph_type}_swap_rate_metrics_output.csv", index=False)
        
        print(f"Simulating distillation effect for {graph_type}")
        '''
        distillations = list(range(1, 8))

        results = [
            simulate(G, consumption_rates, 2_500_000, generation_rate, 200, d, False) for d in distillations]
        df = pd.DataFrame({
            "distillation pairs": distillations,
            "total swaps": [r[0] for r in results],
            "optimal": [r[1] for r in results],
            "corrected": [r[2] for r in results],
            "failed": [r[3] for r in results],
            "succeeded": [r[4] for r in results],
            "final bell pairs": [r[5] for r in results]
        })
        df.to_csv(f"{graph_type}_distillation_metrics_output.csv", index=False)

if __name__ == "__main__":
    main()