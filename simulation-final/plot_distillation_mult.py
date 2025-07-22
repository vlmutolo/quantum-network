import pandas as pd
import matplotlib.pyplot as plt

graph_types = ["cycle", "wrap_grid_overlay"]
nodes = 25
data = {}

# Load and compute metrics for both graph types
for graph_type in graph_types:
    file_path = f"{graph_type}_distillation_metrics_output_infinite_swapping.csv"
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    df["swap_overhead"] = df["total swaps"] / df["optimal direct path sum"]
    df["swap_efficiency"] = df["total swaps"] / df["total pairs consumed"]
    df["total_bell_pairs"] = df["total bell pairs"]
    df["total_pairs_consumed"] = df["total pairs consumed"]

    data[graph_type] = df

def get_style(graph_type):
    return {'linestyle': '--'} if graph_type == "wrap_grid_overlay" else {}

# --- Plot 1: Swap Overhead ---
plt.figure(figsize=(10, 5))
for graph_type in graph_types:
    df = data[graph_type]
    plt.plot(df["distillation pairs"], df["swap_overhead"], marker='o', linewidth=3.5, label=graph_type, **get_style(graph_type))
plt.xlabel("Distillation Pairs Count")
plt.ylabel("Swap Overhead (Swaps / Optimal Direct Path Sum)")
plt.title(f"Swap Overhead vs Distillation Pairs (Nodes = {nodes})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Swap Efficiency ---
plt.figure(figsize=(10, 5))
for graph_type in graph_types:
    df = data[graph_type]
    plt.plot(df["distillation pairs"], df["swap_efficiency"], marker='o', linewidth=3.5, label=graph_type, **get_style(graph_type))
plt.xlabel("Distillation Pairs Count")
plt.ylabel("Swap Efficiency (Swaps / Pairs Consumed)")
plt.yscale("log") 
plt.title(f"Swap Efficiency vs Distillation Pairs (Nodes = {nodes})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 3: Total Bell Pairs ---
plt.figure(figsize=(10, 5))
for graph_type in graph_types:
    df = data[graph_type]
    plt.plot(df["distillation pairs"], df["total_bell_pairs"], marker='x', linewidth=3.5, label=graph_type, **get_style(graph_type))
plt.xlabel("Distillation Pairs Count")
plt.ylabel("Total Bell Pairs")
plt.title(f"Total Bell Pairs vs Distillation Pairs (Nodes = {nodes})")
plt.yscale("log") 
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 4: Total Pairs Consumed ---
plt.figure(figsize=(10, 5))
for graph_type in graph_types:
    df = data[graph_type]
    plt.plot(df["distillation pairs"], df["total_pairs_consumed"], marker='x', linewidth=3.5, label=graph_type, **get_style(graph_type))
plt.xlabel("Distillation Pairs Count")
plt.ylabel("Total Pairs Consumed")
plt.title(f"Total Pairs Consumed vs Distillation Pairs (Nodes = {nodes})")
plt.yscale("log") 
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
