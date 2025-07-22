import pandas as pd
import matplotlib.pyplot as plt

graph_types = ["cycle", "wrap_grid_overlay"]
nodes = 25
data = {}

# Load and process both graph types
for graph_type in graph_types:
    file_path = f"{graph_type}_swap_rate_metrics_pool_output.csv"
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # Compute metrics
    df["swap_overhead"] = df["total swaps"] / df["optimal direct path sum"]
    df["swap_efficiency"] = df["total swaps"] / df["total pairs consumed"]
    df["total_bell_pairs"] = df["total bell pairs"]
    df["total_pairs_consumed"] = df["total pairs consumed"]

    data[graph_type] = df

# Helper for line style
def get_style(graph_type):
    return {'linestyle': '--'} if graph_type == "wrap_grid_overlay" else {}

# --- Plot 1: Swap Overhead ---
plt.figure(figsize=(10, 5))
for graph_type in graph_types:
    df = data[graph_type]
    plt.plot(df["swap rate"], df["swap_overhead"], marker='o', linewidth=3.5, label=graph_type, **get_style(graph_type))
plt.xlabel("Swap Rate")
plt.ylabel("Swap Overhead (Swaps / Optimal Direct Path Sum)")
plt.title(f"Swap Overhead vs Swap Rate (Nodes = {nodes})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Swap Efficiency ---
plt.figure(figsize=(10, 5))
for graph_type in graph_types:
    df = data[graph_type]
    plt.plot(df["swap rate"], df["swap_efficiency"], marker='o', linewidth=3.5, label=graph_type, **get_style(graph_type))
plt.xlabel("Swap Rate")
plt.ylabel("Swap Efficiency (Swaps / Pairs Consumed)")
plt.title(f"Swap Efficiency vs Swap Rate (Nodes = {nodes})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 3: Total Bell Pairs ---
plt.figure(figsize=(10, 5))
for graph_type in graph_types:
    df = data[graph_type]
    plt.plot(df["swap rate"], df["total_bell_pairs"], marker='x', linewidth=3.5, label=graph_type, **get_style(graph_type))
plt.xlabel("Swap Rate")
plt.ylabel("Total Bell Pairs")
plt.yscale("log")
plt.title(f"Total Bell Pairs vs Swap Rate (Nodes = {nodes})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 4: Total Pairs Consumed ---
plt.figure(figsize=(10, 5))
for graph_type in graph_types:
    df = data[graph_type]
    plt.plot(df["swap rate"], df["total_pairs_consumed"], marker='x', linewidth=3.5, label=graph_type, **get_style(graph_type))
plt.xlabel("Swap Rate")
plt.ylabel("Total Pairs Consumed")
plt.title(f"Total Pairs Consumed vs Swap Rate (Nodes = {nodes})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
