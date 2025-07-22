import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
file_path = "wrap_grid_overlay_swap_rate_metrics_output-final.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
graph_type = "wrap grid"

# Compute metrics
df["swap_to_direct"] = df["stabilised total swap count"] / df["stabilised optimal direct graph_path_sum"]
df["swap_to_corrected"] = df["stabilised total swap count"] / df["stabilised optimal direct graph_path_sum_corrected"]
df["failure_ratio"] = df["failed consumption count"] / (
    df["failed consumption count"] + df["successful consumption count"]
)
df["total_bell_pairs"] = df["total bell pairs in graph"]

# --- Plot 1: Swap ratios ---
plt.figure(figsize=(10, 5))
plt.plot(df["swap rates"], df["swap_to_direct"], marker='o', label='Swap / Optimal Direct Path Sum', linewidth=3.5)
plt.xlabel("Swap rate")
plt.ylabel("Ratio")
plt.title(f"{graph_type} graph, nodes = 50 Swap Overhead vs Direct Path Sum")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Failure ratio ---
plt.figure(figsize=(8, 4.5))
plt.plot(df["swap rates"], df["failure_ratio"], marker='x', color='red', label='Failure Ratio', linewidth=3.5)
plt.xlabel("Swap rate")
plt.ylabel("Failure / (Failure + Success)")
plt.title(f"{graph_type} graph, nodes = 50 Failure Ratio vs Swap rate")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: Failure ratio ---
plt.figure(figsize=(8, 4.5))
plt.plot(df["swap rates"], df["total_bell_pairs"], marker='x', color='red', label='Failure Ratio', linewidth=3.5)
plt.xlabel("Swap rate")
plt.ylabel("Total bell pairs")
plt.title(f"{graph_type} graph, nodes = 50 Total bell pairs upon stabilisaton")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()