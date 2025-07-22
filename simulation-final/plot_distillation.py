import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
graph_type = "cycle"
nodes = 25
file_path = f"{graph_type}_distillation_metrics_output_infinite_swapping.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# Compute metrics

df["swap_overhead"] = df["total swaps"] / df["optimal direct path sum"]
df["swap_efficiency"] = df["total swaps"] / df["total pairs consumed"]
df["total_bell_pairs"] = df["total bell pairs"]
df["total_pairs_consumed"] = df["total pairs consumed"]

# --- Plot 1: Swap ratios ---
plt.figure(figsize=(10, 5))
plt.plot(df["distillation pairs"], df["swap_overhead"], marker='o', linewidth=3.5, label='Total Swap / Optimal Direct Path Sum')
plt.xlabel("Distillation Pairs Count")
plt.ylabel("Ratio")
plt.title(f"{graph_type} graph, nodes = {nodes} Swap Overhead vs Direct Path Sum")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Failure ratio ---
plt.figure(figsize=(10, 5))
plt.plot(df["distillation pairs"], df["swap_efficiency"], marker='o', label='Total Swaps/Total pairs consumed', linewidth=3.5)
plt.xlabel("Distillation Pairs Count")
plt.ylabel("Ratio")
plt.title(f"{graph_type} graph, nodes = {nodes} total swaps/total pairs consumed")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Failure ratio ---
plt.figure(figsize=(8, 4.5))
plt.plot(df["distillation pairs"], df["total_bell_pairs"], marker='x', color='red', label='Total bell pairs', linewidth=3.5)
plt.xlabel("Distillation Pairs Count")
plt.ylabel("Total bell pairs")
plt.title(f"{graph_type} graph, nodes = {nodes} Total bell pairs upon stabilisaton")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4.5))
plt.plot(df["distillation pairs"], df["total_pairs_consumed"], marker='x', color='red', label='Total pairs consumed upon stabilisation', linewidth=3.5)
plt.xlabel("Distillation Pairs Count")
plt.ylabel("Total bell pairs")
plt.title(f"{graph_type} graph, nodes = {nodes} Total pairs consumed upon stabilisation")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()