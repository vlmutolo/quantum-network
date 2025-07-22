import pandas as pd
import matplotlib.pyplot as plt

# Load CSV
file_path = "cycle_distillation_pair_cpp_output.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()
graph_type = "Cycle"

# Compute metrics
df["swap_to_direct"] = df["total swaps"] / df["optimal"]
df["swap_to_corrected"] = df["total swaps"] / df["corrected"]
df["failure_ratio"] = df["failed"] / (
    df["failed"] + df["succeeded"]
)
df["total_bell_pairs"] = df["final bell pairs"]

# --- Plot 1: Swap ratios ---
plt.figure(figsize=(10, 5))
plt.plot(df["distillation pairs"], df["swap_to_direct"], marker='o', linewidth=3.5, label='Swap / Optimal Direct Path Sum')
plt.plot(df["distillation pairs"], df["swap_to_corrected"], marker='x', linewidth=3.5, label='Swap / Corrected Direct Path Sum')
plt.xlabel("Distillation Pairs Count")
plt.ylabel("Ratio")
plt.title(f"{graph_type} graph, nodes = 50 swap cpp file Overhead vs Direct Path Sum")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot 2: Failure ratio ---
plt.figure(figsize=(8, 4.5))
plt.plot(df["distillation pairs"], df["failure_ratio"], marker='x', linewidth=3.5, color='red', label='Failure Ratio')
plt.xlabel("Distillation Pairs Count")
plt.ylabel("Failure / (Failure + Success)")
plt.title(f"{graph_type} graph, nodes = 50 swap cpp file Failure Ratio vs Distillation Pairs Count")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Plot 2: Failure ratio ---
plt.figure(figsize=(8, 4.5))
plt.plot(df["distillation pairs"], df["total_bell_pairs"], marker='x', color='red', label='Failure Ratio', linewidth=3.5)
plt.xlabel("Distillation Pairs Count")
plt.ylabel("Total bell pairs")
plt.title(f"{graph_type} graph, nodes = 50 swap cpp file Total bell pairs upon stabilisaton")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()