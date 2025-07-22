import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Set global font sizes
plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 20
})

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

# Create PDF
with PdfPages("swap_rate_analysis_plots.pdf") as pdf:

    # Plot 1: Swap Overhead
    plt.figure(figsize=(10, 5))
    for graph_type in graph_types:
        df = data[graph_type]
        plt.plot(df["swap rate"], df["swap_overhead"], marker='o', linewidth=3.5, label=graph_type, **get_style(graph_type))
    plt.xlabel("Swap Rate")
    plt.ylabel("Swap Overhead")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(10, 5))
    for graph_type in graph_types:
        df = data[graph_type]
        plt.plot(df["swap rate"], df["swap_overhead"], marker='o', linewidth=3.5, label=graph_type, **get_style(graph_type))
    plt.xlabel("Swap Rate")
    plt.ylabel("Swap Overhead")
    plt.legend()
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot 2: Swap Efficiency
    plt.figure(figsize=(10, 5))
    for graph_type in graph_types:
        df = data[graph_type]
        plt.plot(df["swap rate"], df["swap_efficiency"], marker='o', linewidth=3.5, label=graph_type, **get_style(graph_type))
    plt.xlabel("Swap Rate")
    plt.ylabel("Swap Efficiency")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot 3: Total Bell Pairs
    plt.figure(figsize=(10, 5))
    for graph_type in graph_types:
        df = data[graph_type]
        plt.plot(df["swap rate"], df["total_bell_pairs"], marker='x', linewidth=3.5, label=graph_type, **get_style(graph_type))
    plt.xlabel("Swap Rate")
    plt.ylabel("Total Bell Pairs")
    plt.yscale("log")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot 4: Total Pairs Consumed
    plt.figure(figsize=(10, 5))
    for graph_type in graph_types:
        df = data[graph_type]
        plt.plot(df["swap rate"], df["total_pairs_consumed"], marker='x', linewidth=3.5, label=graph_type, **get_style(graph_type))
    plt.xlabel("Swap Rate")
    plt.ylabel("Total Pairs Consumed")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

print("PDF exported as 'swap_rate_analysis_plots.pdf'")
