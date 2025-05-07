import polars as pl
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import qnet

# This is for plotting performance.
alt.data_transformers.enable("vegafusion")

sim = qnet.SimParams().build()

df_edges_list = []
flows_list = []
for _ in range(100):
    sim.run(1)
    time = sim.time()

    df_edges_part = (
        pl.DataFrame(sim.edge_capacities())
        .with_columns(time=time)
        .select("time", "node_a", "node_b", "link_capacity")
    )
    df_edges_list.append(df_edges_part)

    max_flow_avg, max_flow_std = sim.max_flow_stats(100)
    flows_point = {
        "time": time,
        "max_flow_avg": max_flow_avg,
        "max_flow_std": max_flow_std,
    }
    flows_list.append(flows_point)

df_edges = pl.concat(df_edges_list)
df_flows = pl.DataFrame(flows_list)

# Find the median link capacity for each node over time.
median_capacities = (
    df_edges.lazy()
    .sort("time")
    .group_by("time")
    .agg(
        link_capacity_p50=pl.col("link_capacity").quantile(0.5),
        link_capacity_avg=pl.col("link_capacity").mean(),
    )
    .sort("time")
    .collect()
)

alt.Chart(median_capacities).mark_point().encode(
    x=alt.X("time:Q", scale=alt.Scale(type="log")),
    y=alt.Y("link_capacity_avg:Q"),
).save("results/edge_capacities.png", ppi=300)

alt.Chart(df_flows).mark_point().encode(
    x=alt.X("time:Q", scale=alt.Scale(type="log")),
    y="max_flow_avg:Q",
).save("results/max_flows.png", ppi=300)
