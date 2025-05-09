import polars as pl
import altair as alt
import qnet

# This is for plotting performance.
alt.data_transformers.enable("vegafusion")

sim = qnet.SimParams().build()

df_edges_list = []
flows_list = []
gen_flows_list = []
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

    # Calculate generalized max flow statistics
    gen_max_flow_avg, gen_max_flow_std = sim.generalized_max_flow_stats(100)
    gen_flows_point = {
        "time": time,
        "gen_max_flow_avg": gen_max_flow_avg,
        "gen_max_flow_std": gen_max_flow_std,
    }
    gen_flows_list.append(gen_flows_point)

df_edges = pl.concat(df_edges_list)
df_flows = pl.DataFrame(flows_list)
df_gen_flows = pl.DataFrame(gen_flows_list)

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

# Plot generalized max flow
alt.Chart(df_gen_flows).mark_point().encode(
    x=alt.X("time:Q", scale=alt.Scale(type="log")),
    y="gen_max_flow_avg:Q",
).save("results/generalized_max_flows.png", ppi=300)

# Combined plot to compare regular vs generalized max flow
df_combined = df_flows.select("time", pl.col("max_flow_avg").alias("Traditional Max Flow"))
df_combined = df_combined.join(
    df_gen_flows.select("time", pl.col("gen_max_flow_avg").alias("Generalized Max Flow")),
    on="time"
)
df_combined_long = df_combined.melt(
    id_vars=["time"],
    value_vars=["Traditional Max Flow", "Generalized Max Flow"],
    variable_name="flow_type",
    value_name="flow_value"
)

alt.Chart(df_combined_long).mark_line().encode(
    x=alt.X("time:Q", scale=alt.Scale(type="log")),
    y="flow_value:Q",
    color="flow_type:N",
).save("results/comparison_max_flows.png", ppi=300)
