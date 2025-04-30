import polars as pl
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import qnet_sim

print(qnet_sim.hello_world())

# This is for plotting performance.
alt.data_transformers.enable("vegafusion")

df = pl.read_parquet("results/simlogs.parquet")

print(df)

# Find the median link capacity for each node over time.
median_capacities = (
    df.lazy()
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
).save("results/plot.png", ppi=300)

# df2 = pl.read_parquet("max_flows.parquet").with_row_index()


# # Create a simple range for the x-axis
# x_range = list(range(len(df2)))

# # Create the scatterplot
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=x_range, y=df2['max_flow_averages'])
# plt.xlabel('Index')
# plt.ylabel('Max Flow Averages')
# plt.semilogx( )
# plt.title('Scatterplot of Max Flow Averages')
# plt.tight_layout()
# plt.savefig('results/plot2.png')
# plt.close()
