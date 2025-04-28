import polars as pl
import altair as alt

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

print(median_capacities)
