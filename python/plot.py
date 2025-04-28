import polars as pl
import altair as alt

# This is for plotting performance.
alt.data_transformers.enable("vegafusion")


df = pl.read_parquet("results/simlogs.parquet")
print(df)

# # Generate a list of all combinations of a and b so tha we can
# # complete the table with zeros instead of nulls. Arguably we should
# # do this in the simulation output so we don't have to do it here.
# num_nodes = df["peer_a"].max()
# print(num_nodes)

# all_pairs_a = list()
# all_pairs_b = list()
# for a in range(num_nodes + 1):
#     for b in range(num_nodes + 1):
#         if a != b:
#             all_pairs_a.append(a)
#             all_pairs_b.append(b)

# all_pairs_df = pl.DataFrame({"peer_a": all_pairs_a, "peer_b": all_pairs_b})

df = (
    df.lazy()
    .group_by("time", "peer_a")
    .agg(link_capacity_p50=pl.col("link_capacity").quantile(0.5))
    .sort("time", "peer_a")
    .collect()
)

alt.Chart(df).mark_point().encode(
    x=alt.X("time:Q"),
    y=alt.Y("link_capacity_p50:Q"),
).save("results/plot.png", ppi=300)

print(df)
