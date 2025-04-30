use std::path::PathBuf;

use preshared_entanglement_sim::{
    graph::DirectGraph,
    simulation::{LogFrequency, SimLog, SimParams, Simulation, TimeMillis},
};

fn main() -> anyhow::Result<()> {
    let params = SimParams {
        swap_fraction: 0.2,
        tick_interval: 1,
        log_frequency: LogFrequency::All,
        decoherence_factor: 0.1,
        link_rate: 5.0,
        num_nodes: 50,
        direct_edge_density: 0.5,
    };

    let direct_graph = DirectGraph::random(&params, &mut rand::rng());
    let mut simulation = Simulation::new(params, direct_graph);

    let logs = simulation.run(500, &mut rand::rng());
    write_simlogs_to_parquet("results/simlogs.parquet".into(), logs)?;

    Ok(())
}

fn write_simlogs_to_parquet(path: PathBuf, logs: Vec<SimLog>) -> anyhow::Result<()> {
    use polars::prelude::ParquetWriter;

    let (peers_a, peers_b, capacities): (Vec<_>, Vec<_>, Vec<_>) = logs
        .iter()
        .flat_map(|log| {
            log.entangle_graph
                .edges()
                .map(|(a, b, edge)| (a, b, edge.link_capacity))
        })
        .collect();

    let times: Vec<TimeMillis> = logs
        .iter()
        .flat_map(|log| log.entangle_graph.edges().map(|_| log.time))
        .collect();

    let max_flow_averages: Vec<f64> = logs.iter().map(|log| log.max_flow_avg).collect();

    let mut df = polars::df!(
        "time" => times,
        "peer_a" => peers_a,
        "peer_b" => peers_b,
        "link_capacity" => capacities,
    )
    .unwrap();

    let mut df2 = polars::df!(
        "max_flow_averages" => max_flow_averages,
    )
    .unwrap();

    let mut file = std::fs::File::create(path)?;
    ParquetWriter::new(&mut file).finish(&mut df)?;

    let mut file2 = std::fs::File::create("max_flows.parquet")?;
    ParquetWriter::new(&mut file2).finish(&mut df2)?;

    Ok(())
}
