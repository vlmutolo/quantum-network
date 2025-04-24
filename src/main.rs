use std::path::PathBuf;

use peer::NeighborGraph;
use simulation::{LogFrequency, SimLog, SimParams, Simulation, TimeMillis};

// TODO(improvement): these should be a parameters of individual nodes and links.
const DECOHERENCE_FACTOR: f64 = 0.1;
const LINK_RATE: f64 = 1.0;

fn main() -> anyhow::Result<()> {
    let neighbor_graph = NeighborGraph::random(100, 0.01, &mut rand::rng());
    let params = SimParams {
        swap_fraction: 10.0,
        tick_interval: 1,
        log_frequency: LogFrequency::All,
    };

    let mut simulation = Simulation::new(params, neighbor_graph);

    let logs = simulation.run(10_000, &mut rand::rng());
    write_simlogs_to_parquet("simlogs.parquet".into(), logs)?;

    Ok(())
}

fn write_simlogs_to_parquet(path: PathBuf, logs: Vec<SimLog>) -> anyhow::Result<()> {
    use polars::prelude::ParquetWriter;

    let (peers_a, peers_b, capacities): (Vec<_>, Vec<_>, Vec<_>) = logs
        .iter()
        .flat_map(|log| {
            log.entangle_graph
                .edges()
                .map(|(a, b, edge)| (a as u64, b as u64, edge.link_capacity))
        })
        .collect();

    let times: Vec<TimeMillis> = logs
        .iter()
        .flat_map(|log| log.entangle_graph.edges().map(|_| log.time))
        .collect();

    let mut df = polars::df!(
        "time" => times,
        "peer_a" => peers_a,
        "peer_b" => peers_b,
        "link_capacity" => capacities,
    )
    .unwrap();

    let mut file = std::fs::File::create(path)?;
    ParquetWriter::new(&mut file).finish(&mut df)?;

    Ok(())
}

pub mod simulation {
    use rand::Rng;

    use crate::peer::{EntangleGraph, NeighborGraph};

    /// This is the base unit of time in the system, in milliseconds.
    pub type TimeMillis = u64;

    /// Expected number of events per timestep.
    pub type Rate = f64;

    pub struct Simulation {
        params: SimParams,
        neighbor_graph: NeighborGraph,
        entangle_graph: EntangleGraph,
        current_time: TimeMillis,
    }

    impl Simulation {
        pub fn run<R: Rng>(&mut self, stop_time: TimeMillis, rng: &mut R) -> Vec<SimLog> {
            // TODO(perf): this function should maybe be able to
            // stream results to a parquet dataframe.

            let mut logs = Vec::new();

            let num_ticks = stop_time
                .saturating_sub(self.current_time)
                .checked_div(self.params.tick_interval)
                .unwrap();

            for _ in 0..num_ticks - 1 {
                let tick_log = self.tick(self.params.tick_interval, rng);
                match self.params.log_frequency {
                    LogFrequency::Final => (),
                    LogFrequency::All => logs.push(tick_log),
                }
            }

            let tick_logs = self.tick(self.params.tick_interval, rng);
            logs.push(tick_logs);

            logs
        }

        fn tick<R: Rng>(&mut self, interval: TimeMillis, rng: &mut R) -> SimLog {
            self.entangle_graph.tick(interval, rng);
            self.neighbor_graph
                .tick(interval, &mut self.entangle_graph, rng);
            self.current_time += interval;

            // TODO(opt): we always create these logs even though they may
            // not be used. Can we do something smarter?
            SimLog {
                entangle_graph: self.entangle_graph.clone(),
                time: self.current_time,
            }
        }

        pub fn new(params: SimParams, neighbor_graph: NeighborGraph) -> Self {
            Self {
                params,
                current_time: 0,
                neighbor_graph,
                entangle_graph: EntangleGraph::default(),
            }
        }
    }

    pub struct SimParams {
        /// This describes what fraction of an entangled channel capacity will
        /// be spent to feed new edges in the entanglement graph.
        pub swap_fraction: Rate,

        /// How far to advance in time per simulation step.
        pub tick_interval: TimeMillis,

        /// Whether or not to save all intermediate states of the simulation.
        pub log_frequency: LogFrequency,
    }

    pub enum LogFrequency {
        Final,
        All,
    }

    /// Summary of a snapshot of the network state
    pub struct SimLog {
        pub entangle_graph: EntangleGraph,
        pub time: TimeMillis,
    }
}

pub mod peer {
    use std::collections::HashMap;

    use rand::Rng;
    use rand_distr::{Binomial, Distribution as _, Poisson};

    use crate::{
        DECOHERENCE_FACTOR, LINK_RATE,
        simulation::{Rate, TimeMillis},
    };

    pub type NodeId = usize;

    /// Holds the connections between nodes that are physically linked and
    /// can generate entangled pairs on-demand.
    #[derive(Clone, Debug, Default)]
    pub struct NeighborGraph {
        map: HashMap<(NodeId, NodeId), NeighborEdge>,
    }

    impl NeighborGraph {
        pub fn tick<R: Rng>(
            &self,
            interval: TimeMillis,
            entangle_peer_map: &mut EntangleGraph,
            rng: &mut R,
        ) {
            for ((peer_a, peer_b), neighbor_edge) in self.map.iter() {
                let num_pairs_expected = neighbor_edge.link_rate * interval as f64;
                let num_pairs_generated = Poisson::new(num_pairs_expected).unwrap().sample(rng);

                let entangle_edge = entangle_peer_map
                    .map
                    .entry((*peer_a, *peer_b))
                    .or_insert(EntangleEdge { link_capacity: 0 });

                entangle_edge.link_capacity += num_pairs_generated as u64;
            }
        }

        /// Generate a new NeighborGraph with random connections.
        pub fn random<R: Rng>(num_nodes: usize, density: f64, rng: &mut R) -> Self {
            let mut neighbor_map = HashMap::new();

            for i in 0..num_nodes {
                for j in 0..num_nodes {
                    if i == j {
                        continue;
                    }

                    if rng.random_bool(density) {
                        let old_val = neighbor_map.insert(
                            (i, j),
                            NeighborEdge {
                                link_rate: LINK_RATE,
                            },
                        );
                        assert!(old_val.is_none());
                    }
                }
            }

            Self { map: neighbor_map }
        }
    }

    #[derive(Clone, Debug)]
    struct NeighborEdge {
        /// How many Bell pairs can be (successfully) exchanged per time step
        link_rate: Rate,
    }

    #[derive(Clone, Debug, Default)]
    pub struct EntangleGraph {
        map: HashMap<(NodeId, NodeId), EntangleEdge>,
    }

    impl EntangleGraph {
        pub fn tick<R: Rng>(&mut self, interval: TimeMillis, rng: &mut R) {
            for entangle_edge in self.map.values_mut() {
                let p_single_decay = 1.0 - (-DECOHERENCE_FACTOR * interval as f64).exp();
                let dist = Binomial::new(entangle_edge.link_capacity, p_single_decay).unwrap();
                let num_decoherences: u64 = dist.sample(rng);

                entangle_edge.link_capacity = entangle_edge
                    .link_capacity
                    .checked_sub(num_decoherences)
                    .unwrap();

                if entangle_edge.link_capacity == 0 {
                    // TODO(improvement): for consistency, maybe we should
                    // figure out a way to fully remove this edge so we
                    // don't get a bunch of "zero" entries in the log.
                }
            }
        }

        pub fn edges(&self) -> impl Iterator<Item = (NodeId, NodeId, &EntangleEdge)> + '_ {
            self.map.iter().map(|((a, b), edge)| (*a, *b, edge))
        }
    }

    #[derive(Clone, Debug)]
    pub struct EntangleEdge {
        /// How many Bell pairs have been (successfully) pre-exchanged and
        /// are now in storage waiting to execute teleportation
        pub link_capacity: u64,
    }
}
