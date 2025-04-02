use std::path::PathBuf;

use peer::DirectGraph;
use simulation::{LogFrequency, SimLog, SimParams, Simulation, TimeMillis};

// TODO(improvement): these should be a parameters of individual nodes and links.
const DECOHERENCE_FACTOR: f64 = 0.1;
const LINK_RATE: f64 = 1.0;

fn main() -> anyhow::Result<()> {
    let direct_graph = DirectGraph::random(100, 0.01, &mut rand::rng());
    let params = SimParams {
        indirect_rate: 10.0,
        tick_interval: 1,
        log_frequency: LogFrequency::All,
    };

    let mut simulation = Simulation::new(params, direct_graph);

    let logs = simulation.run(10_000, &mut rand::rng());
    write_simlogs_to_parquet("simlogs.parquet".into(), logs)?;

    Ok(())
}

fn write_simlogs_to_parquet(path: PathBuf, logs: Vec<SimLog>) -> anyhow::Result<()> {
    use polars::prelude::ParquetWriter;

    let (peers_a, peers_b, capacities): (Vec<_>, Vec<_>, Vec<_>) = logs
        .iter()
        .flat_map(|log| {
            log.indirect
                .edges()
                .map(|(a, b, edge)| (a as u64, b as u64, edge.link_capacity))
        })
        .collect();

    let times: Vec<TimeMillis> = logs
        .iter()
        .flat_map(|log| log.indirect.edges().map(|_| log.time))
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

    use crate::peer::{DirectGraph, IndirectGraph};

    /// This is the base unit of time in the system, in milliseconds.
    pub type TimeMillis = u64;

    /// Expected number of events per timestep.
    pub type Rate = f64;

    pub struct Simulation {
        params: SimParams,
        direct_graph: DirectGraph,
        indirect_graph: IndirectGraph,
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
            self.indirect_graph.tick(interval, rng);
            self.direct_graph
                .tick(interval, &mut self.indirect_graph, rng);
            self.current_time += interval;

            // TODO(opt): we always create these logs even though they may
            // not be used. Can we do something smarter?
            SimLog {
                indirect: self.indirect_graph.clone(),
                time: self.current_time,
            }
        }

        pub fn new(params: SimParams, direct_graph: DirectGraph) -> Self {
            Self {
                params,
                current_time: 0,
                direct_graph,
                indirect_graph: IndirectGraph::default(),
            }
        }
    }

    pub struct SimParams {
        /// Effective channel capacity for indirect connections. This basically
        /// accounts for how many entanglement swaps per second a node can perform.
        pub indirect_rate: Rate,

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
        pub indirect: IndirectGraph,
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
    pub struct DirectGraph {
        map: HashMap<(NodeId, NodeId), DirectEdge>,
    }

    impl DirectGraph {
        pub fn tick<R: Rng>(
            &self,
            interval: TimeMillis,
            indirect_peer_map: &mut IndirectGraph,
            rng: &mut R,
        ) {
            for ((peer_a, peer_b), direct_edge) in self.map.iter() {
                let num_pairs_expected = direct_edge.link_rate * interval as f64;
                let num_pairs_generated = Poisson::new(num_pairs_expected).unwrap().sample(rng);

                let indirect_edge = indirect_peer_map
                    .map
                    .entry((*peer_a, *peer_b))
                    .or_insert(IndirectEdge { link_capacity: 0 });

                indirect_edge.link_capacity += num_pairs_generated as u64;
            }
        }

        /// Generate a new DirectGraph with random connections.
        pub fn random<R: Rng>(num_nodes: usize, density: f64, rng: &mut R) -> Self {
            let mut direct_map = HashMap::new();

            for i in 0..num_nodes {
                for j in 0..num_nodes {
                    if i == j {
                        continue;
                    }

                    if rng.random_bool(density) {
                        let old_val = direct_map.insert(
                            (i, j),
                            DirectEdge {
                                link_rate: LINK_RATE,
                            },
                        );
                        assert!(old_val.is_none());
                    }
                }
            }

            Self { map: direct_map }
        }
    }

    #[derive(Clone, Debug)]
    struct DirectEdge {
        /// How many Bell pairs can be (successfully) exchanged per time step
        link_rate: Rate,
    }

    #[derive(Clone, Debug, Default)]
    pub struct IndirectGraph {
        map: HashMap<(NodeId, NodeId), IndirectEdge>,
    }

    impl IndirectGraph {
        pub fn tick<R: Rng>(&mut self, interval: TimeMillis, rng: &mut R) {
            for indirect_edge in self.map.values_mut() {
                let p_single_decay = 1.0 - (-DECOHERENCE_FACTOR * interval as f64).exp();
                let dist = Binomial::new(indirect_edge.link_capacity, p_single_decay).unwrap();
                let num_decoherences: u64 = dist.sample(rng);

                indirect_edge.link_capacity = indirect_edge
                    .link_capacity
                    .checked_sub(num_decoherences)
                    .unwrap();

                if indirect_edge.link_capacity == 0 {
                    // TODO(improvement): for consistency, maybe we should
                    // figure out a way to fully remove this edge so we
                    // don't get a bunch of "zero" entries in the log.
                }
            }
        }

        pub fn edges(&self) -> impl Iterator<Item = (NodeId, NodeId, &IndirectEdge)> + '_ {
            self.map.iter().map(|((a, b), edge)| (*a, *b, edge))
        }
    }

    #[derive(Clone, Debug)]
    pub struct IndirectEdge {
        /// How many Bell pairs have been (successfully) pre-exchanged and
        /// are now in storage waiting to execute teleportation
        pub link_capacity: u64,
    }
}
