use std::path::PathBuf;

use peer::DirectGraph;
use simulation::{LogFrequency, SimLog, SimParams, Simulation, TimeMillis};

// TODO(improvement): these should be a parameters of individual nodes and links.
const DECOHERENCE_FACTOR: f64 = 0.1; // TODO(!): make this not zero
const LINK_RATE: f64 = 5.0;

fn main() -> anyhow::Result<()> {
    let direct_graph = DirectGraph::random(50, 0.5, &mut rand::rng());
    let params = SimParams {
        swap_fraction: 0.2,
        tick_interval: 1,
        log_frequency: LogFrequency::All,
    };

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
                .map(|(a, b, edge)| (a as u64, b as u64, edge.link_capacity))
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

pub mod simulation {
    use rand::Rng;

    use crate::peer::{DirectGraph, EntangleGraph};

    /// This is the base unit of time in the system, in milliseconds.
    pub type TimeMillis = u64;

    /// Expected number of events per timestep.
    pub type Rate = f64;

    pub struct Simulation {
        params: SimParams,
        direct_graph: DirectGraph,
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

            let tick_log = self.tick(self.params.tick_interval, rng);
            logs.push(tick_log);

            logs
        }

        fn tick<R: Rng>(&mut self, interval: TimeMillis, rng: &mut R) -> SimLog {
            self.entangle_graph
                .tick(interval, self.params.swap_fraction, rng);
            self.direct_graph
                .tick(interval, &mut self.entangle_graph, rng);
            self.current_time += interval;

            const MAX_FLOW_COUNT: u64 = 50;
            let mut max_flow_sum = 0;
            for _ in 0..MAX_FLOW_COUNT {
                let node_a = rng.random_range(0..self.direct_graph.num_nodes());
                let node_b = loop {
                    let candidate = rng.random_range(0..self.direct_graph.num_nodes());
                    if candidate != node_a {
                        break candidate;
                    }
                };

                max_flow_sum += self.entangle_graph.max_flow(node_a, node_b);
            }
            let max_flow_avg = max_flow_sum as f64 / MAX_FLOW_COUNT as f64;

            // TODO(opt): we always create these logs even though they may
            // not be used. Can we do something smarter?
            SimLog {
                entangle_graph: self.entangle_graph.clone(),
                time: self.current_time,
                max_flow_avg,
            }
        }

        pub fn new(params: SimParams, direct_graph: DirectGraph) -> Self {
            Self {
                params,
                current_time: 0,
                direct_graph,
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
        pub max_flow_avg: f64,
    }
}

pub mod peer {
    use std::collections::{HashMap, HashSet, VecDeque};

    use rand::{Rng, seq::IndexedRandom};
    use rand_distr::{Binomial, Distribution as _, Poisson};

    use crate::{
        DECOHERENCE_FACTOR, LINK_RATE,
        simulation::{Rate, TimeMillis},
    };

    pub type NodeId = u64;

    /// Holds the connections between nodes that are physically linked and
    /// can generate entangled pairs on-demand.
    #[derive(Clone, Debug, Default)]
    pub struct DirectGraph {
        num_nodes: u64,
        map: HashMap<(NodeId, NodeId), NeighborEdge>,
    }

    impl DirectGraph {
        pub fn num_nodes(&self) -> u64 {
            self.num_nodes
        }

        pub fn tick<R: Rng>(
            &self,
            interval: TimeMillis,
            entangle_peer_map: &mut EntangleGraph,
            rng: &mut R,
        ) {
            for ((node_a, node_b), direct_edge) in self.map.iter() {
                let num_pairs_expected = direct_edge.link_rate * interval as f64;
                let num_pairs_generated = Poisson::new(num_pairs_expected).unwrap().sample(rng);
                entangle_peer_map.add_capacity(num_pairs_generated as u64, *node_a, *node_b);
            }
        }

        /// Generate a new DirectGraph with random connections.
        pub fn random<R: Rng>(num_nodes: u64, density: f64, rng: &mut R) -> Self {
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

            Self {
                map: neighbor_map,
                num_nodes,
            }
        }
    }

    #[derive(Clone, Debug)]
    struct NeighborEdge {
        /// How many Bell pairs can be (successfully) exchanged per time step
        link_rate: Rate,
    }

    #[derive(Clone, Debug, Default)]
    pub struct EntangleGraph {
        neighbors: HashMap<NodeId, HashSet<NodeId>>,
        edges: HashMap<(NodeId, NodeId), EntangleEdge>,
    }

    /// Get a canonical key representing the edge between two nodes.
    fn edge_key(node_a: NodeId, node_b: NodeId) -> (NodeId, NodeId) {
        (std::cmp::min(node_a, node_b), std::cmp::max(node_a, node_b))
    }

    impl EntangleGraph {
        fn add_capacity(&mut self, capacity_amt: u64, node_a: NodeId, node_b: NodeId) {
            let key = (node_a, node_b);
            let entry = self
                .edges
                .entry(key)
                .or_insert(EntangleEdge { link_capacity: 0 });
            entry.link_capacity = entry.link_capacity.checked_add(capacity_amt).unwrap();

            let entry = self.neighbors.entry(node_a).or_default();
            let _ = entry.insert(node_b);

            let entry = self.neighbors.entry(node_b).or_default();
            let _ = entry.insert(node_a);
        }

        /// PANICS: This will panic if you try to subtract more than a connection has.
        fn sub_capacity(&mut self, capacity_amt: u64, node_a: NodeId, node_b: NodeId) {
            let key = (node_a, node_b);
            let entry = self.edges.get_mut(&key).unwrap();
            entry.link_capacity = entry.link_capacity.checked_sub(capacity_amt).unwrap();

            if entry.link_capacity == 0 {
                self.neighbors.get_mut(&node_a).unwrap().remove(&node_b);
                self.neighbors.get_mut(&node_b).unwrap().remove(&node_a);
            }
        }

        pub fn get_capacity(&self, node_a: NodeId, node_b: NodeId) -> u64 {
            let key = (node_a, node_b);
            match self.edges.get(&key) {
                Some(entangle_edge) => entangle_edge.link_capacity,
                None => 0,
            }
        }

        pub fn tick<R: Rng>(&mut self, interval: TimeMillis, swap_fraction: f64, rng: &mut R) {
            // TODO(opt): Maybe better to handle decoherences as needed instead of accounting for
            // everything on every time step.
            for (node_a, node_b) in self.edges.clone().keys() {
                // Account for decoherence.
                let p_single_decay = 1.0 - (-DECOHERENCE_FACTOR * interval as f64).exp();
                let link_capacity = self.get_capacity(*node_a, *node_b);
                let dist = Binomial::new(link_capacity, p_single_decay).unwrap();
                let num_decoherences: u64 = dist.sample(rng);

                self.sub_capacity(num_decoherences, *node_a, *node_b);
            }

            let neighbor_map_snapshot = self.neighbors.clone();
            for node in neighbor_map_snapshot.keys() {
                let neighbors: Vec<NodeId> = match self.neighbors.get(node) {
                    None => continue,
                    Some(neighbors) if neighbors.len() == 1 => continue,
                    Some(neighbors) => neighbors.iter().copied().collect(),
                };

                // Choose two neighbors and swap swap_fraction of the lower of their channel
                // capacities. We're going to do this a number of times proportional to the
                // number of neighbors the node has.
                for _ in 0..neighbors.len() {
                    let two_neighbors: Vec<_> = neighbors.choose_multiple(rng, 2).collect();
                    let (node_a, node_b) = (*two_neighbors[0], *two_neighbors[1]);

                    let cap_a = self.get_capacity(*node, node_a);
                    let cap_b = self.get_capacity(*node, node_b);
                    let swap_number = (std::cmp::min(cap_a, cap_b) as f64 * swap_fraction) as u64;

                    self.sub_capacity(swap_number, *node, node_a);
                    self.sub_capacity(swap_number, *node, node_b);
                    self.add_capacity(swap_number, node_a, node_b);
                }
            }
        }

        pub fn edges(&self) -> impl Iterator<Item = (NodeId, NodeId, &EntangleEdge)> + '_ {
            self.edges.iter().map(|((a, b), edge)| (*a, *b, edge))
        }

        pub fn max_flow(&self, source: NodeId, destination: NodeId) -> u64 {
            let mut residual = self.edges.clone();
            let mut max_flow = 0;

            loop {
                // Find a path from source to destination using BFS
                let mut parent: HashMap<NodeId, Option<NodeId>> = HashMap::new();
                let mut visited = HashSet::new();
                let mut queue = VecDeque::new();

                queue.push_back(source);
                parent.insert(source, None);
                visited.insert(source);

                let mut found_destination = false;

                while let Some(u) = queue.pop_front() {
                    // TODO: change this to neighbour map code
                    for ((from, to), edge) in residual.iter() {
                        if *from == u && edge.link_capacity > 0 && !visited.contains(to) {
                            parent.insert(*to, Some(u));
                            if *to == destination {
                                found_destination = true;
                                break;
                            }
                            visited.insert(*to);
                            queue.push_back(*to);
                        }
                    }
                    if found_destination {
                        break;
                    }
                }

                // No augmenting path found
                if !found_destination {
                    break;
                }

                // Trace back to find bottleneck capacity
                let mut path = vec![];
                let mut cur = destination;
                while let Some(&Some(prev)) = parent.get(&cur) {
                    path.push((prev, cur));
                    cur = prev;
                }
                path.reverse();

                let min_capacity = path
                    .iter()
                    .map(|(u, v)| residual.get(&(*u, *v)).unwrap().link_capacity)
                    .min()
                    .unwrap();

                // Augment the flow
                for (u, v) in path.iter() {
                    residual.get_mut(&(*u, *v)).unwrap().link_capacity -= min_capacity;
                    // Add reverse edge
                    residual
                        .entry((*v, *u))
                        .and_modify(|e| e.link_capacity += min_capacity)
                        .or_insert(EntangleEdge {
                            link_capacity: min_capacity,
                        });
                }

                max_flow += min_capacity;
            }

            max_flow
        }
    }

    #[derive(Clone, Debug)]
    pub struct EntangleEdge {
        /// How many Bell pairs have been (successfully) pre-exchanged and
        /// are now in storage waiting to execute teleportation
        pub link_capacity: u64,
    }

    #[cfg(test)]
    mod tests {
        use super::EntangleGraph;


        #[test]
        fn test_max_flow() {

            let mut graph2 = EntangleGraph::default();

            graph2.add_capacity(3, 0, 1);
            graph2.add_capacity(3, 1, 3);
            graph2.add_capacity(2, 3, 5);
            graph2.add_capacity(7, 0, 2);
            graph2.add_capacity(5, 2, 1);

            graph2.add_capacity(4, 1, 4);
            graph2.add_capacity(3, 2, 4);
            graph2.add_capacity(3, 3, 4);
            graph2.add_capacity(6, 4, 5);

            assert_eq!(8, graph2.max_flow(0, 5));
        }
    }
}
