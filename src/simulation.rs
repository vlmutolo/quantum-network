use petgraph::graph::NodeIndex;
use rand::{Rng, SeedableRng, rngs::SmallRng, seq::IndexedRandom as _};
use rand_distr::{Binomial, Distribution as _, Poisson};

use crate::graph::{DirectGraph, EntangleGraph};

/// This is the base unit of time in the system, in milliseconds.
pub type TimeMillis = u64;

/// Expected number of events per timestep.
pub type Rate = f64;

#[pyo3::pyclass(name = "Simulation")]
#[derive(Clone, Debug)]
pub struct Simulation {
    rng: SmallRng,
    params: SimParams,
    direct_graph: DirectGraph,
    entangle_graph: EntangleGraph,
    current_time: TimeMillis,
}

impl Simulation {
    pub fn run(&mut self, duration: TimeMillis) {
        let num_ticks = duration.checked_div(self.params.tick_interval).unwrap();

        for _ in 0..num_ticks {
            self.tick(self.params.tick_interval);
        }
    }

    pub fn entangle_graph(&self) -> &EntangleGraph {
        &self.entangle_graph
    }

    pub fn params(&self) -> &SimParams {
        &self.params
    }

    pub fn time(&self) -> TimeMillis {
        self.current_time
    }

    fn tick(&mut self, interval: TimeMillis) {
        self.tick_entangle_graph(interval);
        self.tick_direct_graph(interval);
        self.current_time += interval;
    }

    pub fn max_flow_stats(&self, n_samples: u64) -> (f64, f64) {
        // TODO: we should figure out how to use the deterministic rng here.
        let mut rng = rand::rng();
        let mut max_flows: Vec<u64> = Vec::with_capacity(n_samples as usize);
        for _ in 0..n_samples {
            let node_a = rng.random_range(0..self.direct_graph.num_nodes());
            let node_b = loop {
                let candidate = rng.random_range(0..self.direct_graph.num_nodes());
                if candidate != node_a {
                    break candidate;
                }
            };

            let node_a = NodeIndex::new(node_a);
            let node_b = NodeIndex::new(node_b);

            max_flows.push(self.entangle_graph.max_flow(node_a, node_b));
        }

        // Calculate mean
        let sum: u64 = max_flows.iter().sum();
        let avg = sum as f64 / max_flows.len() as f64;

        // Calculate standard deviation
        let variance = max_flows
            .iter()
            .map(|&value| {
                let diff = value as f64 - avg;
                diff * diff
            })
            .sum::<f64>()
            / max_flows.len() as f64;

        let std = variance.sqrt();

        (avg, std)
    }
    
    pub fn generalized_max_flow_stats(&self, n_samples: u64) -> (f64, f64) {
        // TODO: we should figure out how to use the deterministic rng here.
        let mut rng = rand::rng();
        let mut max_flows: Vec<f64> = Vec::with_capacity(n_samples as usize);
        for _ in 0..n_samples {
            let node_a = rng.random_range(0..self.direct_graph.num_nodes());
            let node_b = loop {
                let candidate = rng.random_range(0..self.direct_graph.num_nodes());
                if candidate != node_a {
                    break candidate;
                }
            };

            let node_a = NodeIndex::new(node_a);
            let node_b = NodeIndex::new(node_b);

            max_flows.push(self.entangle_graph.generalized_max_flow(node_a, node_b));
        }

        // Calculate mean
        let sum: f64 = max_flows.iter().sum();
        let avg = sum / max_flows.len() as f64;

        // Calculate standard deviation
        let variance = max_flows
            .iter()
            .map(|&value| {
                let diff = value - avg;
                diff * diff
            })
            .sum::<f64>()
            / max_flows.len() as f64;

        let std = variance.sqrt();

        (avg, std)
    }

    pub fn tick_direct_graph(&mut self, interval: TimeMillis) {
        for (node_a, node_b, direct_edge) in self.direct_graph.edges() {
            let num_pairs_expected = direct_edge.link_rate * interval as f64;
            let num_pairs_generated = Poisson::new(num_pairs_expected)
                .unwrap()
                .sample(&mut self.rng);
            self.entangle_graph
                .add_capacity(num_pairs_generated as u64, node_a, node_b);
        }
    }

    pub fn tick_entangle_graph(&mut self, interval: TimeMillis) {
        // TODO(opt): Maybe better to handle decoherences as needed instead of accounting for
        // everything on every time step.
        for (node_a, node_b, _edge_data) in self.entangle_graph.clone().edges() {
            // Account for decoherence.
            let p_single_decay = 1.0 - (-self.params.decoherence_factor * interval as f64).exp();
            let link_capacity = self.entangle_graph.get_capacity(node_a, node_b);
            let dist = Binomial::new(link_capacity, p_single_decay).unwrap();
            let num_decoherences: u64 = dist.sample(&mut self.rng);

            self.entangle_graph
                .sub_capacity(num_decoherences, node_a, node_b);
        }

        for node in self.entangle_graph().nodes() {
            let neighbors: Vec<NodeIndex> = self.entangle_graph.neighbors_of(node);
            if neighbors.len() < 2 {
                continue;
            }

            // Choose two neighbors and swap swap_fraction of the lower of their channel
            // capacities. We're going to do this a number of times proportional to the
            // number of neighbors the node has.
            for _ in 0..neighbors.len() {
                let two_neighbors: Vec<_> = neighbors.choose_multiple(&mut self.rng, 2).collect();
                let (node_a, node_b) = (*two_neighbors[0], *two_neighbors[1]);

                let cap_a = self.entangle_graph.get_capacity(node, node_a);
                let cap_b = self.entangle_graph.get_capacity(node, node_b);
                let swap_number =
                    (std::cmp::min(cap_a, cap_b) as f64 * self.params.swap_fraction) as u64;

                self.entangle_graph.sub_capacity(swap_number, node, node_a);
                self.entangle_graph.sub_capacity(swap_number, node, node_b);
                
                // Add capacity with the default swap decay
                self.entangle_graph
                    .add_capacity(swap_number, node_a, node_b);
                
                // Set the decay factor for the newly created edge
                if let Some(edge_idx) = self.entangle_graph.graph.find_edge(node_a, node_b) {
                    let edge = self.entangle_graph.graph.edge_weight_mut(edge_idx).unwrap();
                    edge.swap_decay = self.params.default_swap_decay;
                }
            }
        }
    }

    pub fn new(params: SimParams) -> Self {
        let mut rng = SmallRng::from_seed(params.rng_seed);
        let direct_graph = DirectGraph::random(&params, &mut rng);
        let entangle_graph = EntangleGraph::with_node_capacity(params.num_nodes);
        Self {
            rng,
            params,
            direct_graph,
            entangle_graph,
            current_time: 0,
        }
    }
}

#[pyo3::pyclass(name = "SimParams", eq, get_all, set_all, str)]
#[derive(Clone, Debug, PartialEq)]
pub struct SimParams {
    /// This describes what fraction of an entangled channel capacity will
    /// be spent to feed new edges in the entanglement graph.
    pub swap_fraction: Rate,

    /// How far to advance in time per simulation step.
    pub tick_interval: TimeMillis,

    /// Controls how fast the qubits decohere.
    pub decoherence_factor: f64,

    /// How fast the direct connections generate entangled pairs.
    pub link_rate: Rate,

    /// How many nodes to simulate.
    pub num_nodes: usize,

    /// Probability that two nodes will be connected in the direct graph.
    /// This applies only if the graph is generated randomly.
    pub direct_edge_density: f64,

    /// Seed to make the random number generation deterministic.
    pub rng_seed: [u8; 32],
    
    /// Default decay factor for swapped entanglement (0.0 to 1.0)
    /// Used in generalized max flow calculations
    pub default_swap_decay: f64,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            swap_fraction: 0.2,
            tick_interval: 1,
            decoherence_factor: 0.8,
            link_rate: 10.0,
            num_nodes: 50,
            direct_edge_density: 0.5,
            rng_seed: [42; 32],
            default_swap_decay: 0.9,
        }
    }
}

impl std::fmt::Display for SimParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self, f)
    }
}
