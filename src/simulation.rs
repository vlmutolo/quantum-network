use rand::{Rng, seq::IndexedRandom as _};
use rand_distr::{Binomial, Distribution as _, Poisson};

use crate::graph::{DirectGraph, EntangleGraph, NodeId};

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
        self.tick_entangle_graph(interval, rng);
        self.tick_direct_graph(interval, rng);
        self.current_time += interval;

        let max_flow_avg = self.max_flow_avg(rng);

        // TODO(opt): we always create these logs even though they may
        // not be used. Can we do something smarter?
        SimLog {
            entangle_graph: self.entangle_graph.clone(),
            time: self.current_time,
            max_flow_avg,
        }
    }

    fn max_flow_avg<R: Rng>(&mut self, rng: &mut R) -> f64 {
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
        max_flow_sum as f64 / MAX_FLOW_COUNT as f64
    }

    pub fn tick_direct_graph<R: Rng>(&mut self, interval: TimeMillis, rng: &mut R) {
        for (node_a, node_b, direct_edge) in self.direct_graph.edges() {
            let num_pairs_expected = direct_edge.link_rate * interval as f64;
            let num_pairs_generated = Poisson::new(num_pairs_expected).unwrap().sample(rng);
            self.entangle_graph
                .add_capacity(num_pairs_generated as u64, node_a, node_b);
        }
    }

    pub fn tick_entangle_graph<R: Rng>(&mut self, interval: TimeMillis, rng: &mut R) {
        // TODO(opt): Maybe better to handle decoherences as needed instead of accounting for
        // everything on every time step.
        for (node_a, node_b, _edge_data) in self.entangle_graph.clone().edges() {
            // Account for decoherence.
            let p_single_decay = 1.0 - (-self.params.decoherence_factor * interval as f64).exp();
            let link_capacity = self.entangle_graph.get_capacity(node_a, node_b);
            let dist = Binomial::new(link_capacity, p_single_decay).unwrap();
            let num_decoherences: u64 = dist.sample(rng);

            self.entangle_graph
                .sub_capacity(num_decoherences, node_a, node_b);
        }

        for node in self.nodes() {
            let neighbors: Vec<NodeId> = self.entangle_graph.neighbors_of(node);
            if neighbors.len() < 2 {
                continue;
            }

            // Choose two neighbors and swap swap_fraction of the lower of their channel
            // capacities. We're going to do this a number of times proportional to the
            // number of neighbors the node has.
            for _ in 0..neighbors.len() {
                let two_neighbors: Vec<_> = neighbors.choose_multiple(rng, 2).collect();
                let (node_a, node_b) = (*two_neighbors[0], *two_neighbors[1]);

                let cap_a = self.entangle_graph.get_capacity(node, node_a);
                let cap_b = self.entangle_graph.get_capacity(node, node_b);
                let swap_number =
                    (std::cmp::min(cap_a, cap_b) as f64 * self.params.swap_fraction) as u64;

                self.entangle_graph.sub_capacity(swap_number, node, node_a);
                self.entangle_graph.sub_capacity(swap_number, node, node_b);
                self.entangle_graph
                    .add_capacity(swap_number, node_a, node_b);
            }
        }
    }

    fn nodes(&self) -> impl Iterator<Item = NodeId> + 'static {
        0..self.params.num_nodes
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

    /// Controls how fast the qubits decohere.
    pub decoherence_factor: f64,

    /// How fast the direct connections generate entangled pairs.
    pub link_rate: Rate,

    /// How many nodes to simulate.
    pub num_nodes: u64,

    /// Probability that two nodes will be connected in the direct graph.
    /// This applies only if the graph is generated randomly.
    pub direct_edge_density: f64,
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
