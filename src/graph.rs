use petgraph::{
    Undirected,
    graph::{DiGraph, Graph, NodeIndex, UnGraph},
    visit::EdgeRef,
};
use rand::Rng;

use crate::simulation::{Rate, SimParams};

/// Holds the connections between nodes that are physically linked and
/// can generate entangled pairs on-demand.
#[derive(Clone, Debug)]
pub struct DirectGraph {
    graph: UnGraph<(), DirectEdge>,
}

impl Default for DirectGraph {
    fn default() -> Self {
        Self {
            graph: UnGraph::new_undirected(),
        }
    }
}

impl Default for EntangleEdge {
    fn default() -> Self {
        Self {
            link_capacity: 0,
            swap_decay: 1.0, // Default: no loss in flow (traditional max flow)
        }
    }
}

impl EntangleEdge {
    pub fn with_capacity(capacity: u64) -> Self {
        Self {
            link_capacity: capacity,
            swap_decay: 1.0,
        }
    }

    pub fn with_capacity_and_decay(capacity: u64, decay: f64) -> Self {
        Self {
            link_capacity: capacity,
            swap_decay: decay,
        }
    }
}

impl DirectGraph {
    pub fn num_nodes(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edges(&self) -> impl Iterator<Item = (NodeIndex, NodeIndex, &DirectEdge)> + '_ {
        self.graph.edge_references().filter_map(|edge_ref| {
            let node_a = edge_ref.source();
            let node_b = edge_ref.target();
            Some((node_a, node_b, edge_ref.weight()))
        })
    }

    /// Generate a new DirectGraph with random connections.
    pub fn random<R: Rng>(params: &SimParams, rng: &mut R) -> Self {
        let mut graph = UnGraph::with_capacity(
            params.num_nodes as usize,
            (params.num_nodes * params.num_nodes) as usize,
        );

        // First add all nodes
        for _ in 0..params.num_nodes {
            let _idx = graph.add_node(());
        }

        // Then add random edges
        for i in 0..params.num_nodes {
            for j in 0..params.num_nodes {
                if i == j {
                    continue;
                }

                if rng.random::<f64>() < params.direct_edge_density {
                    let source_idx = NodeIndex::new(i as usize);
                    let target_idx = NodeIndex::new(j as usize);
                    graph.add_edge(
                        source_idx,
                        target_idx,
                        DirectEdge {
                            link_rate: params.link_rate,
                        },
                    );
                }
            }
        }

        Self { graph }
    }
}

#[derive(Clone, Debug)]
pub struct DirectEdge {
    /// How many Bell pairs can be (successfully) exchanged per time step
    pub link_rate: Rate,
}

#[derive(Clone, Debug)]
pub struct EntangleGraph {
    pub graph: Graph<(), EntangleEdge, Undirected>,
}

impl EntangleGraph {
    pub fn add_capacity(&mut self, capacity_amt: u64, node_a: NodeIndex, node_b: NodeIndex) {
        assert_ne!(node_a, node_b);

        match self.graph.find_edge(node_a, node_b) {
            // Check if edge already exists
            Some(edge_idx) => {
                let edge = self.graph.edge_weight_mut(edge_idx).unwrap();
                edge.link_capacity = edge.link_capacity.checked_add(capacity_amt).unwrap();
            }

            // Create new edge
            None => {
                self.graph.add_edge(
                    node_a,
                    node_b,
                    EntangleEdge {
                        link_capacity: capacity_amt,
                        swap_decay: 1.0, // Default value, no decay
                    },
                );
            }
        }
    }

    /// PANICS: This will panic if you try to subtract more than a connection has.
    pub fn sub_capacity(&mut self, capacity_amt: u64, node_a: NodeIndex, node_b: NodeIndex) {
        assert_ne!(node_a, node_b);

        if let Some(edge_idx) = self.graph.find_edge(node_a, node_b) {
            let edge = self.graph.edge_weight_mut(edge_idx).unwrap();
            edge.link_capacity = edge.link_capacity.checked_sub(capacity_amt).unwrap();
        }

        // TODO(opt): Do we want to remove edges if they reach zero capacity? That
        // might save us some memory, but it will also possibly require moving to
        // another graph type because removing indices makes them unstable. Does
        // this matter for us? Likely only matters if we're storing node or edge
        // indices somewhere, which I don't think we are.
    }

    pub fn get_capacity(&self, node_a: NodeIndex, node_b: NodeIndex) -> u64 {
        assert_ne!(node_a, node_b);

        if let Some(edge_idx) = self.graph.find_edge(node_a, node_b) {
            self.graph.edge_weight(edge_idx).unwrap().link_capacity
        } else {
            0
        }
    }

    pub fn edges(&self) -> impl Iterator<Item = (NodeIndex, NodeIndex, &EntangleEdge)> + '_ {
        self.graph.edge_references().map(|edge_ref| {
            let source_idx = edge_ref.source();
            let target_idx = edge_ref.target();
            (source_idx, target_idx, edge_ref.weight())
        })
    }

    pub fn neighbors_of(&self, node: NodeIndex) -> Vec<NodeIndex> {
        self.graph.neighbors(node).collect()
    }

    pub fn max_flow(&self, src: NodeIndex, dst: NodeIndex) -> u64 {
        // Create a directed graph from the undirected EntangleGraph.
        // Each undirected edge becomes two directed edges with the same capacity.
        let mut network: DiGraph<(), u64> = DiGraph::new();

        for node_idx in self.graph.node_indices() {
            network.add_node(self.graph[node_idx].clone());
        }

        // Add two directed edges (one in each direction) for each undirected edge
        // in the original graph.
        for edge_ref in self.graph.edge_references() {
            let u = edge_ref.source();
            let v = edge_ref.target();
            let capacity = edge_ref.weight().link_capacity;
            network.add_edge(u, v, capacity);
            network.add_edge(v, u, capacity);
        }

        let (max_flow, _edge_flows) = petgraph::algo::ford_fulkerson(&network, src, dst);

        max_flow
    }

    pub fn nodes(&self) -> impl Iterator<Item = NodeIndex> + 'static {
        self.graph.node_indices()
    }

    pub fn with_node_capacity(num_nodes: usize) -> Self {
        let mut graph = UnGraph::new_undirected();
        for _ in 0..num_nodes {
            let _idx = graph.add_node(());
        }
        Self { graph }
    }

    /// Calculates the generalized maximum flow from source to sink
    /// using linear programming, accounting for decay factors on edges
    pub fn generalized_max_flow(&self, src: NodeIndex, dst: NodeIndex) -> f64 {
        use good_lp::{
            Expression, Solution, SolverModel, constraint, solvers::microlp::microlp, variable,
            variables,
        };
        use std::collections::HashMap;

        // Number of edges in the graph (each undirected edge becomes two directed edges)
        let edge_count = self.graph.edge_count() * 2;

        // Create variables for each directed edge: flow amount on that edge.
        // Index them as (from_node, to_node).
        let mut edge_vars = Vec::with_capacity(edge_count);
        let mut edge_indices = HashMap::new();

        let mut variables = variables!();

        // Add directed edges from the undirected graph, with flow variables
        for edge_ref in self.graph.edge_references() {
            let u = edge_ref.source();
            let v = edge_ref.target();
            let capacity = edge_ref.weight().link_capacity as f64;
            let decay = edge_ref.weight().swap_decay;

            // Edge from u to v
            let var_u_to_v = variables.add(variable().min(0.0).max(capacity));
            edge_indices.insert((u.index(), v.index()), var_u_to_v);
            edge_vars.push((u.index(), v.index(), var_u_to_v, decay));

            // Edge from v to u
            let var_v_to_u = variables.add(variable().min(0.0).max(capacity));
            edge_indices.insert((v.index(), u.index()), var_v_to_u);
            edge_vars.push((v.index(), u.index(), var_v_to_u, decay));
        }

        // Create objective: maximize the flow out of the source node
        let objective: Expression = edge_vars
            .iter()
            .filter(|(from, _, _, _)| *from == src.index())
            .fold(Expression::from(0.0), |acc, (_, _, var_idx, _)| {
                acc + *var_idx
            });

        // Start building the problem
        let mut problem = variables.maximise(objective).using(microlp);

        // Flow conservation constraints for each node (except source and sink)
        for node in self.graph.node_indices() {
            if node == src || node == dst {
                continue;
            }

            // For each node, incoming flow * decay = outgoing flow
            let mut outgoing_flow = Expression::from(0.0);
            for neigh in self.graph.neighbors(node) {
                if let Some(&var_idx) = edge_indices.get(&(node.index(), neigh.index())) {
                    outgoing_flow = outgoing_flow + var_idx;
                }
            }

            let mut incoming_flow = Expression::from(0.0);
            for (_from, to, var_idx, decay) in &edge_vars {
                if *to == node.index() {
                    incoming_flow = incoming_flow + (*decay * *var_idx);
                }
            }

            // Add the flow conservation constraint: sum(incoming*decay) = sum(outgoing)
            problem = problem.with(constraint!(incoming_flow == outgoing_flow));
        }

        // Solve with microlp solver (pure Rust)
        match problem.solve() {
            Ok(solution) => {
                // Calculate total outflow from source
                let mut total_flow = 0.0;
                for (from, _, var_idx, _) in &edge_vars {
                    if *from == src.index() {
                        total_flow += solution.value(*var_idx);
                    }
                }

                total_flow
            }
            Err(_) => 0.0, // Error occurred
        }
    }
}

#[derive(Clone, Debug)]
pub struct EntangleEdge {
    /// How many Bell pairs have been (successfully) pre-exchanged and
    /// are now in storage waiting to execute teleportation
    pub link_capacity: u64,
    /// Decay factor representing the loss when swapping through this edge
    /// A value between 0 and 1 representing the fraction of entanglement preserved
    pub swap_decay: f64,
}

#[cfg(test)]
mod tests {
    use super::{DirectGraph, EntangleGraph, SimParams};
    use petgraph::graph::NodeIndex;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_entangle_graph_operations() {
        let mut graph = EntangleGraph::with_node_capacity(2);

        assert_eq!(0, graph.get_capacity(0.into(), 1.into()));
        assert_eq!(0, graph.get_capacity(1.into(), 0.into()));

        graph.add_capacity(1, 0.into(), 1.into());
        assert_eq!(1, graph.get_capacity(0.into(), 1.into()));
        assert_eq!(1, graph.get_capacity(1.into(), 0.into()));

        graph.sub_capacity(1, 1.into(), 0.into());
        assert_eq!(0, graph.get_capacity(0.into(), 1.into()));
        assert_eq!(0, graph.get_capacity(1.into(), 0.into()));
    }

    #[test]
    fn test_entangle_graph_multiple_edges() {
        let mut graph = EntangleGraph::with_node_capacity(5);

        // Add multiple edges
        graph.add_capacity(2, 0.into(), 1.into());
        graph.add_capacity(3, 1.into(), 2.into());
        graph.add_capacity(4, 0.into(), 2.into());

        // Check capacities
        assert_eq!(2, graph.get_capacity(0.into(), 1.into()));
        assert_eq!(3, graph.get_capacity(1.into(), 2.into()));
        assert_eq!(4, graph.get_capacity(0.into(), 2.into()));

        // Check neighbor lists
        let neighbors_0 = graph.neighbors_of(0.into());
        assert_eq!(2, neighbors_0.len());
        assert!(neighbors_0.contains(&1.into()));
        assert!(neighbors_0.contains(&2.into()));

        let neighbors_1 = graph.neighbors_of(1.into());
        assert_eq!(2, neighbors_1.len());
        assert!(neighbors_1.contains(&0.into()));
        assert!(neighbors_1.contains(&2.into()));
    }

    #[test]
    fn test_direct_graph_creation() {
        // Create params for testing
        let params = SimParams {
            num_nodes: 3,
            direct_edge_density: 1.0, // Ensure all edges are created
            link_rate: 10.0,
            ..Default::default()
        };

        // Create a deterministic RNG for testing
        let mut rng = StdRng::seed_from_u64(42);

        // Create the graph
        let graph = DirectGraph::random(&params, &mut rng);

        // Check number of nodes
        assert_eq!(3, graph.num_nodes());

        // Collect all edges to check
        let edges: Vec<_> = graph.edges().collect();

        // Should have 6 edges (3 nodes, each connected to 2 others)
        assert_eq!(6, edges.len());

        // Check that edges have the expected link_rate
        for (_, _, edge) in &edges {
            assert_eq!(10.0, edge.link_rate);
        }
    }

    #[test]
    fn test_generalized_max_flow() {
        // Create a simple graph for testing
        let mut graph = EntangleGraph::with_node_capacity(4);

        // Source is node 0, sink is node 3
        let src = NodeIndex::new(0);
        let dst = NodeIndex::new(3);

        // Add edges with capacities and decay factors
        // Path 1: 0 -> 1 -> 3
        graph.add_capacity(10, src, NodeIndex::new(1));
        if let Some(edge_idx) = graph.graph.find_edge(src, NodeIndex::new(1)) {
            let edge = graph.graph.edge_weight_mut(edge_idx).unwrap();
            edge.swap_decay = 0.8; // 80% efficiency
        }

        graph.add_capacity(10, NodeIndex::new(1), dst);
        if let Some(edge_idx) = graph.graph.find_edge(NodeIndex::new(1), dst) {
            let edge = graph.graph.edge_weight_mut(edge_idx).unwrap();
            edge.swap_decay = 0.8; // 80% efficiency
        }

        // Path 2: 0 -> 2 -> 3
        graph.add_capacity(10, src, NodeIndex::new(2));
        if let Some(edge_idx) = graph.graph.find_edge(src, NodeIndex::new(2)) {
            let edge = graph.graph.edge_weight_mut(edge_idx).unwrap();
            edge.swap_decay = 0.7; // 70% efficiency
        }

        graph.add_capacity(10, NodeIndex::new(2), dst);
        if let Some(edge_idx) = graph.graph.find_edge(NodeIndex::new(2), dst) {
            let edge = graph.graph.edge_weight_mut(edge_idx).unwrap();
            edge.swap_decay = 0.7; // 70% efficiency
        }

        // Regular max flow should be 20 (10 through each path)
        assert_eq!(20, graph.max_flow(src, dst));

        // Generalized max flow with decay should be less
        // Path 1: 10 * 0.8 * 0.8 = 6.4
        // Path 2: 10 * 0.7 * 0.7 = 4.9
        // Total: 11.3
        let gen_max_flow = graph.generalized_max_flow(src, dst);
        assert!((gen_max_flow - 11.3).abs() < 0.1); // Allow for small floating point differences
    }
}
