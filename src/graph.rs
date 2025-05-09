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
    graph: Graph<(), EntangleEdge, Undirected>,
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
}

#[derive(Clone, Debug)]
pub struct EntangleEdge {
    /// How many Bell pairs have been (successfully) pre-exchanged and
    /// are now in storage waiting to execute teleportation
    pub link_capacity: u64,
}

#[cfg(test)]
mod tests {
    use super::{DirectGraph, EntangleGraph, SimParams};
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
}
