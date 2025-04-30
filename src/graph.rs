use std::collections::{HashMap, HashSet, VecDeque};

use rand::Rng;

use crate::simulation::{Rate, SimParams};

pub type NodeId = u64;

/// Holds the connections between nodes that are physically linked and
/// can generate entangled pairs on-demand.
#[derive(Clone, Debug, Default)]
pub struct DirectGraph {
    num_nodes: u64,
    map: HashMap<(NodeId, NodeId), DirectEdge>,
}

impl DirectGraph {
    pub fn num_nodes(&self) -> u64 {
        self.num_nodes
    }

    pub fn edges(&self) -> impl Iterator<Item = (NodeId, NodeId, &DirectEdge)> + '_ {
        self.map.iter().map(|((a, b), edge)| (*a, *b, edge))
    }

    /// Generate a new DirectGraph with random connections.
    pub fn random<R: Rng>(params: &SimParams, rng: &mut R) -> Self {
        let mut neighbor_map = HashMap::new();

        for i in 0..params.num_nodes {
            for j in 0..params.num_nodes {
                if i == j {
                    continue;
                }

                if rng.random_bool(params.direct_edge_density) {
                    let old_val = neighbor_map.insert(
                        (i, j),
                        DirectEdge {
                            link_rate: params.link_rate,
                        },
                    );
                    assert!(old_val.is_none());
                }
            }
        }

        Self {
            map: neighbor_map,
            num_nodes: params.num_nodes,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DirectEdge {
    /// How many Bell pairs can be (successfully) exchanged per time step
    pub link_rate: Rate,
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
    pub fn add_capacity(&mut self, capacity_amt: u64, node_a: NodeId, node_b: NodeId) {
        let key = edge_key(node_a, node_b);
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
    pub fn sub_capacity(&mut self, capacity_amt: u64, node_a: NodeId, node_b: NodeId) {
        let key = edge_key(node_a, node_b);
        let entry = self.edges.get_mut(&key).unwrap();
        entry.link_capacity = entry.link_capacity.checked_sub(capacity_amt).unwrap();

        if entry.link_capacity == 0 {
            self.neighbors.get_mut(&node_a).unwrap().remove(&node_b);
            self.neighbors.get_mut(&node_b).unwrap().remove(&node_a);
        }
    }

    pub fn get_capacity(&self, node_a: NodeId, node_b: NodeId) -> u64 {
        let key = edge_key(node_a, node_b);
        match self.edges.get(&key) {
            Some(entangle_edge) => entangle_edge.link_capacity,
            None => 0,
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

    pub fn neighbors_of(&self, node: NodeId) -> Vec<NodeId> {
        self.neighbors
            .get(&node)
            .map(|neighbors| neighbors.iter().copied().collect())
            .unwrap_or_default()
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
    fn test_entangle_graph_operations() {
        let mut graph = EntangleGraph::default();

        assert_eq!(0, graph.get_capacity(0, 1));
        assert_eq!(0, graph.get_capacity(1, 0));

        graph.add_capacity(1, 0, 1);
        assert_eq!(1, graph.get_capacity(0, 1));
        assert_eq!(1, graph.get_capacity(1, 0));

        graph.sub_capacity(1, 1, 0);
        assert_eq!(0, graph.get_capacity(0, 1));
        assert_eq!(0, graph.get_capacity(1, 0));
    }
}
