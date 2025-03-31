use simulation::{SimParams, simulate};

use crate::network::NetworkState;

fn main() {
    let network = NetworkState::init();
    let params = SimParams {
        indirect_rate: todo!(),
    };

    let logs = simulate(network, params);
}

pub mod simulation {
    use priority_queue::PriorityQueue;

    use crate::{network::NetworkState, node::NodeId};

    /// This is the base unit of time in the system, in nanoseconds.
    pub type Time = u64;

    /// Expected number of events per nanosecond.
    pub type Rate = f64;

    struct Simulation {
        params: SimParams,
        log: SimLog,
        timer: PriorityQueue<Event, Time>,
        node_last_update_times: Vec<Time>,
        network: NetworkState,
    }

    impl Simulation {
        pub fn new(params: SimParams, network: NetworkState) -> Self {
            let log = SimLog {};
            let timer = PriorityQueue::new();
            let node_last_update_times =
                Vec::from_iter(std::iter::repeat_n(0, network.len_nodes()));

            Self {
                params,
                log,
                timer,
                node_last_update_times,
                network,
            }
        }

        pub fn run(&mut self, stop_time: Option<Time>) {
            todo!()
        }

        fn handle_event(&mut self, event: Event) {
            match event {
                Event::EntanglementSwap(exchange_event) => todo!(),
                Event::GeneratePair(generate_pair_event) => todo!(),
            }
        }
    }

    pub struct SimParams {
        /// Effective channel capacity for indirect connections. This basically
        /// accounts for how many entanglement swaps per second a node can perform.
        pub indirect_rate: Rate,
    }

    /// Summary stats for a snapshot of the network state
    pub struct SimLog {}

    #[derive(Clone, Debug, Hash, PartialEq, Eq)]
    enum Event {
        /// Two connected peers swap their entanglement.
        EntanglementSwap(EntanglementSwapEvent),

        /// Two peers jk
        GeneratePair(GeneratePairEvent),
    }

    /// A node randomly chooses one of its direct
    /// peers to generate a new Bell pair with.
    #[derive(Clone, Debug, Hash, PartialEq, Eq)]
    struct GeneratePairEvent {
        node: NodeId,
    }

    /// A node randomly chooses two of its peers (either direct or indirect)
    /// to connect via entanglement swapping. The node loses its connection
    /// (or decreases the weight of its connections) in the process.
    #[derive(Clone, Debug, Hash, PartialEq, Eq)]
    struct EntanglementSwapEvent {
        node: NodeId,
    }
}

pub mod network {
    use crate::node::{NodeId, NodeState};

    pub struct NetworkState {
        nodes: Vec<NodeState>,
    }

    impl NetworkState {
        pub fn init() -> Self {
            todo!()
        }

        pub fn len_nodes(&self) -> usize {
            self.nodes.len()
        }

        pub fn node_mut(&mut self, node_id: NodeId) -> &mut NodeState {
            self.nodes.get_mut(node_id).unwrap()
        }
    }
}

pub mod node {
    use rand::Rng;
    use rand_distr::{Distribution as _, Poisson};

    use crate::simulation::{Rate, Time};

    pub type NodeId = usize;

    pub struct NodeState {
        direct_peers: Vec<DirectPeer>,
        indirect_peers: Vec<IndirectPeer>,
        last_updated: Time,
    }

    impl NodeState {
        /// Account for decoherence by removing a certain number of qubits
        /// randomly based on sampling from Poisson distribution.
        fn update<R: Rng>(&mut self, time_delta: Time, decoherence_rate: Rate, rng: &mut R) {
            let expected_number = decoherence_rate * time_delta as f64;
            let dist = Poisson::new(expected_number).unwrap();
            let num_decoherences: f64 = dist.sample(rng);
        }
    }

    struct DirectPeer {
        peer_id: NodeId,

        /// How many Bell pairs can be (successfully) exchanged per time step
        link_capacity: u64,
    }

    struct IndirectPeer {
        peer_id: usize,

        /// How many Bell pairs have been (successfully) pre-exchanged and
        /// are now in storage waiting to execute teleportation
        link_capacity: u64,
    }
}
