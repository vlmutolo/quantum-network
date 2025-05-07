use preshared_entanglement_sim::simulation::{SimParams, Simulation};

fn main() -> anyhow::Result<()> {
    let params = SimParams {
        swap_fraction: 0.2,
        tick_interval: 1,
        decoherence_factor: 0.1,
        link_rate: 5.0,
        num_nodes: 50,
        direct_edge_density: 0.5,
        rng_seed: [42; 32],
    };

    let mut simulation = Simulation::new(params);

    simulation.run(500);

    Ok(())
}
