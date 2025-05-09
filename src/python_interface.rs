use std::collections::HashMap;

use crate::simulation::{self, SimParams, Simulation};
use petgraph::graph::NodeIndex;
use pyo3::{prelude::*, types::PyDict};

#[pymodule(name = "qnet")]
fn qnet_python_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<simulation::SimParams>()?;
    m.add_class::<simulation::Simulation>()?;
    Ok(())
}

#[pymethods]
impl Simulation {
    #[pyo3(name = "run")]
    fn py_run(&mut self, duration_millis: u64) {
        self.run(duration_millis);
    }

    #[pyo3(name = "edge_capacities")]
    fn py_edge_capacities(&self) -> Vec<HashMap<String, u64>> {
        let mut result = Vec::new();
        for a in 0..self.params().num_nodes {
            for b in (a + 1)..self.params().num_nodes {
                let capacity = self
                    .entangle_graph()
                    .get_capacity(NodeIndex::new(a), NodeIndex::new(b));
                let mut entry = HashMap::new();
                entry.insert("node_a".to_string(), a as u64);
                entry.insert("node_b".to_string(), b as u64);
                entry.insert("link_capacity".to_string(), capacity);
                result.push(entry);
            }
        }
        result
    }

    // TODO: It's probably better to just give Python access to the max flow
    // routines and figure out all the stats on the Python side. There are
    // going to be too many analyses we want to experiment with to keep
    // it all on the Rust side.
    /// Returns (avg, std)
    #[pyo3(name = "max_flow_stats")]
    fn py_max_flow_stats(&self, n_samples: u64) -> (f64, f64) {
        self.max_flow_stats(n_samples)
    }
    
    /// Returns (avg, std) for generalized max flow
    #[pyo3(name = "generalized_max_flow_stats")]
    fn py_generalized_max_flow_stats(&self, n_samples: u64) -> (f64, f64) {
        self.generalized_max_flow_stats(n_samples)
    }

    #[pyo3(name = "time")]
    fn py_time(&self) -> u64 {
        self.time()
    }
}

#[pymethods]
impl SimParams {
    fn build(&self) -> PyResult<Simulation> {
        let sim = Simulation::new(self.clone());
        Ok(sim)
    }

    #[new]
    #[pyo3(signature = (**py_kwargs))]
    fn new_py(py_kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let mut params = Self::default();

        if let Some(kwargs) = py_kwargs {
            if let Some(val) = kwargs.get_item("swap_fraction")? {
                params.swap_fraction = val.extract()?;
            }
            if let Some(val) = kwargs.get_item("tick_interval")? {
                params.tick_interval = val.extract()?;
            }
            if let Some(val) = kwargs.get_item("decoherence_factor")? {
                params.decoherence_factor = val.extract()?;
            }
            if let Some(val) = kwargs.get_item("link_rate")? {
                params.link_rate = val.extract()?;
            }
            if let Some(val) = kwargs.get_item("num_nodes")? {
                params.num_nodes = val.extract()?;
            }
            if let Some(val) = kwargs.get_item("direct_edge_density")? {
                params.direct_edge_density = val.extract()?;
            }
        }

        Ok(params)
    }
}
