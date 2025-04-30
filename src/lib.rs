use pyo3::{PyResult, Python, pymodule, types::PyModule};

pub mod graph;
pub mod simulation;

#[pymodule]
fn qnet_sim(_py: Python, m: &PyModule) -> PyResult<()> {
    python_interface::init_module(m)?;
    Ok(())
}

mod python_interface;
