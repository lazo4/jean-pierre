use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod pyhebb {
    use pyo3::prelude::*;

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: u64, b: u64) -> PyResult<String> {
        Ok(superhebb_core::add(a, b).to_string())
    }
}
