use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod pyhebb {
    use pyo3::{prelude::*, types::PyBytes};

    #[pyclass]
    struct VisionSNN {
        inner: superhebb_core::VisionSNN,
    }

    #[pymethods]
    impl VisionSNN {
        #[new]
        fn new() -> Self {
            Self {
                inner: superhebb_core::VisionSNN::new(),
            }
        }

        fn run_micro_ticks<'py>(
            &mut self,
            py: Python<'py>,
            input_frame: &[u8],
            micro_ticks: u32,
        ) -> PyResult<Bound<'py, PyBytes>> {
            let size = 1024 * 1024; // 1 MB buffer

            // 1. Allocate memory directly inside Python so Rust can write into it safely
            let py_bytes = PyBytes::new_with(py, size, |buffer| {
                // 2. Call your inner logic, passing the uninitialized/empty buffer to be filled
                // Note: This requires tweaking your inner function to accept a &mut [u8] slice
                self.inner.run_micro_ticks(input_frame, micro_ticks, buffer);
                Ok(())
            })?;

            Ok(py_bytes)
        }
    }
}
