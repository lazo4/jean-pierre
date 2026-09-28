use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod pyjeanpierre {
    use pyo3::{prelude::*, types::PyBytes};

    #[pyclass]
    struct VisionSNN {
        inner: jean_pierre_core::VisionSNN,
    }

    #[pymethods]
    impl VisionSNN {
        #[new]
        fn new() -> Self {
            Self {
                inner: jean_pierre_core::VisionSNN::new(),
            }
        }

        fn run_micro_ticks<'py>(
            &mut self,
            py: Python<'py>,
            input_frame: &[u8],
            micro_ticks: u32,
        ) -> PyResult<(
            Bound<'py, PyBytes>,
            Bound<'py, PyBytes>,
            Bound<'py, PyBytes>,
        )> {
            // 1. Allocate temporary vectors
            let mut retina_out = vec![0u8; 1024 * 1024];
            let mut layer1_out = vec![0u8; 1024 * 1024];
            let mut layer1_weights_out = vec![0u8; 5 * 5];

            // 2. Call your original function
            self.inner.run_micro_ticks(
                input_frame,
                micro_ticks,
                retina_out.as_mut(),
                layer1_out.as_mut(),
                layer1_weights_out.as_mut(),
            );

            // 3. Convert the vectors into Python bytes objects (copies data)
            let retina_bytes = PyBytes::new(py, &retina_out);
            let layer1_bytes = PyBytes::new(py, &layer1_out);
            let layer1_weights_bytes = PyBytes::new(py, &layer1_weights_out);

            // 4. Return as a Python tuple
            Ok((retina_bytes, layer1_bytes, layer1_weights_bytes))
        }
    }
}
