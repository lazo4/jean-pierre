use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
mod pyjeanpierre {
    use pyo3::{prelude::*, types::PyBytes};

    #[pyclass]
    struct VisionSNN {
        inner: jean_pierre_core::VisionSNN,
        crop_size: usize,
    }

    #[pymethods]
    impl VisionSNN {
        #[new]
        fn new(crop_size: usize) -> Self {
            Self {
                inner: jean_pierre_core::VisionSNN::new(crop_size as u32),
                crop_size,
            }
        }

        fn run_micro_ticks<'py>(
            &mut self,
            py: Python<'py>,
            input_frame: &[u8],
            micro_ticks: u32,
        ) -> (
            Bound<'py, PyBytes>,
            Bound<'py, PyBytes>,
            Bound<'py, PyBytes>,
        ) {
            // 1. Allocate temporary vectors
            let mut retina_out = vec![0u8; self.crop_size * self.crop_size];
            let mut layer1_out = vec![0u8; self.crop_size * self.crop_size];
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
            (retina_bytes, layer1_bytes, layer1_weights_bytes)
        }
    }
}
