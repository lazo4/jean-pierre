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
            Vec<Bound<'py, PyBytes>>,
            Vec<Bound<'py, PyBytes>>,
        ) {
            let (retina_out, layer1_out, weights) =
                self.inner.run_micro_ticks(input_frame, micro_ticks);

            // 3. Convert the vectors into Python bytes objects (copies data)
            let retina_bytes = PyBytes::new(py, &retina_out);
            let layer1_bytes = layer1_out
                .iter()
                .map(|out| PyBytes::new(py, &out))
                .collect();
            let weights_bytes = weights.iter().map(|k| PyBytes::new(py, &k)).collect();

            // 4. Return as a Python tuple
            (retina_bytes, layer1_bytes, weights_bytes)
        }
    }
}
