use rand::RngExt;
use rand::distr::Uniform;
use wgpu::BufferUsages;

use wgpu::ComputePipelineDescriptor;

use wgpu::ShaderModule;

use super::GpuContext;

pub struct ConvLayer {
    pub(crate) inference_pipeline: wgpu::ComputePipeline,
    pub(crate) stdp_pipeline: wgpu::ComputePipeline,
    pub(crate) apply_stdp_deltas_pipeline: wgpu::ComputePipeline,
    pub(crate) activations: wgpu::Buffer,
    pub(crate) next_activations: wgpu::Buffer,
    pub(crate) weights: wgpu::Buffer,
    pub(crate) next_weights: wgpu::Buffer,
    pub(crate) weight_deltas: wgpu::Buffer,
}

impl ConvLayer {
    pub fn new(
        gpu_ctx: &GpuContext,
        shader: &ShaderModule,
        pipeline_layout: &wgpu::PipelineLayout,
        crop_size: u32,
    ) -> Self {
        let inference_pipeline =
            gpu_ctx
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some("Layer 1 inference pipe"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("inference"),
                    compilation_options: Default::default(),
                    cache: None,
                });
        let stdp_pipeline = gpu_ctx
            .device
            .create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Layer 1 STDP pipe"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("compute_stdp_deltas"),
                compilation_options: Default::default(),
                cache: None,
            });
        let apply_stdp_deltas_pipeline =
            gpu_ctx
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some("Apply STDP pipe"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("apply_stdp_deltas"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let activations = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Layer 1 Activations"),
            size: crop_size as u64 * crop_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let next_activations = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Layer 1 Next Activations"),
            size: crop_size as u64 * crop_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let weights = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Layer 1 Vertical Weights"),
            // 2 input channels 8 rows, 8 columns, 8 kernels
            size: 5 * 5 * 8 * 2,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut rng = rand::rng();
        let range = Uniform::new(-20, 20).unwrap();

        let vals: Vec<i8> = (0..(5 * 5 * 8 * 2)).map(|_| rng.sample(range)).collect();
        // let mut vals: Vec<i8> = vec![0; 5 * 5 * 8 * 2];
        // vals[0..5].copy_from_slice(&[-54, -27, 0, 27, 54]);
        // vals[5..10].copy_from_slice(&[-54, -27, 0, 27, 54]);
        // vals[10..15].copy_from_slice(&[-108, -54, 0, 54, 108]);
        // vals[15..20].copy_from_slice(&[-54, -27, 0, 27, 54]);
        // vals[20..25].copy_from_slice(&[-54, -27, 0, 27, 54]);
        gpu_ctx
            .queue
            .write_buffer(&weights, 0, bytemuck::cast_slice(&vals));

        let next_weights = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Layer 1 Next Vertical Weights"),
            // 5 rows, 5 columns 8 kernels, 2 input channels
            size: 5 * 5 * 8 * 2,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let weight_deltas = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Layer 1 Vertical Deltas"),
            size: 5 * 5 * 8 * 2 * 4, // 5 rows, 5 columns, 8 kernels, 4 bytes per delta
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            inference_pipeline,
            stdp_pipeline,
            apply_stdp_deltas_pipeline,
            activations,
            next_activations,
            weights,
            next_weights,
            weight_deltas,
        }
    }
}
