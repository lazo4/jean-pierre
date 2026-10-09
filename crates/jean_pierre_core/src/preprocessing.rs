use wgpu::ComputePipelineDescriptor;

use wgpu::BufferUsages;

use wgpu::Extent3d;
use wgpu::ShaderModule;

use super::GpuContext;

pub struct PreprocessingLayer {
    pub(crate) input_frame_texture: wgpu::Texture,
    pub(crate) input_frame_view: wgpu::TextureView,
    pub(crate) input_sampler: wgpu::Sampler,
    pub(crate) prev_output: wgpu::Buffer,
    pub(crate) next_output: wgpu::Buffer,
    pub(crate) micro_tick_pipeline: wgpu::ComputePipeline,
}

impl PreprocessingLayer {
    pub fn new(
        gpu_ctx: &GpuContext,
        shader: &ShaderModule,
        pipeline_layout: &wgpu::PipelineLayout,
        crop_size: u32,
    ) -> Self {
        let input_frame_texture = gpu_ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Input frame texture"),
            size: Extent3d {
                width: crop_size,
                height: crop_size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let input_frame_view =
            input_frame_texture.create_view(&wgpu::wgt::TextureViewDescriptor::default());
        let input_sampler = gpu_ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Input Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let prev_output = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: crop_size as u64 * crop_size as u64 / 4,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let next_output = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: crop_size as u64 * crop_size as u64 / 4,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let micro_tick_pipeline =
            gpu_ctx
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some("Compute pipe"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("preprocess_frame"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        Self {
            input_frame_texture,
            input_frame_view,
            input_sampler,
            prev_output,
            next_output,
            micro_tick_pipeline,
        }
    }
}
