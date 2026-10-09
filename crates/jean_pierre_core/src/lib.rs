use image::GrayImage;
use pollster::FutureExt;
use rand::RngExt;
use rand::distr::Uniform;
use wgpu::{
    BufferUsages, ComputePipelineDescriptor, Extent3d, Instance, Origin3d, ShaderModule,
    ShaderModuleDescriptor,
};

use crate::saccade::SaccadeGenerator;

mod saccade;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameParams {
    width: u32,
    height: u32,
    saccade_x: f32,
    saccade_y: f32,
}

pub fn decode_spike_words(words: &[u32], width: u32, height: u32) -> GrayImage {
    let mut img = GrayImage::new(width, height);
    let words_per_row = width / 32;

    for y in 0..height {
        for x in 0..width {
            let row_offset = x / 32;
            let bit_idx = x % 32;

            // WGSL index calculation matching: (y * words_per_row + row_offset) * 2
            let word_idx = ((y * words_per_row + row_offset) * 2) as usize;

            let word_on = words[word_idx];
            let word_off = words[word_idx + 1];

            let is_on = (word_on & (1 << bit_idx)) != 0;
            let is_off = (word_off & (1 << bit_idx)) != 0;

            let pixel_val = match (is_on, is_off) {
                (true, _) => 255, // White (ON Spike)
                (_, true) => 0,   // Black (OFF Spike)
                _ => 128,         // Grey (Sub-threshold / Silent)
            };

            img.put_pixel(x, y, image::Luma([pixel_val]));
        }
    }

    img
}

pub fn decode_activations(
    words: &[u32],
    width: u32,
    height: u32,
    kernels: u32,
    target_kernel: u32,
) -> GrayImage {
    let mut img = GrayImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let words_per_row = width / 32;
            let row_offset = x / 32;
            let word_index = (y * words_per_row + row_offset) * kernels;
            let activations = words[(word_index + target_kernel) as usize];

            let spike = (activations & (1 << x % 32)) != 0;

            let pixel_val = if spike { 255 } else { 0 };
            img.put_pixel(x, y, image::Luma([pixel_val]));
        }
    }

    img
}

pub fn decode_vertical_weights(words: &[u8], rows: u32, cols: u32) -> GrayImage {
    let mut img = GrayImage::new(rows, cols);

    for y in 0..rows {
        for x in 0..cols {
            let index = (y * cols + x) as usize;
            let weight = words[index] as i8;
            let pixel_val = (weight as i16 + 128) as u8;
            img.put_pixel(x, y, image::Luma([pixel_val]));
        }
    }

    img
}

pub struct GpuContext {
    pub instance: Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    pub fn new() -> Self {
        // Init the wgpu pipeline
        let instance = Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .block_on()
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::SUBGROUP,
                required_limits: Default::default(),
                experimental_features: Default::default(),
                memory_hints: Default::default(),
                trace: Default::default(),
            })
            .block_on()
            .unwrap();

        Self {
            instance,
            adapter,
            device,
            queue,
        }
    }
}

pub struct PreprocessingLayer {
    input_frame_texture: wgpu::Texture,
    input_frame_view: wgpu::TextureView,
    input_sampler: wgpu::Sampler,
    prev_output: wgpu::Buffer,
    next_output: wgpu::Buffer,
    micro_tick_pipeline: wgpu::ComputePipeline,
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

pub struct VisionSNN {
    gpu_ctx: GpuContext,
    preprocessing_layer: PreprocessingLayer,
    spike_bucket_pipeline: wgpu::ComputePipeline,
    layer1_inference_pipeline: wgpu::ComputePipeline,
    layer1_stdp_pipeline: wgpu::ComputePipeline,
    apply_stdp_deltas_pipeline: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    epsilon_buf: wgpu::Buffer,
    ping_bind_group: wgpu::BindGroup,
    pong_bind_group: wgpu::BindGroup,
    staging_output: wgpu::Buffer,
    staging_layer1_activations: wgpu::Buffer,
    staging_layer1_vertical_weights: wgpu::Buffer,
    layer1_activations: wgpu::Buffer,
    layer1_next_activations: wgpu::Buffer,
    layer1_vertical_weights: wgpu::Buffer,
    layer1_next_vertical_weights: wgpu::Buffer,
    layer1_vertical_deltas: wgpu::Buffer,
    saccade_generator: saccade::SaccadeGenerator<rand::rngs::SmallRng>,
    global_iterations: u32,
    crop_size: u32,
}

impl VisionSNN {
    pub fn new(crop_size: u32) -> Self {
        let gpu_ctx = GpuContext::new();

        let shader = gpu_ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let bind_group_layout =
            gpu_ctx
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: Default::default(),
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 7,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 8,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 9,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 10,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            gpu_ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[Some(&bind_group_layout)],
                    immediate_size: 0,
                });

        let preprocessing_layer =
            PreprocessingLayer::new(&gpu_ctx, &shader, &pipeline_layout, crop_size);

        let spike_bucket_pipeline =
            gpu_ctx
                .device
                .create_compute_pipeline(&ComputePipelineDescriptor {
                    label: Some("Spike bucket pipe"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("write_spike_counts"),
                    compilation_options: Default::default(),
                    cache: None,
                });
        let layer1_inference_pipeline =
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
        let layer1_stdp_pipeline =
            gpu_ctx
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
        let spike_counter = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spike Counter"),
            size: crop_size as u64 * crop_size as u64 / 32 * 32 * 4, // 1024 bytes
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buf = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer"),
            size: std::mem::size_of::<FrameParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let epsilon_buf = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Epsilon Buffer"),
            size: crop_size as u64 * crop_size as u64 * 2 * 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layer1_activations = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Layer 1 Activations"),
            size: crop_size as u64 * crop_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layer1_next_activations = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Layer 1 Next Activations"),
            size: crop_size as u64 * crop_size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let layer1_vertical_weights = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
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
            .write_buffer(&layer1_vertical_weights, 0, bytemuck::cast_slice(&vals));

        let layer1_next_vertical_weights = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Layer 1 Next Vertical Weights"),
            // 5 rows, 5 columns 8 kernels, 2 input channels
            size: 5 * 5 * 8 * 2,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layer1_vertical_deltas = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Layer 1 Vertical Deltas"),
            size: 5 * 5 * 8 * 2 * 4, // 5 rows, 5 columns, 8 kernels, 4 bytes per delta
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ping_bind_group = gpu_ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &preprocessing_layer.input_frame_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            &preprocessing_layer.input_sampler,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: preprocessing_layer.prev_output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: preprocessing_layer.next_output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: epsilon_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: layer1_activations.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: layer1_next_activations.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: layer1_vertical_weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: layer1_next_vertical_weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: layer1_vertical_deltas.as_entire_binding(),
                    },
                ],
            });

        // Weights and activations are swapped between ticks
        let pong_bind_group = gpu_ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Pong Bind group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(
                            &preprocessing_layer.input_frame_view,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            &preprocessing_layer.input_sampler,
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: preprocessing_layer.next_output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: preprocessing_layer.prev_output.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: epsilon_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: layer1_next_activations.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: layer1_activations.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: layer1_next_vertical_weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: layer1_vertical_weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: layer1_vertical_deltas.as_entire_binding(),
                    },
                ],
            });

        let staging_output = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Output Buffer"),
            size: (crop_size * crop_size / 4) as u64, // 262,144 bytes
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let staging_layer1_activations = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Layer 1 Activations"),
            size: crop_size as u64 * crop_size as u64, // 262,144 bytes
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let staging_layer1_vertical_weights =
            gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging Layer 1 Vertical Weights"),
                size: (5 * 5 * 8 * 2) as u64, // 262,144 bytes
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

        Self {
            gpu_ctx,
            preprocessing_layer,
            spike_bucket_pipeline,
            layer1_inference_pipeline,
            layer1_stdp_pipeline,
            apply_stdp_deltas_pipeline,
            params_buf,
            epsilon_buf,
            staging_output,
            staging_layer1_activations,
            staging_layer1_vertical_weights,
            ping_bind_group,
            pong_bind_group,
            saccade_generator: SaccadeGenerator::new(rand::make_rng()),
            layer1_activations,
            layer1_next_activations,
            layer1_vertical_weights,
            layer1_next_vertical_weights,
            layer1_vertical_deltas,
            crop_size,
            global_iterations: 0,
        }
    }

    /// Run microticks with the given input frame
    pub fn run_micro_ticks(
        &mut self,
        input_frame: &[u8],
        micro_ticks: u32,
        retina_out: &mut [u8],
        layer1_out: &mut [u8],
        layer1_weights_out: &mut [u8],
    ) {
        self.gpu_ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.preprocessing_layer.input_frame_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: Default::default(),
            },
            input_frame,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.crop_size * 4),
                rows_per_image: None,
            },
            Extent3d {
                width: self.crop_size,
                height: self.crop_size,
                depth_or_array_layers: 1,
            },
        );

        // Run the microtick iterations
        for _ in 0..micro_ticks {
            self.global_iterations += 1;
            let mut encoder = self
                .gpu_ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let (saccade_x, saccade_y) = self.saccade_generator.update();
            let params = FrameParams {
                width: self.crop_size,
                height: self.crop_size,
                saccade_x,
                saccade_y,
            };

            self.gpu_ctx
                .queue
                .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

            let bind_group = if self.global_iterations.is_multiple_of(2) {
                &self.ping_bind_group
            } else {
                &self.pong_bind_group
            };

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.preprocessing_layer.micro_tick_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(4, self.crop_size, 1);

            // // Inference pass
            // pass.set_pipeline(&self.layer1_inference_pipeline);
            // pass.set_bind_group(0, bind_group, &[]);
            // pass.dispatch_workgroups(self.crop_size / 256, self.crop_size, 8);

            // Update spike counters
            pass.set_pipeline(&self.spike_bucket_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(4, self.crop_size, 1);
            //
            // // STDP pass
            // pass.set_pipeline(&self.layer1_stdp_pipeline);
            // pass.set_bind_group(0, bind_group, &[]);
            // pass.dispatch_workgroups(1, 8192 / self.crop_size, 1);
            //
            // // Apply Deltas
            // pass.set_pipeline(&self.apply_stdp_deltas_pipeline);
            // pass.set_bind_group(0, bind_group, &[]);
            // pass.dispatch_workgroups(1, 1, 1);

            drop(pass);

            self.gpu_ctx.queue.submit(Some(encoder.finish()));
        }

        // Map our buffers back
        let mut encoder = self
            .gpu_ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(
            &self.preprocessing_layer.next_output,
            0,
            &self.staging_output,
            0,
            None,
        );
        encoder.copy_buffer_to_buffer(
            &self.layer1_next_activations,
            0,
            &self.staging_layer1_activations,
            0,
            None,
        );
        encoder.copy_buffer_to_buffer(
            &self.layer1_vertical_weights,
            0,
            &self.staging_layer1_vertical_weights,
            0,
            None,
        );
        encoder.map_buffer_on_submit(&self.staging_output, wgpu::MapMode::Read, .., |_| {});
        encoder.map_buffer_on_submit(
            &self.staging_layer1_activations,
            wgpu::MapMode::Read,
            ..,
            |_| {},
        );
        encoder.map_buffer_on_submit(
            &self.staging_layer1_vertical_weights,
            wgpu::MapMode::Read,
            ..,
            |_| {},
        );
        self.gpu_ctx.queue.submit(Some(encoder.finish()));
        self.gpu_ctx
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        // 4. Access mapped slice as &[u32]
        let img = {
            let data = self.staging_output.get_mapped_range(..).unwrap();
            let words: &[u32] = bytemuck::cast_slice(&data);
            decode_spike_words(words, self.crop_size, self.crop_size).to_vec()
        };

        let layer1_activations = {
            let data = self
                .staging_layer1_activations
                .get_mapped_range(..)
                .unwrap();
            let words: &[u32] = bytemuck::cast_slice(&data);
            decode_activations(words, self.crop_size, self.crop_size, 8, 0).to_vec()
        };

        let layer1_vertical_weights = {
            let data = self
                .staging_layer1_vertical_weights
                .get_mapped_range(..)
                .unwrap();
            let words: &[u8] = bytemuck::cast_slice(&data);
            decode_vertical_weights(words, 5, 5).to_vec()
        };

        self.staging_output.unmap();
        self.staging_layer1_activations.unmap();
        self.staging_layer1_vertical_weights.unmap();

        retina_out.copy_from_slice(&img);
        layer1_out.copy_from_slice(&layer1_activations);
        layer1_weights_out.copy_from_slice(&layer1_vertical_weights);
    }
}
