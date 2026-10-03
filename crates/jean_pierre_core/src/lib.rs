use bytemuck::bytes_of;
use image::GrayImage;
use pollster::FutureExt;
use rand::distr::Uniform;
use rand::{RngExt, random};
use wgpu::{
    BufferUsages, ComputePipelineDescriptor, Extent3d, Instance, Origin3d, ShaderModuleDescriptor,
};

use rand::Fill;
use rand::Rng;

pub struct SaccadeGenerator<T: Rng> {
    pub x: f32,
    pub y: f32,
    decay: f32,     // Restoring force toward center (0.0, 0.0)
    step_size: f32, // Drift velocity scale (sub-pixel)
    rng: T,
}

impl<T: Rng> SaccadeGenerator<T> {
    pub fn new(rng: T) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            decay: 0.85,     // Pulls back to center each tick
            step_size: 0.15, // Max ~0.15 pixel step per frame
            rng,
        }
    }

    /// Call this every game loop frame tick (e.g., at 60 FPS)
    pub fn update(&mut self) -> (f32, f32) {
        // 1. Add random Gaussian-like step
        let dx: f32 = self.rng.random_range(-self.step_size..self.step_size);
        let dy: f32 = self.rng.random_range(-self.step_size..self.step_size);

        // 2. Apply step + decay (prevents the image from drifting off to infinity)
        self.x = (self.x + dx) * self.decay;
        self.y = (self.y + dy) * self.decay;

        (self.x, self.y)
    }
}

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

pub struct VisionSNN {
    pub instance: Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub shader: wgpu::ShaderModule,
    pub micro_tick_pipeline: wgpu::ComputePipeline,
    pub epsilon_pipeline: wgpu::ComputePipeline,
    pub layer1_inference_pipeline: wgpu::ComputePipeline,
    pub layer1_stdp_pipeline: wgpu::ComputePipeline,
    pub apply_stdp_deltas_pipeline: wgpu::ComputePipeline,
    pub input_frame_texture: wgpu::Texture,
    pub input_frame_view: wgpu::TextureView,
    pub input_sampler: wgpu::Sampler,
    pub layer1_input: wgpu::Buffer,
    pub spike_counter: wgpu::Buffer,
    pub params_buf: wgpu::Buffer,
    pub epsilon_buf: wgpu::Buffer,
    pub ping_bind_group: wgpu::BindGroup,
    pub pong_bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub staging_output: wgpu::Buffer,
    pub staging_layer1_activations: wgpu::Buffer,
    pub staging_layer1_vertical_weights: wgpu::Buffer,
    pub layer1_activations: wgpu::Buffer,
    pub layer1_next_activations: wgpu::Buffer,
    pub layer1_vertical_weights: wgpu::Buffer,
    pub layer1_next_vertical_weights: wgpu::Buffer,
    pub layer1_vertical_deltas: wgpu::Buffer,
    pub saccade_generator: SaccadeGenerator<rand::rngs::SmallRng>,
    pub global_iterations: u32,
}

impl VisionSNN {
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
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let micro_tick_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Compute pipe"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("preprocess_frame"),
            compilation_options: Default::default(),
            cache: None,
        });
        let epsilon_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Epsilon pipe"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_epsilon"),
            compilation_options: Default::default(),
            cache: None,
        });
        let layer1_inference_pipeline =
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Layer 1 inference pipe"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("inference"),
                compilation_options: Default::default(),
                cache: None,
            });
        let layer1_stdp_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Layer 1 STDP pipe"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_stdp_deltas"),
            compilation_options: Default::default(),
            cache: None,
        });
        let apply_stdp_deltas_pipeline =
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some("Apply STDP pipe"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("apply_stdp_deltas"),
                compilation_options: Default::default(),
                cache: None,
            });
        let input_frame_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Input frame texture"),
            size: Extent3d {
                width: 512,
                height: 512,
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
        let input_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Input Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let layer1_input = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: 512 * 512 / 4,
            usage: BufferUsages::COPY_DST | BufferUsages::STORAGE | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let spike_counter = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spike Counter"),
            size: 32,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer"),
            size: std::mem::size_of::<FrameParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let epsilon_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Epsilon Buffer"),
            size: 4 * 2,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layer1_activations = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Layer 1 Activations"),
            size: 512 * 512,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layer1_next_activations = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Layer 1 Next Activations"),
            size: 512 * 512,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let layer1_vertical_weights = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Layer 1 Vertical Weights"),
            // 2 input channels 8 rows, 8 columns, 8 kernels
            size: 5 * 5 * 8 * 2,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut rng = rand::rng();
        let range = Uniform::new(-20, 20).unwrap();

        let vals: Vec<i8> = (0..(5 * 5 * 8 * 2)).map(|_| rng.sample(&range)).collect();
        // let mut vals: Vec<i8> = vec![0; 5 * 5 * 8 * 2];
        // vals[0..5].copy_from_slice(&[-54, -27, 0, 27, 54]);
        // vals[5..10].copy_from_slice(&[-54, -27, 0, 27, 54]);
        // vals[10..15].copy_from_slice(&[-108, -54, 0, 54, 108]);
        // vals[15..20].copy_from_slice(&[-54, -27, 0, 27, 54]);
        // vals[20..25].copy_from_slice(&[-54, -27, 0, 27, 54]);
        queue.write_buffer(&layer1_vertical_weights, 0, bytemuck::cast_slice(&vals));

        let layer1_next_vertical_weights = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Layer 1 Next Vertical Weights"),
            // 5 rows, 5 columns 8 kernels, 2 input channels
            size: 5 * 5 * 8 * 2,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let layer1_vertical_deltas = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Layer 1 Vertical Deltas"),
            size: 5 * 5 * 8 * 2 * 4, // 5 rows, 5 columns, 8 kernels, 4 bytes per delta
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ping_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input_frame_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&input_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: layer1_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: spike_counter.as_entire_binding(),
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
        let pong_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Pong Bind group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&input_frame_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&input_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: layer1_input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: spike_counter.as_entire_binding(),
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

        let staging_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Output Buffer"),
            size: (512 * 512 / 4) as u64, // 262,144 bytes
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let staging_layer1_activations = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Layer 1 Activations"),
            size: (512 * 512) as u64, // 262,144 bytes
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let staging_layer1_vertical_weights = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Layer 1 Vertical Weights"),
            size: (5 * 5 * 8 * 2) as u64, // 262,144 bytes
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            instance,
            adapter,
            device,
            queue,
            shader,
            micro_tick_pipeline,
            epsilon_pipeline,
            layer1_inference_pipeline,
            layer1_stdp_pipeline,
            apply_stdp_deltas_pipeline,
            input_frame_texture,
            input_frame_view,
            input_sampler,
            layer1_input,
            spike_counter,
            params_buf,
            epsilon_buf,
            staging_output,
            staging_layer1_activations,
            staging_layer1_vertical_weights,
            ping_bind_group,
            pong_bind_group,
            bind_group_layout,
            saccade_generator: SaccadeGenerator::new(rand::make_rng()),
            layer1_activations,
            layer1_next_activations,
            layer1_vertical_weights,
            layer1_next_vertical_weights,
            layer1_vertical_deltas,
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
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.input_frame_texture,
                mip_level: 0,
                origin: Origin3d::ZERO,
                aspect: Default::default(),
            },
            &input_frame,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(512 * 4),
                rows_per_image: None,
            },
            Extent3d {
                width: 512,
                height: 512,
                depth_or_array_layers: 1,
            },
        );

        // Run the microtick iterations
        for _ in 0..micro_ticks {
            self.global_iterations += 1;
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let (saccade_x, saccade_y) = self.saccade_generator.update();
            let params = FrameParams {
                width: 512,
                height: 512,
                saccade_x,
                saccade_y,
            };

            self.queue
                .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

            let bind_group = if self.global_iterations % 2 == 0 {
                &self.ping_bind_group
            } else {
                &self.pong_bind_group
            };

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.micro_tick_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(4, 512, 1);

            // Inference pass
            pass.set_pipeline(&self.layer1_inference_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(512 / 256, 512, 8);

            // Update epsilon
            pass.set_pipeline(&self.epsilon_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);

            // STDP pass
            pass.set_pipeline(&self.layer1_stdp_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 8192 / 512, 1);

            // Apply Deltas
            pass.set_pipeline(&self.apply_stdp_deltas_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);

            drop(pass);

            self.queue.submit(Some(encoder.finish()));
        }

        // Map our buffers back
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(&self.layer1_input, 0, &self.staging_output, 0, None);
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
        self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        // 4. Access mapped slice as &[u32]
        let img = {
            let data = self.staging_output.get_mapped_range(..).unwrap();
            let words: &[u32] = bytemuck::cast_slice(&data);
            decode_spike_words(words, 512, 512).to_vec()
        };

        let layer1_activations = {
            let data = self
                .staging_layer1_activations
                .get_mapped_range(..)
                .unwrap();
            let words: &[u32] = bytemuck::cast_slice(&data);
            decode_activations(words, 512, 512, 8, 0).to_vec()
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
