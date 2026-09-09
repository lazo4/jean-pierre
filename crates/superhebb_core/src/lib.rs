use image::{
    GrayImage,
    imageops::{FilterType, resize},
};
use pollster::FutureExt;
use rand::RngExt;
use wgpu::{
    BufferUsages, ComputePipelineDescriptor, Extent3d, Instance, Origin3d, ShaderModuleDescriptor,
};

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

pub struct VisionSNN {
    pub instance: Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub shader: wgpu::ShaderModule,
    pub micro_tick_pipeline: wgpu::ComputePipeline,
    pub epsilon_pipeline: wgpu::ComputePipeline,
    pub input_frame_texture: wgpu::Texture,
    pub input_frame_view: wgpu::TextureView,
    pub input_sampler: wgpu::Sampler,
    pub output_buffer: wgpu::Buffer,
    pub spike_counter: wgpu::Buffer,
    pub params_buf: wgpu::Buffer,
    pub epsilon_buf: wgpu::Buffer,
    pub micro_tick_group: wgpu::BindGroup,
    pub epsilon_group: wgpu::BindGroup,
    pub staging_output: wgpu::Buffer,
    pub saccade_generator: SaccadeGenerator<rand::rngs::SmallRng>,
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
        let micro_tick_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Compute pipe"),
            layout: None,
            module: &shader,
            entry_point: Some("preprocess_frame"),
            compilation_options: Default::default(),
            cache: None,
        });
        let epsilon_pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Epsilon pipe"),
            layout: None,
            module: &shader,
            entry_point: Some("update_epsilon"),
            compilation_options: Default::default(),
            cache: None,
        });

        let input_frame_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Input frame texture"),
            size: Extent3d {
                width: 1024,
                height: 1024,
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
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: 1024 * 1024 / 4,
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
            size: 4,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let micro_tick_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind group"),
            layout: &micro_tick_pipeline.get_bind_group_layout(0),
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
                    resource: output_buffer.as_entire_binding(),
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
            ],
        });
        let epsilon_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Epsilon Bind group"),
            layout: &epsilon_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: spike_counter.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: epsilon_buf.as_entire_binding(),
                },
            ],
        });

        let staging_output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Output Buffer"),
            size: (1024 * 1024 / 4) as u64, // 262,144 bytes
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
            input_frame_texture,
            input_frame_view,
            input_sampler,
            output_buffer,
            spike_counter,
            params_buf,
            epsilon_buf,
            micro_tick_group,
            epsilon_group,
            staging_output,
            saccade_generator: SaccadeGenerator::new(rand::make_rng()),
        }
    }

    /// Run microticks with the given input frame
    pub fn run_micro_ticks(&mut self, input_frame: &[u8], micro_ticks: u32, out: &mut [u8]) {
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
                bytes_per_row: Some(1024 * 4),
                rows_per_image: None,
            },
            Extent3d {
                width: 1024,
                height: 1024,
                depth_or_array_layers: 1,
            },
        );

        // Run the microtick iterations
        for _ in 0..micro_ticks {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            let (saccade_x, saccade_y) = self.saccade_generator.update();
            let params = FrameParams {
                width: 1024,
                height: 1024,
                saccade_x,
                saccade_y,
            };

            self.queue
                .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.micro_tick_pipeline);
            pass.set_bind_group(0, &self.micro_tick_group, &[]);
            pass.dispatch_workgroups(4, 1024, 1);
            drop(pass);
            // Update epsilon
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.epsilon_pipeline);
            pass.set_bind_group(0, &self.epsilon_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            drop(pass);

            self.queue.submit(Some(encoder.finish()));
        }

        // Map our buffers back
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(&self.output_buffer, 0, &self.staging_output, 0, None);
        encoder.map_buffer_on_submit(&self.staging_output, wgpu::MapMode::Read, .., |_| {});
        self.queue.submit(Some(encoder.finish()));
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();

        // 4. Access mapped slice as &[u32]
        let img = {
            let data = self.staging_output.get_mapped_range(..).unwrap();
            let words: &[u32] = bytemuck::cast_slice(&data);
            decode_spike_words(words, 1024, 1024).to_vec()
        };

        self.staging_output.unmap();

        out.copy_from_slice(&img);
    }
}
