use std::time::Instant;

use image::{
    GrayImage,
    imageops::{FilterType, resize},
};
use pollster::FutureExt;
use rand::RngExt;
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindGroupLayout, BufferDescriptor, BufferUsages,
    ComputePipelineDescriptor, Extent3d, Instance, ShaderModuleDescriptor,
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

struct FrameParams {
    width: u32,
    height: u32,
    epsilon: f32,
    saccade_x: f32,
    saccade_y: f32,
}

/// Maps i8 (-128..=127) buffer to u8 (0..=255) for grayscale imaging
fn i8_to_u8_buffer(data: &[i8]) -> Vec<u8> {
    data.iter().map(|&x| (x as i16 + 128) as u8).collect()
}

/// Saves the 509x509 activation map as a PNG
pub fn save_activations(activations: &[u8], path: &str) {
    let activations = activations.to_vec();
    let img = GrayImage::from_raw(509, 509, activations).expect("Invalid activation buffer size");
    img.save(path).unwrap();
}

/// Upscales and saves small N x N kernels (e.g., 8x8 or 3x3) as sharp PNGs
pub fn save_kernel(weights: &[i8], side_len: u32, display_size: u32, path: &str) {
    let raw = i8_to_u8_buffer(weights);
    let img = GrayImage::from_raw(side_len, side_len, raw).expect("Invalid kernel buffer size");

    // FilterType::Nearest keeps pixel edges sharp when scaling up
    let scaled = resize(&img, display_size, display_size, FilterType::Nearest);
    scaled.save(path).unwrap();
}

const KERNELS: usize = 8;
const INPUT_SIZE: usize = 1024;
const KERNEL_SIZE: usize = 8;
const KERNEL_STRIDE: usize = 2;
const ACTIVATIONS_SIZE: usize = (INPUT_SIZE - KERNEL_SIZE) / KERNEL_STRIDE + 1;

fn main() {
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

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("Compute pipe"),
        layout: None,
        module: &shader,
        entry_point: None,
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
        size: 1024 * 1024 / 8,
        usage: BufferUsages::COPY_DST | BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let spike_counter = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Spike Counter"),
        size: 32,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_spike_counter = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Spike Counter"),
        size: 32,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Params Buffer"),
        size: std::mem::size_of::<FrameParams>() as u64,
        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Bind group"),
        layout: &pipeline.get_bind_group_layout(0),
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
        ],
    });

    // Run the game loop
    let mut iteration_count = 0;
    // Uncapped fps we'll see how fast we go.
    let mut frame_count = 0u64;
    let mut last_fps_check = Instant::now();

    let static_image = image::open("assets/stairs.jpg").unwrap();
    let input_image_raw = static_image.to_rgba8().into_raw();

    let mut saccade_generator = SaccadeGenerator::new(rand::rng());

    loop {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute encoder"),
        });

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(64, 64, 2);
        drop(pass);

        queue.submit(Some(encoder.finish()));
        // --- 3. FPS Counter ---
        frame_count += 1;
        iteration_count += 1;
        let elapsed = last_fps_check.elapsed();

        if elapsed.as_secs_f64() >= 1.0 {
            let fps = frame_count as f64 / elapsed.as_secs_f64();
            println!(
                "Performance: {:.2} FPS, iterations: {}",
                fps, iteration_count
            );
            //
            frame_count = 0;
            last_fps_check = Instant::now();
        }
    }
}
