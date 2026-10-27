use pollster::FutureExt;
use wgpu::{BufferUsages, Extent3d, Instance, Origin3d, ShaderModuleDescriptor};

use crate::{
    bindgroup_layout::init_bindgroup_layout, convlayer::ConvLayer,
    preprocessing::PreprocessingLayer, saccade::SaccadeGenerator,
};

mod bindgroup_layout;
mod convlayer;
mod preprocessing;
mod saccade;
mod spike_decoder;

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct FrameParams {
    width: u32,
    height: u32,
    saccade_x: f32,
    saccade_y: f32,
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
                required_limits: wgpu::Limits {
                    max_storage_buffers_per_shader_stage: 16,
                    ..Default::default()
                },
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

pub struct VisionSNN {
    gpu_ctx: GpuContext,
    preprocessing_layer: PreprocessingLayer,
    layer1: ConvLayer,
    params_buf: wgpu::Buffer,
    epsilon_pipeline: wgpu::ComputePipeline,
    ping_bind_group: wgpu::BindGroup,
    pong_bind_group: wgpu::BindGroup,
    staging_output: wgpu::Buffer,
    staging_layer1_activations: wgpu::Buffer,
    staging_layer1_vertical_weights: wgpu::Buffer,
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

        let bind_group_layout = init_bindgroup_layout(&gpu_ctx.device);

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

        let layer1 = ConvLayer::new(&gpu_ctx, &shader, &pipeline_layout, crop_size);

        let params_buf = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params Buffer"),
            size: std::mem::size_of::<FrameParams>() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let spike_counts = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spike Counts Buffer"),
            size: 4 * 9,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let epsilon_buf = gpu_ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Epsilon Buffer"),
            size: 4 * 9,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let epsilon_pipeline =
            gpu_ctx
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Epsilon Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader,
                    entry_point: Some("update_epsilon"),
                    compilation_options: Default::default(),
                    cache: None,
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
                        resource: spike_counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: layer1.activations.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: layer1.next_activations.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: layer1.weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: layer1.next_weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: layer1.weight_deltas.as_entire_binding(),
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
                        resource: spike_counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: layer1.next_activations.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: layer1.activations.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: layer1.next_weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: layer1.weights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: layer1.weight_deltas.as_entire_binding(),
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
                size: 4 * 7 * 2 * 8,
                usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

        Self {
            gpu_ctx,
            preprocessing_layer,
            layer1,
            params_buf,
            epsilon_pipeline,
            staging_output,
            staging_layer1_activations,
            staging_layer1_vertical_weights,
            ping_bind_group,
            pong_bind_group,
            saccade_generator: SaccadeGenerator::new(rand::make_rng()),
            crop_size,
            global_iterations: 1,
        }
    }

    /// Run microticks with the given input frame
    pub fn run_micro_ticks(
        &mut self,
        input_frame: &[u8],
        micro_ticks: u32,
    ) -> (Vec<u8>, Vec<Vec<u8>>, Vec<Vec<u8>>) {
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

            // Inference pass
            pass.set_pipeline(&self.layer1.inference_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(self.crop_size / 128, self.crop_size, 8);

            // Update epsilon
            pass.set_pipeline(&self.epsilon_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);

            // STDP pass
            pass.set_pipeline(&self.layer1.stdp_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 8192 / self.crop_size, 8);

            // Apply Deltas
            pass.set_pipeline(&self.layer1.apply_stdp_deltas_pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 8);

            drop(pass);

            self.gpu_ctx.queue.submit(Some(encoder.finish()));
        }

        // Map our buffers back
        let mut encoder = self
            .gpu_ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let (src_output, src_activations, src_weights) = if self.global_iterations.is_multiple_of(2)
        {
            // Ping finished: latest outputs are in the `next_*` buffers
            (
                &self.preprocessing_layer.next_output,
                &self.layer1.next_activations,
                &self.layer1.next_weights,
            )
        } else {
            // Pong finished: latest outputs are in `prev_output`, `activations`, and `weights`
            (
                &self.preprocessing_layer.prev_output,
                &self.layer1.activations,
                &self.layer1.weights,
            )
        };

        encoder.copy_buffer_to_buffer(src_output, 0, &self.staging_output, 0, None);
        encoder.copy_buffer_to_buffer(
            src_activations,
            0,
            &self.staging_layer1_activations,
            0,
            None,
        );
        encoder.copy_buffer_to_buffer(
            src_weights,
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
            spike_decoder::decode_spike_words(words, self.crop_size, self.crop_size).to_vec()
        };

        let layer1_activations = {
            let data = self
                .staging_layer1_activations
                .get_mapped_range(..)
                .unwrap();
            let words: &[u32] = bytemuck::cast_slice(&data);
            (0..8)
                .into_iter()
                .map(|i| {
                    spike_decoder::decode_activations(words, self.crop_size, self.crop_size, 8, i)
                        .to_vec()
                })
                .collect()
        };

        let weights = {
            let data = self
                .staging_layer1_vertical_weights
                .get_mapped_range(..)
                .unwrap();
            let words: &[u8] = bytemuck::cast_slice(&data);
            (0..8)
                .into_iter()
                .flat_map(|i| {
                    [
                        spike_decoder::decode_vertical_weights(words, 5, 5, false, i).to_vec(),
                        spike_decoder::decode_vertical_weights(words, 5, 5, true, i).to_vec(),
                    ]
                    .into_iter()
                })
                .collect()
        };

        self.staging_output.unmap();
        self.staging_layer1_activations.unmap();
        self.staging_layer1_vertical_weights.unmap();

        (img, layer1_activations, weights)
    }
}
