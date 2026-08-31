use std::time::Instant;

use image::{
    GrayImage,
    imageops::{FilterType, resize},
};
use rand::RngExt;
use rayon::{
    iter::{IndexedParallelIterator, ParallelIterator},
    slice::ParallelSliceMut,
};

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
    // 8 pixel wide kernel with a stride of 2 over a 1024*1024 input
    // (1024 - 8) / 2 + 1 = 509 * 8 kernels
    // NOTE: Activations are always positive (due to relu)
    let mut activations: Vec<u8> = vec![0; KERNELS * ACTIVATIONS_SIZE * ACTIVATIONS_SIZE];
    let mut rng = rand::rng();
    // 8*8 kernel w/ 8 filter and a single greyscale channel
    let mut weights: Vec<i8> = (0..(KERNELS * KERNEL_SIZE * KERNEL_SIZE))
        .map(|_| rng.random_range(-10..10))
        .collect();
    println!("Weights: {:?}", weights);
    // Each kernel has a set of lateral weights
    let mut lat_weights: Vec<i8> = vec![0; KERNELS * 3 * 3];
    let mut cross_kernel_weights: Vec<i8> = vec![0; KERNELS * KERNELS];
    let mut next_activations: Vec<u8> = vec![0; KERNELS * ACTIVATIONS_SIZE * ACTIVATIONS_SIZE];
    let mut next_weights: Vec<i8> = vec![0; KERNELS * KERNEL_SIZE * KERNEL_SIZE];
    let mut next_lat_weights: Vec<i8> = vec![0; KERNELS * 3 * 3];
    let mut next_cross_kernel_weights: Vec<i8> = vec![0; KERNELS * KERNELS];

    // Uncapped fps we'll see how fast we go.
    let mut frame_count = 0u64;
    let mut last_fps_check = Instant::now();

    let mut total_frames = 0u64;

    // Use an image
    let file = image::open("assets/tennis.jpg").unwrap();
    let input: Vec<u8> = file.to_luma8().to_vec();
    let mut delta_weights = vec![0; KERNELS * KERNEL_SIZE * KERNEL_SIZE];
    let mut delta_lat_weights = vec![0; KERNELS * 3 * 3];
    let mut delta_cross_kernel_weights = vec![0; KERNELS * KERNELS];

    loop {
        // --- 1. Compute Tick ---
        // (Your activation and weight updates go here)

        // Calculate the activations
        // TODO: Optimize!!
        next_activations
            .par_chunks_exact_mut(ACTIVATIONS_SIZE * ACTIVATIONS_SIZE)
            .enumerate()
            .for_each(|(kernel, next_activations)| {
                let kernel_weights = &weights
                    [kernel * KERNEL_SIZE * KERNEL_SIZE..(kernel + 1) * KERNEL_SIZE * KERNEL_SIZE];
                let kernel_lat_weights = &lat_weights[kernel * 3 * 3..(kernel + 1) * 3 * 3];
                let kernel_cross_weights =
                    &cross_kernel_weights[kernel * KERNELS..(kernel + 1) * KERNELS];

                for i in 0..ACTIVATIONS_SIZE {
                    for j in 0..ACTIVATIONS_SIZE {
                        let mut sum = 0;

                        // Handle the vertical weights
                        for k in 0..KERNEL_SIZE {
                            for l in 0..KERNEL_SIZE {
                                let input_value = input
                                    [(i * KERNEL_STRIDE + k) * INPUT_SIZE + j * KERNEL_STRIDE + l];
                                let weight = kernel_weights[k * KERNEL_SIZE + l];
                                sum += input_value as i32 * weight as i32;
                            }
                        }

                        // Calculate the lateral weights
                        for dx in 0..=2 {
                            for dy in 0..=2 {
                                if dx == 1 && dy == 1 {
                                    continue;
                                }
                                let input =
                                    activations[kernel * ACTIVATIONS_SIZE * ACTIVATIONS_SIZE
                                        + (i + dx).saturating_sub(1).min(ACTIVATIONS_SIZE - 1)
                                            * ACTIVATIONS_SIZE
                                        + (j + dy).saturating_sub(1).min(ACTIVATIONS_SIZE - 1)];
                                let weight = kernel_lat_weights[dx * 3 + dy];
                                sum += input as i32 * weight as i32;
                            }
                        }

                        // Calculate the cross-kernel weights
                        for source_kernel in 0..KERNELS {
                            let input =
                                activations[source_kernel * ACTIVATIONS_SIZE * ACTIVATIONS_SIZE
                                    + i * ACTIVATIONS_SIZE
                                    + j];
                            let weight = kernel_cross_weights[source_kernel];
                            sum += input as i32 * weight as i32;
                        }

                        next_activations[i * ACTIVATIONS_SIZE + j] = (sum >> 7).clamp(0, 255) as u8;
                    }
                }
            });

        // Now update the weights
        //
        delta_weights
            .par_chunks_exact_mut(KERNEL_SIZE * KERNEL_SIZE)
            .zip(delta_lat_weights.par_chunks_exact_mut(3 * 3))
            .zip(delta_cross_kernel_weights.par_chunks_exact_mut(KERNELS))
            .enumerate()
            .for_each(|(kernel, ((weights, lat_weights), cross_weights))| {
                let kernel_activations = &activations[kernel * ACTIVATIONS_SIZE * ACTIVATIONS_SIZE
                    ..(kernel + 1) * ACTIVATIONS_SIZE * ACTIVATIONS_SIZE];
                let next_kernel_activations =
                    &next_activations[kernel * ACTIVATIONS_SIZE * ACTIVATIONS_SIZE
                        ..(kernel + 1) * ACTIVATIONS_SIZE * ACTIVATIONS_SIZE];

                for i in 0..ACTIVATIONS_SIZE {
                    for j in 0..ACTIVATIONS_SIZE {
                        // Update the vertical weights
                        for k in 0..KERNEL_SIZE {
                            for l in 0..KERNEL_SIZE {
                                let presynaptic = input
                                    [(i * KERNEL_STRIDE + k) * INPUT_SIZE + j * KERNEL_STRIDE + l]
                                    as i32;
                                let next_presynaptic = input
                                    [(i * KERNEL_STRIDE + k) * INPUT_SIZE + j * KERNEL_STRIDE + l]
                                    as i32;
                                let postsynaptic =
                                    next_kernel_activations[i * ACTIVATIONS_SIZE + j] as i32;
                                let prev_postsynaptic =
                                    kernel_activations[i * ACTIVATIONS_SIZE + j] as i32;
                                let delta = presynaptic * postsynaptic
                                    - next_presynaptic * prev_postsynaptic;
                                weights[k * KERNEL_SIZE + l] += delta;
                            }
                        }

                        // Update the lateral weights
                        for dx in 0..=2 {
                            for dy in 0..=2 {
                                if dx == 1 && dy == 1 {
                                    continue;
                                }

                                // Presynaptic neuron is always the current one
                                let presynaptic =
                                    kernel_activations[i * ACTIVATIONS_SIZE + j] as i32;
                                let next_presynaptic =
                                    next_kernel_activations[i * ACTIVATIONS_SIZE + j] as i32;
                                let postsynaptic = next_kernel_activations[(i + dx)
                                    .saturating_sub(1)
                                    .min(ACTIVATIONS_SIZE - 1)
                                    * ACTIVATIONS_SIZE
                                    + (j + dy).saturating_sub(1).min(ACTIVATIONS_SIZE - 1)]
                                    as i32;
                                let prev_postsynaptic = kernel_activations[(i + dx)
                                    .saturating_sub(1)
                                    .min(ACTIVATIONS_SIZE - 1)
                                    * ACTIVATIONS_SIZE
                                    + (j + dy).saturating_sub(1).min(ACTIVATIONS_SIZE - 1)]
                                    as i32;
                                let delta = presynaptic * postsynaptic
                                    - next_presynaptic * prev_postsynaptic;
                                lat_weights[dx * 3 + dy] += delta;
                            }
                        }

                        // Update the cross-kernel weights
                        for source_kernel in 0..KERNELS {
                            let presynaptic =
                                activations[source_kernel * ACTIVATIONS_SIZE * ACTIVATIONS_SIZE
                                    + i * ACTIVATIONS_SIZE
                                    + j] as i32;
                            let next_presynaptic = next_activations[source_kernel
                                * ACTIVATIONS_SIZE
                                * ACTIVATIONS_SIZE
                                + i * ACTIVATIONS_SIZE
                                + j] as i32;
                            let postsynaptic =
                                next_kernel_activations[i * ACTIVATIONS_SIZE + j] as i32;
                            let prev_postsynaptic =
                                kernel_activations[i * ACTIVATIONS_SIZE + j] as i32;
                            let delta =
                                presynaptic * postsynaptic - next_presynaptic * prev_postsynaptic;
                            cross_weights[source_kernel] += delta;
                        }
                    }
                }
            });

        // Now apply the deltas
        for kernel in 0..KERNELS {
            for i in 0..KERNEL_SIZE {
                for j in 0..KERNEL_SIZE {
                    next_weights[kernel * KERNEL_SIZE * KERNEL_SIZE + i * KERNEL_SIZE + j] = weights
                        [kernel * KERNEL_SIZE * KERNEL_SIZE + i * KERNEL_SIZE + j]
                        + ((delta_weights
                            [kernel * KERNEL_SIZE * KERNEL_SIZE + i * KERNEL_SIZE + j]
                            >> 7)
                            >> 9) as i8;
                }
            }

            for dx in 0..=2 {
                for dy in 0..=2 {
                    next_lat_weights[kernel * 3 * 3 + dx * 3 + dy] = lat_weights
                        [kernel * 3 * 3 + dx * 3 + dy]
                        + ((delta_lat_weights[kernel * 3 * 3 + dx * 3 + dy] >> 7) >> 9) as i8;
                }
            }
            for source_kernel in 0..KERNELS {
                next_cross_kernel_weights[kernel * KERNELS + source_kernel] = cross_kernel_weights
                    [kernel * KERNELS + source_kernel]
                    + ((delta_cross_kernel_weights[kernel * KERNELS + source_kernel] >> 7) >> 9)
                        as i8;
            }
        }

        // --- 2. Dual Buffer Swap ---
        std::mem::swap(&mut activations, &mut next_activations);
        std::mem::swap(&mut weights, &mut next_weights);
        std::mem::swap(&mut lat_weights, &mut next_lat_weights);
        std::mem::swap(&mut cross_kernel_weights, &mut next_cross_kernel_weights);
        // Zero out the delta buffers
        delta_weights.fill(0);
        delta_lat_weights.fill(0);
        delta_cross_kernel_weights.fill(0);

        // --- 3. FPS Counter ---
        frame_count += 1;
        total_frames += 1;
        let elapsed = last_fps_check.elapsed();

        if elapsed.as_secs_f64() >= 1.0 {
            let fps = frame_count as f64 / elapsed.as_secs_f64();
            println!("Performance: {:.2} FPS, iterations: {}", fps, total_frames);

            frame_count = 0;
            last_fps_check = Instant::now();
        }
        if total_frames % 500 == 0 {
            println!("Saving images");
            for kernel in 0..8 {
                save_activations(
                    &activations[kernel * 509 * 509..(kernel + 1) * 509 * 509],
                    format!("weights/output_activations{kernel}_{}.png", total_frames).as_str(),
                );

                // Upscale 8x8 weights to 256x256 for easy viewing
                save_kernel(
                    &weights[kernel * 64..(kernel + 1) * 64],
                    8,
                    256,
                    format!("weights/output_weights{kernel}_{}.png", total_frames).as_str(),
                );

                // Upscale 3x3 lateral weights to 192x192
                save_kernel(
                    &lat_weights[kernel * 9..(kernel + 1) * 9],
                    3,
                    192,
                    format!("weights/output_lat_weights{kernel}_{}.png", total_frames).as_str(),
                );
            }
            save_kernel(
                &cross_kernel_weights,
                KERNELS as u32,
                256,
                format!("weights/output_cross_weights_{}.png", total_frames).as_str(),
            );
        }
    }
}
