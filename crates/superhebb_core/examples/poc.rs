use std::time::Instant;

use image::{
    GrayImage,
    imageops::{FilterType, resize},
};
use rand::RngExt;

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

fn main() {
    // 8 pixel wide kernel with a stride of 2 over a 1024*1024 input
    // (1024 - 8) / 2 + 1 = 509 ; 509^2 = 259081
    // NOTE: Activations are always positive (due to relu)
    let mut activations: Vec<u8> = vec![0; 259081];
    let mut rng = rand::rng();
    // 8*8 kernel w/ 1 filter and a single greyscale channel
    let mut weights: Vec<i8> = (0..(8 * 8)).map(|_| rng.random_range(-10..10)).collect();
    println!("Weights: {:?}", weights);
    let mut lat_weights: Vec<i8> = vec![0; 3 * 3];
    let mut next_activations: Vec<u8> = vec![0; 259081];
    let mut next_weights: Vec<i8> = vec![0; 8 * 8];
    let mut next_lat_weights: Vec<i8> = vec![0; 3 * 3];

    // Uncapped fps we'll see how fast we go.
    let mut frame_count = 0u64;
    let mut last_fps_check = Instant::now();

    let mut total_frames = 0u64;

    // Use an image
    let file = image::open("assets/stripe.png").unwrap();
    let input: Vec<u8> = file.to_luma8().to_vec();
    let mut delta_weights = vec![0; 8 * 8];
    let mut delta_lat_weights = vec![0; 3 * 3];

    loop {
        // --- 1. Compute Tick ---
        // (Your activation and weight updates go here)

        // Calculate the activations
        // TODO: Optimize!!
        // TODO: Stop hardcoding stuff
        for i in 0..509 {
            for j in 0..509 {
                // Compute the dot product of the input and the kernel
                let mut sum = 0;
                for k in 0..8 {
                    for l in 0..8 {
                        let input_value = input[(i * 2 + k) * 1024 + j * 2 + l];
                        sum += input_value as i32 * weights[k * 8 + l] as i32;
                    }
                }

                // Now calculate the lateral weights
                for dx in 0..=2 {
                    for dy in 0..=2 {
                        if dx == 1 && dy == 1 {
                            continue;
                        }
                        let input = activations[(i + dx).saturating_sub(1).min(508) * 509
                            + (j + dy).saturating_sub(1).min(508)];
                        let weight = lat_weights[dx * 3 + dy];
                        sum += input as i32 * weight as i32;
                    }
                }
                next_activations[i * 509 + j] = (sum >> 7).clamp(0, 255) as u8;
            }
        }

        // Now update the weights
        for i in 0..509 {
            for j in 0..509 {
                for k in 0..8 {
                    for l in 0..8 {
                        let presynaptic = input[(i * 2 + k) * 1024 + j * 2 + l] as i32;
                        let next_presynaptic = input[(i * 2 + k) * 1024 + j * 2 + l] as i32;
                        let postsynaptic = next_activations[i * 509 + j] as i32;
                        let prev_postsynaptic = activations[i * 509 + j] as i32;
                        let delta =
                            presynaptic * postsynaptic - next_presynaptic * prev_postsynaptic;
                        delta_weights[k * 8 + l] += delta;
                    }
                }

                // Now update the lateral weights
                for dx in 0..=2 {
                    for dy in 0..=2 {
                        if dx == 1 && dy == 1 {
                            continue;
                        }
                        // Presynaptic neuron is always the current one
                        let presynaptic = activations[i * 509 + j] as i32;
                        let next_presynaptic = next_activations[i * 509 + j] as i32;
                        let postsynaptic = next_activations[(i + dx).saturating_sub(1).min(508)
                            * 509
                            + (j + dy).saturating_sub(1).min(508)]
                            as i32;
                        let prev_postsynaptic = activations[(i + dx).saturating_sub(1).min(508)
                            * 509
                            + (j + dy).saturating_sub(1).min(508)]
                            as i32;
                        let delta =
                            presynaptic * postsynaptic - next_presynaptic * prev_postsynaptic;
                        delta_lat_weights[dx * 3 + dy] += delta;
                    }
                }
            }
        }

        for i in 0..8 {
            for j in 0..8 {
                next_weights[i * 8 + j] =
                    weights[i * 8 + j] + ((delta_weights[i * 8 + j] >> 7) >> 9) as i8;
            }
        }

        for dx in 0..=2 {
            for dy in 0..=2 {
                next_lat_weights[dx * 3 + dy] =
                    lat_weights[dx * 3 + dy] + ((delta_lat_weights[dx * 3 + dy] >> 7) >> 9) as i8;
            }
        }

        // --- 2. Dual Buffer Swap ---
        std::mem::swap(&mut activations, &mut next_activations);
        std::mem::swap(&mut weights, &mut next_weights);
        std::mem::swap(&mut lat_weights, &mut next_lat_weights);

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
            save_activations(
                &activations,
                format!("weights/output_activations_{:.2}.png", total_frames).as_str(),
            );

            // Upscale 8x8 weights to 256x256 for easy viewing
            save_kernel(
                &weights,
                8,
                256,
                format!("weights/output_weights_{:.2}.png", total_frames).as_str(),
            );

            // Upscale 3x3 lateral weights to 192x192
            save_kernel(
                &lat_weights,
                3,
                192,
                format!("weights/output_lat_weights_{:.2}.png", total_frames).as_str(),
            );
        }
    }
}
