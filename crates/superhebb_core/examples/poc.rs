use std::time::Instant;

fn main() {
    // 8 pixel wide kernel with a stride of 2 over a 1024*1024 input
    // (1024 - 8) / 2 + 1 = 509 ; 509^2 = 259081
    let mut activations: Vec<i8> = vec![0; 259081];
    // 8*8 kernel w/ 1 filter and a single greyscale channel
    let mut weights: Vec<i8> = vec![0; 8 * 8];
    let mut next_activations: Vec<i8> = vec![0; 259081];
    let mut next_weights: Vec<i8> = vec![0; 8 * 8];

    // Uncapped fps we'll see how fast we go.
    let mut frame_count = 0u64;
    let mut last_fps_check = Instant::now();
    // Use a random buffer as input
    // TODO: Use a video camera or smth
    let mut input: Vec<i8> = vec![0; 1024 * 1024];

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
                activations[i * 509 + j] = sum as i8;
            }
        }

        // --- 2. Dual Buffer Swap ---
        std::mem::swap(&mut activations, &mut next_activations);
        std::mem::swap(&mut weights, &mut next_weights);

        // --- 3. FPS Counter ---
        frame_count += 1;
        let elapsed = last_fps_check.elapsed();

        if elapsed.as_secs_f64() >= 1.0 {
            let fps = frame_count as f64 / elapsed.as_secs_f64();
            println!("Performance: {:.2} FPS", fps);

            frame_count = 0;
            last_fps_check = Instant::now();
        }
    }
}
