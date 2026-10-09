use image::GrayImage;

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

