struct FrameParams {
    width: u32,
    height: u32,
    saccade_x: f32,
    saccade_y: f32,
};

@group(0) @binding(0) var raw_frame: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
// Channel packed, so even channels have ON bits and odd channels have OFF bits
@group(0) @binding(2) var<storage, read_write> layer1_input: array<u32>;
@group(0) @binding(3) var<storage, read_write> spike_counter: atomic<u32>;
@group(0) @binding(4) var<uniform> params: FrameParams;
@group(0) @binding(5) var<storage, read_write> epsilon: f32;

fn get_luminance(coords: vec2f) -> f32 {
    let jittered_pos = coords + vec2f(params.saccade_x, params.saccade_y);

    // Normalize coordinates to UV space [0.0, 1.0] and clamp edges
    let uv = clamp(
        jittered_pos / vec2f(f32(params.width), f32(params.height)),
        vec2f(0.0),
        vec2f(1.0)
    );
    // Filtered sub-pixel sample
    let rgba = textureSampleLevel(raw_frame, input_sampler, uv, 0.0);
    return dot(rgba.rgb, vec3f(0.2126, 0.7152, 0.0722));
}

// Each thread processes 1 pixel. Pixels are then aggregated by subgroup
@compute @workgroup_size(256, 1, 1)
fn preprocess_frame(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(subgroup_invocation_id) lane_id: u32,
) {
    let x = global_id.x;
    let y = global_id.y;

    if x >= params.width || y >= params.height { return; }

    let coords = vec2f(f32(x), f32(y));

    let center_lum = get_luminance(coords);
    var surround_sum: f32 = 0.0;

    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            // if dx == 0 && dy == 0 { continue; }
            surround_sum += get_luminance(coords + vec2f(f32(dx), f32(dy)));
        }
    }

    let diff = center_lum - (surround_sum * 0.11111111);

    // Now shift by SIMD lane id

    let is_on = u32(diff > epsilon) << lane_id;
    let is_off = u32(diff < -epsilon) << lane_id;

    let word_on = subgroupOr(is_on);
    let word_off = subgroupOr(is_off);

    if lane_id == 0u {
        let words_per_row = params.width / 32u;
        let row_offset = x / 32u;
        let word_index = (y * words_per_row + row_offset) * 2u;
        layer1_input[word_index] = word_on;
        layer1_input[word_index + 1] = word_off;

        let spike_count = countOneBits(word_on) + countOneBits(word_off);
        if spike_count > 0u {
            atomicAdd(&spike_counter, spike_count);
        }
    }
}

const DECAY: f32 = 0.0001;  // Smooth threshold decrease rate
const SPIKE: f32 = 0.0019;  // 19 * DECAY -> 5% steady-state density
const TOTAL_NEURONS: u32 = 1024 * 1024 * 2;

@compute @workgroup_size(1, 1, 1)
fn update_epsilon() {
    // Read total spikes AND reset counter to 0 in 1 atomic operation!
    let total_spikes = atomicExchange(&spike_counter, 0u);

    // Express spikes as a 0.0 to 1.0 rate
    let spike_rate = f32(total_spikes) / f32(TOTAL_NEURONS);
    let zero_rate = 1.0 - spike_rate;

    let delta = (spike_rate * SPIKE) - (zero_rate * DECAY);
    // Update GPU epsilon directly
    epsilon = clamp(epsilon + delta, 0.0, 1.0);
}

// Activations for layer 1
// Indexed by y * 8 * params.width / 32 + x * 8 / 32 + kernel 
@group(0) @binding(6) var<storage, read_write> layer1_activations: array<u32>;
@group(0) @binding(7) var<storage, read_write> layer1_next_activations: array<u32>;
@group(0) @binding(8) var<storage, read_write> layer1_vertical_weights: array<u32>;
@group(0) @binding(9) var<storage, read_write> layer1_next_vertical_weights: array<u32>;
@group(0) @binding(10) var<storage, read_write> layer1_vertical_deltas: array<u32>;
@group(0) @binding(11) var<storage, read_write> layer1_epsilon: array<u32>;

fn get_weight(is_off: bool, kernel: u32, row: u32, bit_idx: u32) -> u32 {
    // 1. Calculate linear byte offset (0..1023)
    let byte_idx = select(0u, 512u, is_off) + kernel * 8 * 8 + row * 8 + bit_idx;

    let word_idx = byte_idx / 4;          // byte_idx / 4
    let bit_offset = (byte_idx % 4) * 8u;    // (byte_idx % 4) * 8

    // 3. Single clock cycle hardware bitfield extraction
    return extractBits(layer1_vertical_weights[word_idx], bit_offset, 8u);
}

fn get_input_word(is_off: bool, x: u32, y: u32) -> u32 {
    let word_idx = (y * params.width / 32u + x / 32u) * 2u + select(0u, 1u, is_off);
    return layer1_input[word_idx];
}

// Compute the inference pass of the V1 exploiting the power of subgroups
@compute @workgroup_size(256, 1, 1)
fn inference(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(subgroup_invocation_id) lane_id: u32,
) {
    // Each thread processes 1 neuron 
    // Each subgroup processes 32 neurons = 1 word worth of activations, accumulated through subgroupBallot

    let kernel = global_id.z;

    // The current sum of all the active weights
    // This will be thresholded in the end
    var sum: i32 = 0;

    // Start by computing the vertical weights
    // Iterate over the rows
    for (var row = 0u; row < 8; row++) {
        // Extract the relevant input bits
        let input_word = get_input_word(false, global_id.x, global_id.y + row);
        let next_input_word = get_input_word(true, global_id.x + 1, global_id.y + row);

        let low_bits = input_word >> lane_id;
        let high_bits = select(0, next_input_word << (32 - lane_id), lane_id > 24);

        let input_bits = (low_bits | high_bits) & 0xFF;

        // Now and and shift for each bitplane
        // NOTE: Since the weights are signed, bit 7 is treated specially
        for (var bitplane = 0u; bitplane < 7; bitplane++) {
            let weight = get_weight(false, kernel, row, bitplane);
            sum += i32(countOneBits(input_bits & weight) << bitplane);
        }
        // The rest of the numbers must be sign extended (subtracted)
        let weight = get_weight(false, kernel, row, 7);
        sum -= i32(countOneBits(input_bits & weight) << 7);
    }

    let activation = subgroupBallot(sum > layer1_epsilon).x;

    // Write the activation (once per subgroup)
    if lane_id == 0u {
        let out_idx = 0;
        layer1_next_activations[out_idx] = activation;
    }
}

