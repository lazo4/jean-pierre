struct FrameParams {
    width: u32,
    height: u32,
    saccade_x: f32,
    saccade_y: f32,
};

@group(0) @binding(0) var raw_frame: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
// Channel packed, so even channels have OFF bits and odd channels have ON bits
// Each word is an 8x4 rectangle of pixels, to improve memory locality
@group(0) @binding(2) var<storage, read_write> layer1_input: array<u32>;
@group(0) @binding(3) var<storage, read_write> spike_counter: array<atomic<u32>>;
@group(0) @binding(4) var<uniform> params: FrameParams;
@group(0) @binding(5) var<storage, read_write> epsilon: array<f32>;

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

// Each thread processes 1 pixel. Pixels are then aggregated by subgroup into 8x4 windows
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

    let is_on = u32(diff > epsilon[0]) << lane_id;
    let is_off = u32(diff < -epsilon[0]) << lane_id;

    let word_on = subgroupOr(is_on);
    let word_off = subgroupOr(is_off);

    if lane_id == 0u {
        let words_per_row = params.width / 32u;
        let row_offset = x / 32u;
        let word_index = (y * words_per_row + row_offset) * 2u;
        layer1_input[word_index] = word_off;
        layer1_input[word_index + 1] = word_on;

        let spike_count = countOneBits(word_on) + countOneBits(word_off);
        if spike_count > 0u {
            atomicAdd(&spike_counter[0], spike_count);
        }
    }
}

const DECAY: f32 = 0.001;  // Smooth threshold decrease rate
const SPIKE: f32 = 0.019;  // 19 * DECAY -> 5% steady-state density
const TOTAL_INPUT_NEURONS: u32 = 1024 * 1024 * 2;
const DECAY_1: f32 = 1;
const SPIKE_1: f32 = 19;
const TOTAL_LAYER1_NEURONS: u32 = 1024 * 1024 * 8;

@compute @workgroup_size(1, 1, 1)
fn update_epsilon() {
    // Read total spikes AND reset counter to 0 in 1 atomic operation!
    let total_spikes = atomicExchange(&spike_counter[0], 0u);

    // Express spikes as a 0.0 to 1.0 rate
    let spike_rate = f32(total_spikes) / f32(TOTAL_INPUT_NEURONS);
    let zero_rate = 1.0 - spike_rate;

    let delta = (spike_rate * SPIKE) - (zero_rate * DECAY);
    // Update GPU epsilon directly
    epsilon[0] = clamp(epsilon[0] + delta, 0.0, 1.0);

    let total_spikes_1 = atomicExchange(&spike_counter[1], 0u);
    let spike_rate_1 = f32(total_spikes_1) / f32(TOTAL_LAYER1_NEURONS);
    let zero_rate_1 = 1.0 - spike_rate_1;

    let delta_1 = (spike_rate_1 * SPIKE_1) - (zero_rate_1 * DECAY_1);
    // Update GPU epsilon directly
    epsilon[1] = max(epsilon[1] + delta_1, 0.0);
}

// Activations for layer 1
@group(0) @binding(6) var<storage, read_write> layer1_activations: array<u32>;
@group(0) @binding(7) var<storage, read_write> layer1_next_activations: array<u32>;
@group(0) @binding(8) var<storage, read_write> layer1_vertical_weights: array<u32>;
@group(0) @binding(9) var<storage, read_write> layer1_next_vertical_weights: array<u32>;
@group(0) @binding(10) var<storage, read_write> layer1_vertical_deltas: array<u32>;

fn get_weight_word(is_on: u32, kernel: u32, row: u32, column: u32) -> u32 {
    let words_per_kernel = 8u * 8u * 2u / 4u;
    let words_per_row = 8u * 2 / 4;
    let row_offset = kernel * words_per_kernel + row * words_per_row;
    let index = row_offset + (column * 2u / 4u) + is_on;

    return layer1_vertical_weights[index];
}

fn get_input_word(is_on: u32, x: i32, y: i32) -> u32 {
    if x < 0 || y < 0 || x >= i32(params.width) || y >= i32(params.height) {
        return 0u;
    }
    let words_per_row = params.width / 32u;
    let row_offset = u32(x) / 32u;
    let word_index = (u32(y) * words_per_row + row_offset) * 2u;
    return layer1_input[word_index + is_on];
}

// Compute the inference pass of the V1 exploiting the power of subgroups
@compute @workgroup_size(256, 1, 1)
fn inference(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(subgroup_invocation_id) lane_id: u32,
) {
    // Each thread processes 1 neuron 

    var sum = 0;

    for (var row = 0; row < 8; row++) {
        for (var is_on = 0u; is_on < 2; is_on++) {
        }
    }

    // Threshold the sum and write the result
    let activations = subgroupBallot(f32(sum) > epsilon[1]).x;
    if lane_id == 0u {
        let words_per_row = params.width / 32u;
        let row_offset = global_id.x / 32u;
        let word_index = (global_id.y * words_per_row + row_offset) * 8u;
        layer1_next_activations[word_index + global_id.z] = activations;

        let spike_count = countOneBits(activations);
        if spike_count > 0u {
            atomicAdd(&spike_counter[1], spike_count);
        }
    }
}
