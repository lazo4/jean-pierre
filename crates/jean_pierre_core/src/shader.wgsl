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

fn get_weight(idx: u32) -> i32 {
    let word = layer1_vertical_weights[idx / 4u];
    let weightvec = unpack4xI8(word);
    return weightvec[idx % 4u];
}

// Compute the inference pass of the V1 exploiting the power of subgroups
@compute @workgroup_size(256, 1, 1)
fn inference(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(subgroup_invocation_id) lane_id: u32,
) {
    // Each thread processes 1 neuron 
    let x = global_id.x;
    let y = global_id.y;

    var accum: i32 = 0;
    let words_per_row = params.width / 32u;

    // Loop over 5 vertical rows (-2 to +2)
    for (var ry = -2; ry <= 2; ry++) {
        let word_y = u32(clamp(i32(y) + ry, 0, i32(params.height) - 1));
        let row_offset = x / 32u;
        let word_index = (word_y * words_per_row + row_offset) * 2u;

        // 1. Fetch current 32-bit word for this subgroup (1 load covers all 32 threads)
        let word_curr = layer1_input[word_index + 1];
        let my_act = ((word_curr >> lane_id) & 1u);

        // 2. Exchange activation bits across thread registers in 0 cycles
        var act_m2 = subgroupShuffleUp(my_act, 2u) != 0u;
        var act_m1 = subgroupShuffleUp(my_act, 1u) != 0u;
        let act_p0 = my_act != 0u;
        var act_p1 = subgroupShuffleDown(my_act, 1u) != 0u;
        var act_p2 = subgroupShuffleDown(my_act, 2u) != 0u;

        // 3. Fallback loads for 32-bit word boundaries (lanes 0, 1, 30, 31)
        if lane_id < 2u {
            if row_offset > 0u {
                let word_prev = layer1_input[word_index - 2 + 1];
                if lane_id == 0u {
                    act_m2 = ((word_prev >> 30u) & 1u) != 0u;
                    act_m1 = ((word_prev >> 31u) & 1u) != 0u;
                } else {
                    act_m2 = ((word_prev >> 31u) & 1u) != 0u;
                }
            } else {
                act_m2 = false;
                if lane_id == 0u { act_m1 = false; }
            }
        }

        if lane_id >= 30u {
            if row_offset + 2u + 1 < words_per_row {
                let word_next = layer1_input[word_index + 2 + 1];
                if lane_id == 31u {
                    act_p1 = ((word_next >> 0u) & 1u) != 0u;
                    act_p2 = ((word_next >> 1u) & 1u) != 0u;
                } else {
                    act_p2 = ((word_next >> 0u) & 1u) != 0u;
                }
            } else {
                act_p2 = false;
                if lane_id == 31u { act_p1 = false; }
            }
        }

        // 4. Conditional accumulation via WGSL select() (no branching)
        let w_base = u32(ry + 2) * 5u;
        accum += select(0, get_weight(w_base + 0u), act_m2);
        accum += select(0, get_weight(w_base + 1u), act_m1);
        accum += select(0, get_weight(w_base + 2u), act_p0);
        accum += select(0, get_weight(w_base + 3u), act_p1);
        accum += select(0, get_weight(w_base + 4u), act_p2);
    }
    // Threshold the sum and write the result
    let activations = subgroupBallot(f32(accum) > epsilon[1]).x;
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
