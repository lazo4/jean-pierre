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
@group(0) @binding(3) var<storage, read_write> spike_counter: array<f32>;
@group(0) @binding(4) var<uniform> params: FrameParams;
@group(0) @binding(5) var<storage, read_write> input_epsilon: array<f32>;

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

const DECAY: f32 = 0.0001;  // Smooth threshold decrease rate
const SPIKE: f32 = 0.0019;  // 19 * DECAY -> 5% steady-state density
const EPSILON_BASE: f32 = 0.01;

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

    let epsilon_idx = (y * params.width + x) * 2;
    let off_epsilon = input_epsilon[epsilon_idx];
    let on_epsilon = input_epsilon[epsilon_idx + 1];

    let is_off = diff < -off_epsilon;
    let is_on = diff > on_epsilon;

    if is_off {
        let overshot = off_epsilon + diff;
        input_epsilon[epsilon_idx] = off_epsilon + max(overshot * 1.01, SPIKE);
    } else {
        let undershot = diff - off_epsilon;
        input_epsilon[epsilon_idx] = max(off_epsilon - DECAY, EPSILON_BASE);
    }

    if is_on {
        let overshot = diff - on_epsilon;
        input_epsilon[epsilon_idx + 1] = on_epsilon + max(overshot * 1.01, SPIKE);
    } else {
        let undershot = on_epsilon - diff;
        input_epsilon[epsilon_idx + 1] = max(on_epsilon - DECAY, EPSILON_BASE);
    }

    let word_on = subgroupBallot(is_on).x;
    let word_off = subgroupBallot(is_off).x;

    if lane_id == 0u {
        let words_per_row = params.width / 32u;
        let row_offset = x / 32u;
        let word_index = (y * words_per_row + row_offset) * 2u;
        layer1_input[word_index] = word_off;
        layer1_input[word_index + 1] = word_on;
    }
}

const TOTAL_INPUT_NEURONS: u32 = 512 * 512 * 2;
const DECAY_1: f32 = 1;
const SPIKE_1: f32 = 19;
const TOTAL_LAYER1_NEURONS: u32 = 512 * 512 * 8;

@compute @workgroup_size(256, 1, 1)
fn write_spike_counts(
    @builtin(global_invocation_id) global_id: vec3u,
) {
    // NOTE: Since we have to keep our workgroups one dimensional, each workgroup is going to process several rows
    let rows_per_workgroup = PIXELS_PER_WORKGROUP / params.width;
    let y = global_id.y * rows_per_workgroup + (global_id.x * 32) / params.width;
    let x_begin = (global_id.x * 32) % params.width;

    if x_begin >= params.width || y >= params.height { return; }

    let words_per_row = params.width / 32u;
    let row_offset = x_begin / 32u;
    let word_index = (y * words_per_row + row_offset) * 2u;

    let x_center = f32(x_begin) + 15.5;
}

// Activations for layer 1
@group(0) @binding(6) var<storage, read_write> layer1_activations: array<u32>;
@group(0) @binding(7) var<storage, read_write> layer1_next_activations: array<u32>;
@group(0) @binding(8) var<storage, read_write> layer1_vertical_weights: array<u32>;
@group(0) @binding(9) var<storage, read_write> layer1_next_vertical_weights: array<u32>;
@group(0) @binding(10) var<storage, read_write> layer1_vertical_deltas: array<atomic<i32>>;

// TODO: Process weights 4 at a time!!
fn get_weight(idx: u32) -> i32 {
    let word = layer1_vertical_weights[idx / 4u];
    let weightvec = unpack4xI8(word);
    return weightvec[idx % 4u];
}

// TODO: Same as above
fn set_weight(idx: u32, val: i32) {
    let word = layer1_next_vertical_weights[idx / 4u];
    var weightvec = unpack4xI8(word);
    weightvec[idx % 4u] = val;
    layer1_next_vertical_weights[idx / 4u] = pack4xI8(weightvec);
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
    /*let activations = subgroupBallot(f32(accum) > epsilon[1]).x;
    if lane_id == 0u {
        let words_per_row = params.width / 32u;
        let row_offset = global_id.x / 32u;
        let word_index = (global_id.y * words_per_row + row_offset) * 8u;
        layer1_next_activations[word_index + global_id.z] = activations;
    }*/
}

const ETA = 0.001;
const PIXELS_PER_WORKGROUP = 256 * 32;

@compute @workgroup_size(256, 1, 1)
fn compute_stdp_deltas(
    @builtin(global_invocation_id) global_id: vec3u,
    @builtin(subgroup_invocation_id) lane_id: u32,
) {
    // Each thread updates all 25 weights for 32 neurons
    // NOTE: Since we have to keep our workgroups one dimensional, each workgroup is going to process several rows
    let rows_per_workgroup = PIXELS_PER_WORKGROUP / params.width;
    let y = global_id.y * rows_per_workgroup + (global_id.x * 32) / params.width;
    let x_begin = (global_id.x * 32) % params.width;

    if x_begin >= params.width || y >= params.height { return; }

    let words_per_row = params.width / 32u;
    let row_offset = x_begin / 32u;
    let word_index = (y * words_per_row + row_offset) * 8u;
    let prev_postsyn = layer1_activations[word_index + global_id.z];
    let next_postsyn = layer1_next_activations[word_index + global_id.z];

    for (var ry = -2; ry <= 2; ry++) {
        let presyn_y = i32(y) + ry;

        // Handle vertical zero-padding bounds
        if presyn_y < 0 || presyn_y >= i32(params.height) {
            continue;
        }

        let word_index = (u32(presyn_y) * words_per_row + row_offset) * 2u;

        let presyn_word_curr = layer1_input[word_index + 1];
        var presyn_word_prev = 0u;
        var presyn_word_next = 0u;
        if row_offset > 0u {
            presyn_word_prev = layer1_input[word_index - 2 + 1];
        }
        if row_offset + 2u + 1 < words_per_row {
            presyn_word_next = layer1_input[word_index + 2 + 1];
        }

        for (var rx = -2; rx <= 2; rx++) {
            let weight_idx = u32(ry + 2) * 5u + u32(rx + 2);

            // --- Bit-Align Presynaptic Window for Offset rx ---
            var presyn: u32;
            if rx < 0 {
                let shift = u32(-rx);
                presyn = (presyn_word_curr << shift) | (presyn_word_prev >> (32u - shift));
            } else if rx > 0 {
                let shift = u32(rx);
                presyn = (presyn_word_curr >> shift) | (presyn_word_next << (32u - shift));
            } else {
                presyn = presyn_word_curr;
            }

            let prev_presyn = presyn;
            let next_presyn = prev_presyn; // TODO: Soundness: Previous and next inputs should be separate too...

            // The beautiful part
            let delta = i32(countOneBits(prev_presyn & next_postsyn)) - i32(countOneBits(next_presyn & prev_postsyn));
            atomicAdd(&layer1_vertical_deltas[weight_idx], delta);
        }
    }
}

@compute @workgroup_size(25, 1, 1)
fn apply_stdp_deltas(
    @builtin(global_invocation_id) global_id: vec3u,
) {
    let weight_idx = global_id.x;
    let delta = atomicExchange(&layer1_vertical_deltas[weight_idx], 0);
    let weight = clamp(get_weight(weight_idx) + i32(f32(delta) * ETA), -128, 127);
    set_weight(weight_idx, weight);
}

