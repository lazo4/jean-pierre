struct FrameParams {
    width: u32,
    height: u32,
    saccade_x: f32,
    saccade_y: f32,
};

@group(0) @binding(0) var raw_frame: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
// Channel packed, so even channels have OFF bits and odd channels have ON bits
@group(0) @binding(2) var<storage, read_write> layer1_input: array<u32>;
@group(0) @binding(3) var<storage, read_write> layer1_next_input: array<u32>;
@group(0) @binding(4) var<uniform> params: FrameParams;
@group(0) @binding(5) var<storage, read_write> epsilon: array<f32>;
@group(0) @binding(6) var<storage, read_write> spike_counts: array<atomic<u32>>;

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

    let off_epsilon = epsilon[0];
    let on_epsilon = epsilon[0];

    let words_per_row = params.width / 32u;
    let row_offset = x / 32u;
    let word_index = (y * words_per_row + row_offset) * 2u;
    let prev_word_off = layer1_input[word_index];
    let prev_word_on = layer1_input[word_index + 1];

    let prev_off = ((prev_word_off >> lane_id) & 1u) == 1u;
    let prev_on = ((prev_word_on >> lane_id) & 1u) == 1u;

    let is_off = (diff < -off_epsilon) && !prev_off;
    let is_on = (diff > on_epsilon) && !prev_on;

    let word_on = subgroupBallot(is_on).x;
    let word_off = subgroupBallot(is_off).x;

    if lane_id == 0u {
        let words_per_row = params.width / 32u;
        let row_offset = x / 32u;
        let word_index = (y * words_per_row + row_offset) * 2u;
        layer1_next_input[word_index] = word_off;
        layer1_next_input[word_index + 1] = word_on;

        let spike_count = countOneBits(word_on) + countOneBits(word_off);
        if spike_count > 0u {
            atomicAdd(&spike_counts[0], spike_count);
        }
    }
}

const DECAY: f32 = 0.0001;  // Smooth threshold decrease rate
const SPIKE: f32 = 0.0019;  // 19 * DECAY -> 5% steady-state density
const EPSILON_BASE: f32 = 0.015;

const DECAY_1: f32 = 100;
const SPIKE_1: f32 = 1900;
const INFERENCE_EPSILON_BASE = 1;

@compute @workgroup_size(1, 1, 1)
fn update_epsilon() {
    // Read total spikes AND reset counter to 0 in 1 atomic operation!
    let total_spikes = atomicExchange(&spike_counts[0], 0u);

    // Express spikes as a 0.0 to 1.0 rate
    let spike_rate = f32(total_spikes) / f32(128 * 128 * 2);
    let zero_rate = 1.0 - spike_rate;

    let delta = (spike_rate * SPIKE) - (zero_rate * DECAY);

    // Update GPU epsilon directly
    epsilon[0] = max(epsilon[0] + delta, EPSILON_BASE);

    let total_spikes_1 = atomicExchange(&spike_counts[1], 0u);
    let spike_rate_1 = f32(total_spikes_1) / f32(128 * 128 * 1);
    let zero_rate_1 = 1.0 - spike_rate_1;

    let delta_1 = (spike_rate_1 * SPIKE_1) - (zero_rate_1 * DECAY_1);
    // Update GPU epsilon directly
    epsilon[1] = max(epsilon[1] + delta_1, INFERENCE_EPSILON_BASE);
}

// Activations for layer 1
@group(0) @binding(7) var<storage, read_write> layer1_activations: array<u32>;
@group(0) @binding(8) var<storage, read_write> layer1_next_activations: array<u32>;
@group(0) @binding(9) var<storage, read_write> layer1_vertical_weights: array<u32>;
@group(0) @binding(10) var<storage, read_write> layer1_next_vertical_weights: array<u32>;
@group(0) @binding(11) var<storage, read_write> layer1_vertical_deltas: array<atomic<i32>>;

// TODO: Process weights 4 at a time!!
fn get_weight(idx: u32, is_on: bool) -> i32 {
    let word_idx = (idx / 4u) + u32(is_on) * 7;
    let word = layer1_vertical_weights[word_idx];
    let weightvec = unpack4xI8(word);
    return weightvec[idx % 4u];
}

// Count set bits of `word` at positions [lo, hi], clamped to [0, 31]
fn bits_in(word: u32, lo: i32, hi: i32) -> u32 {
    let l = max(lo, 0);
    let h = min(hi, 31);
    if l > h { return 0; }
    return extractBits(word, u32(l), u32(h - l + 1));
}

// Compute the inference pass of the V1 exploiting the power of subgroups
@compute @workgroup_size(128, 1, 1)
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
        let off_curr = layer1_input[word_index];
        let on_curr = layer1_input[word_index + 1];
        let my_off = ((off_curr >> lane_id) & 1u);
        let my_on = ((on_curr >> lane_id) & 1u);

        // 2. Exchange activation bits across thread registers in 0 cycles
        var off_m2 = subgroupShuffleUp(my_off, 2u) != 0u;
        var on_m2 = subgroupShuffleUp(my_on, 2u) != 0u;
        var off_m1 = subgroupShuffleUp(my_off, 1u) != 0u;
        var on_m1 = subgroupShuffleUp(my_on, 1u) != 0u;
        let off_p0 = my_off != 0u;
        let on_p0 = my_on != 0u;
        var off_p1 = subgroupShuffleDown(my_off, 1u) != 0u;
        var on_p1 = subgroupShuffleDown(my_on, 1u) != 0u;
        var off_p2 = subgroupShuffleDown(my_off, 2u) != 0u;
        var on_p2 = subgroupShuffleDown(my_on, 2u) != 0u;

        // 3. Fallback loads for 32-bit word boundaries (lanes 0, 1, 30, 31)
        if lane_id < 2u {
            if row_offset > 0u {
                let off_word_prev = layer1_input[word_index - 2];
                let on_word_prev = layer1_input[word_index - 2 + 1];
                if lane_id == 0u {
                    off_m2 = ((off_word_prev >> 30u) & 1u) != 0u;
                    on_m2 = ((on_word_prev >> 30u) & 1u) != 0u;
                    off_m1 = ((off_word_prev >> 31u) & 1u) != 0u;
                    on_m1 = ((on_word_prev >> 31u) & 1u) != 0u;
                } else {
                    off_m2 = ((off_word_prev >> 31u) & 1u) != 0u;
                    on_m2 = ((on_word_prev >> 31u) & 1u) != 0u;
                }
            } else {
                off_m2 = false;
                on_m2 = false;
                if lane_id == 0u { off_m1 = false; on_m1 = false; }
            }
        }

        if lane_id >= 30u {
            if row_offset + 2u + 1 < words_per_row {
                let off_word_next = layer1_input[word_index + 2];
                let on_word_next = layer1_input[word_index + 2 + 1];
                if lane_id == 31u {
                    off_p1 = ((off_word_next >> 0u) & 1u) != 0u;
                    on_p1 = ((on_word_next >> 0u) & 1u) != 0u;
                    off_p2 = ((off_word_next >> 1u) & 1u) != 0u;
                    on_p2 = ((on_word_next >> 1u) & 1u) != 0u;
                } else {
                    off_p2 = ((off_word_next >> 0u) & 1u) != 0u;
                    on_p2 = ((on_word_next >> 0u) & 1u) != 0u;
                }
            } else {
                off_p2 = false;
                on_p2 = false;
                if lane_id == 31u { off_p1 = false; on_p1 = false; }
            }
        }

        // 4. Conditional accumulation via WGSL select() (no branching)
        let w_base = u32(ry + 2) * 5u;
        accum += select(0, get_weight(w_base + 0u, false), off_m2);
        accum += select(0, get_weight(w_base + 0u, true), on_m2);
        accum += select(0, get_weight(w_base + 1u, false), off_m1);
        accum += select(0, get_weight(w_base + 1u, true), on_m1);
        accum += select(0, get_weight(w_base + 2u, false), off_p0);
        accum += select(0, get_weight(w_base + 2u, true), on_p0);
        accum += select(0, get_weight(w_base + 3u, false), off_p1);
        accum += select(0, get_weight(w_base + 3u, true), on_p1);
        accum += select(0, get_weight(w_base + 4u, false), off_p2);
        accum += select(0, get_weight(w_base + 4u, true), on_p2);
    }

    // Threshold the sum and write the result
    let row_offset = global_id.x / 32u;
    let word_index = (global_id.y * words_per_row + row_offset) * 8u;
    let prev_word = layer1_activations[word_index + global_id.z];
    let prev_activation = ((prev_word >> lane_id) & 1u) == 1u;

    let activations = subgroupBallot((f32(accum) > epsilon[1]) && !prev_activation).x;
    if lane_id == 0u {
        let words_per_row = params.width / 32u;
        let row_offset = global_id.x / 32u;
        let word_index = (global_id.y * words_per_row + row_offset) * 8u;
        layer1_next_activations[word_index + global_id.z] = activations;

        let spike_count = countOneBits(activations);
        if spike_count > 0u {
            atomicAdd(&spike_counts[1], spike_count);
        }
    }
}

const ETA = 0.02;
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

        let presyn_off_curr = layer1_input[word_index];
        let presyn_on_curr = layer1_input[word_index + 1];
        let next_presyn_off_curr = layer1_next_input[word_index];
        let next_presyn_on_curr = layer1_next_input[word_index + 1];
        var presyn_off_prev = 0u;
        var presyn_on_prev = 0u;
        var next_presyn_off_prev = 0u;
        var next_presyn_on_prev = 0u;
        var presyn_off_next = 0u;
        var presyn_on_next = 0u;
        var next_presyn_off_next = 0u;
        var next_presyn_on_next = 0u;

        if row_offset > 0u {
            presyn_off_prev = layer1_input[word_index - 2];
            presyn_on_prev = layer1_input[word_index - 2 + 1];
            next_presyn_off_prev = layer1_next_input[word_index - 2];
            next_presyn_on_prev = layer1_next_input[word_index - 2 + 1];
        }
        if row_offset + 2u + 1 < words_per_row {
            presyn_off_next = layer1_input[word_index + 2];
            presyn_on_next = layer1_input[word_index + 2 + 1];
            next_presyn_off_next = layer1_next_input[word_index + 2];
            next_presyn_on_next = layer1_next_input[word_index + 2 + 1];
        }

        for (var rx = -2; rx <= 2; rx++) {
            let off_weight_idx = (u32(ry + 2) * 5u + u32(rx + 2)) * 2u;
            let on_weight_idx = (u32(ry + 2) * 5u + u32(rx + 2) + 25) * 2u;

            // --- Bit-Align Presynaptic Window for Offset rx ---
            var off_presyn: u32;
            var on_presyn: u32;
            var next_off_presyn: u32;
            var next_on_presyn: u32;
            if rx < 0 {
                let shift = u32(-rx);
                off_presyn = (presyn_off_curr << shift) | (presyn_off_prev >> (32u - shift));
                on_presyn = (presyn_on_curr << shift) | (presyn_on_prev >> (32u - shift));
                next_off_presyn = (next_presyn_off_curr << shift) | (next_presyn_off_prev >> (32u - shift));
                next_on_presyn = (next_presyn_on_curr << shift) | (next_presyn_on_prev >> (32u - shift));
            } else if rx > 0 {
                let shift = u32(rx);
                off_presyn = (presyn_off_curr >> shift) | (presyn_off_next << (32u - shift));
                on_presyn = (presyn_on_curr >> shift) | (presyn_on_next << (32u - shift));
                next_off_presyn = (next_presyn_off_curr >> shift) | (next_presyn_off_next << (32u - shift));
                next_on_presyn = (next_presyn_on_curr >> shift) | (next_presyn_on_next << (32u - shift));
            } else {
                off_presyn = presyn_off_curr;
                on_presyn = presyn_on_curr;
                next_off_presyn = next_presyn_off_curr;
                next_on_presyn = next_presyn_on_curr;
            }

            let prev_off_presyn = off_presyn;
            let prev_on_presyn = on_presyn;

            // The beautiful part
            let off_ltp = i32(countOneBits(prev_off_presyn & next_postsyn));
            let off_ltd = i32(countOneBits(next_off_presyn & prev_postsyn));
            let on_ltp = i32(countOneBits(prev_on_presyn & next_postsyn));
            let on_ltd = i32(countOneBits(next_on_presyn & prev_postsyn));

            atomicAdd(&layer1_vertical_deltas[off_weight_idx], off_ltp);
            atomicAdd(&layer1_vertical_deltas[off_weight_idx + 1], off_ltd);
            atomicAdd(&layer1_vertical_deltas[on_weight_idx], on_ltp);
            atomicAdd(&layer1_vertical_deltas[on_weight_idx + 1], on_ltd);
        }
    }
}

@compute @workgroup_size(7, 1, 1)
fn apply_stdp_deltas(
    @builtin(global_invocation_id) global_id: vec3u,
) {
    let weight_idx = global_id.x;
    var off_weights = unpack4xI8(layer1_vertical_weights[weight_idx]);
    var on_weights = unpack4xI8(layer1_vertical_weights[weight_idx + 7]);

    for (var i = 0u; i < 4u; i++) {
        let off_idx = (weight_idx * 4u + i) * 2u;
        let on_idx = (weight_idx * 4u + i + 25) * 2u;
        if off_idx < 50u {
            let off_ltp = f32(atomicExchange(&layer1_vertical_deltas[off_idx], 0));
            let off_ltd = f32(atomicExchange(&layer1_vertical_deltas[off_idx + 1], 0));
            let on_ltp = f32(atomicExchange(&layer1_vertical_deltas[on_idx], 0));
            let on_ltd = f32(atomicExchange(&layer1_vertical_deltas[on_idx + 1], 0));

            let off_delta = (off_ltp * ETA) - (off_ltd * ETA);
            let on_delta = (on_ltp * ETA) - (on_ltd * ETA);

            off_weights[i] = clamp(off_weights[i] + i32(f32(off_delta)), -128, 127);
            on_weights[i] = clamp(on_weights[i] + i32(f32(on_delta)), -128, 127);
        }
    }
    layer1_next_vertical_weights[weight_idx] = pack4xI8(off_weights);
    layer1_next_vertical_weights[weight_idx + 7] = pack4xI8(on_weights);
}

