struct FrameParams {
    width: u32,
    height: u32,
    saccade_x: f32,
    saccade_y: f32,
};

@group(0) @binding(0) var raw_frame: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
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
    // Touch the varibales to keep them in the layout
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
        output[word_index] = word_on;
        output[word_index + 1] = word_off;

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
