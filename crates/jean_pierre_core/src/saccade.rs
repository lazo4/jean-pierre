use rand::{Rng, RngExt};

pub struct SaccadeGenerator<T: Rng> {
    pub x: f32,
    pub y: f32,
    pub(crate) decay: f32,     // Restoring force toward center (0.0, 0.0)
    pub(crate) step_size: f32, // Drift velocity scale (sub-pixel)
    pub(crate) rng: T,
}

impl<T: Rng> SaccadeGenerator<T> {
    pub fn new(rng: T) -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            decay: 0.85,     // Pulls back to center each tick
            step_size: 0.15, // Max ~0.15 pixel step per frame
            rng,
        }
    }

    /// Call this every game loop frame tick (e.g., at 60 FPS)
    pub fn update(&mut self) -> (f32, f32) {
        // 1. Add random Gaussian-like step
        let dx: f32 = self.rng.random_range(-self.step_size..self.step_size);
        let dy: f32 = self.rng.random_range(-self.step_size..self.step_size);

        // 2. Apply step + decay (prevents the image from drifting off to infinity)
        self.x = (self.x + dx) * self.decay;
        self.y = (self.y + dy) * self.decay;

        (self.x, self.y)
    }
}
