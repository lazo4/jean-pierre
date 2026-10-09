import time

import cv2
import numpy as np
from pyjeanpierre import VisionSNN

MICRO_TICKS_PER_FRAME = 1
CROP_SIZE = 128


def center_crop_square(frame, target_size=CROP_SIZE):
    h, w, _ = frame.shape

    # Calculate bounding box for center crop
    start_x = (w - target_size) // 2
    start_y = (h - target_size) // 2
    # Crop to 1:1 square ratio
    square_frame = frame[
        start_y : start_y + target_size, start_x : start_x + target_size
    ]
    # Resize cleanly without distortion
    return square_frame


# Initialize Rust retina engine
snn = VisionSNN(CROP_SIZE)

# Continuity Camera capture (0 or 1 depending on built-in Mac webcam setup)
cap = cv2.VideoCapture(0)

fps_counter = 0
last_time = time.time()
fps_display = "0 FPS"

# frame = cv2.imread("assets/stripes.png")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # 1. Resize & convert BGR (OpenCV format) to RGBA (wgpu format)
    frame_resized = center_crop_square(frame)
    frame_rgba = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGBA)

    # 2. Step 10 GPU micro-ticks in Rust
    (spike_bytes, activations, weights) = snn.run_micro_ticks(
        frame_rgba.tobytes(), MICRO_TICKS_PER_FRAME
    )

    # 3. Convert returned bytes back into a NumPy image
    spike_img = np.frombuffer(spike_bytes, dtype=np.uint8).reshape(
        (CROP_SIZE, CROP_SIZE)
    )
    activations = np.frombuffer(activations, dtype=np.uint8).reshape(
        (CROP_SIZE, CROP_SIZE)
    )
    weights = np.frombuffer(weights, dtype=np.uint8).reshape((5, 5))

    # 4. Calculate FPS
    fps_counter += 1
    if time.time() - last_time >= 1.0:
        fps_display = (
            f"FPS: {fps_counter} | Micro-FPS: {fps_counter * MICRO_TICKS_PER_FRAME}"
        )
        fps_counter = 0
        last_time = time.time()

    # 5. Draw live debug HUD on the output frame
    debug_view = cv2.cvtColor(spike_img, cv2.COLOR_GRAY2BGR)

    activations_view = cv2.cvtColor(activations, cv2.COLOR_GRAY2BGR)
    weights_view = cv2.cvtColor(weights, cv2.COLOR_GRAY2BGR)
    weights_view = cv2.resize(
        weights_view, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_NEAREST
    )

    # Display side-by-side: Live iPhone Feed | GPU Retina Spikes
    top = np.hstack((frame_resized, debug_view))
    bottom = np.hstack((activations_view, weights_view))
    combined = np.vstack((top, bottom))
    cv2.putText(
        combined, fps_display, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
    )

    cv2.imshow("iPhone Continuity Retina Simulator", combined)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
