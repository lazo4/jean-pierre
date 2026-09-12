import cv2
import numpy as np
import time
from pyhebb import VisionSNN

def center_crop_square(frame, target_size=1024):
    h, w, _ = frame.shape
    min_dim = min(h, w)

    # Calculate bounding box for center crop
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2

    # Crop to 1:1 square ratio
    square_frame = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]

    # Resize cleanly without distortion
    return cv2.resize(square_frame, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

# Initialize Rust retina engine
snn = VisionSNN()

# Continuity Camera capture (0 or 1 depending on built-in Mac webcam setup)
cap = cv2.VideoCapture(1)

fps_counter = 0
last_time = time.time()
fps_display = "0 FPS"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Resize & convert BGR (OpenCV format) to RGBA (wgpu format)
    frame_resized = center_crop_square(frame)
    frame_rgba = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGBA)

    # 2. Step 10 GPU micro-ticks in Rust
    spike_bytes = snn.run_micro_ticks(frame_rgba.tobytes(), 10)

    # 3. Convert returned bytes back into a NumPy image
    spike_img = np.frombuffer(spike_bytes, dtype=np.uint8).reshape((1024, 1024))

    # 4. Calculate FPS
    fps_counter += 1
    if time.time() - last_time >= 1.0:
        fps_display = f"FPS: {fps_counter} | Micro-FPS: {fps_counter * 10}"
        fps_counter = 0
        last_time = time.time()

    # 5. Draw live debug HUD on the output frame
    debug_view = cv2.cvtColor(spike_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(debug_view, fps_display, (20, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Display side-by-side: Live iPhone Feed | GPU Retina Spikes
    combined = np.hstack((frame_resized, debug_view))
    cv2.imshow("iPhone Continuity Retina Simulator", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
