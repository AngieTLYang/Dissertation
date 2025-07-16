import cv2
import os

# --- Input settings ---
video_path = rf"C:\Users\88690\Desktop\Dissertation\paddleOCR\How to improve your handwriting.mp4"
output_dir = rf"C:\Users\88690\Desktop\Dissertation\paddleOCR\frames5"
target_fps = 0.4  # Desired frame extraction rate

# --- Prepare output directory ---
os.makedirs(output_dir, exist_ok=True)

# --- Open video ---
cap = cv2.VideoCapture(video_path)
video_fps = cap.get(cv2.CAP_PROP_FPS)

if video_fps == 0:
    print("Failed to read video FPS. Please check the file path.")
    exit()

frame_interval = int(video_fps / target_fps)
count = 0
saved = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if count % frame_interval == 0:
        filename = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1

    count += 1

cap.release()
print(f"Saved {saved} frames at {target_fps} fps to: {output_dir}")
