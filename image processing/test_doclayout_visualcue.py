import cv2
import numpy as np
import math # For Euclidean distance
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download

# --- 1. Load the DocLayout-YOLO model ---
print("Loading DocLayout-YOLO model...")
try:
    filepath = hf_hub_download(repo_id="juliozhao/DocLayout-Yolo-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
    model = YOLOv10(filepath)
    print("DocLayout-YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading DocLayout-YOLO model: {e}")
    print("Please ensure 'doclayout-yolo' is installed and you have network access to download the model.")
    exit()

# --- 2. Define the image path ---
image_path = "leafletwithpen.png"

# Load the original image for both DocLayout prediction and custom color processing
original_image_bgr = cv2.imread(image_path)
if original_image_bgr is None:
    print(f"Error: Could not load image at {image_path}. Make sure the image file is in the same directory.")
    exit()

# --- 3. Perform DocLayout prediction ---
print(f"Performing DocLayout prediction on {image_path}...")
try:
    det_res = model.predict(
        image_path,
        imgsz=1024, # Prediction image size
        conf=0.2,   # Confidence threshold
        device="cpu" # Device to use (e.g., 'cuda:0' or 'cpu')
    )
    print("DocLayout prediction complete.")
except Exception as e:
    print(f"Error during DocLayout prediction: {e}")
    exit()

# --- 4. Prepare annotated frame for visualization ---
# det_res[0].plot(pil=True) returns a PIL Image if pil=True
annotated_frame_pil = det_res[0].plot(pil=True, line_width=5, font_size=20)
# Convert PIL Image to OpenCV BGR format
annotated_frame = np.array(annotated_frame_pil)
annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

# --- 5. Detect the Red Pen using HSV color segmentation ---
print("Detecting red pen...")
hsv_image = cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2HSV)

# Define HSV ranges for red color (red wraps around 0 and 180 degrees in OpenCV's Hue channel)
# Lower red range (Hue: 0-10, Saturation: 100-255, Value: 100-255)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)

# Upper red range (Hue: 170-180, Saturation: 100-255, Value: 100-255)
lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

# Combine the two masks to get the full red mask
red_mask = cv2.add(mask1, mask2)

# Optional: Apply morphological operations to clean up the mask
# Adjust kernel size (e.g., 5x5, 7x7) based on image resolution and noise
kernel = np.ones((7,7), np.uint8) 
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)  # Erosion followed by dilation to remove small noise
red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel) # Dilation followed by erosion to close small holes

# Find contours (outlines) of the red regions
# cv2.findContours returns a tuple, the second element is the hierarchy which we don't need here
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

pen_bbox_xywh = None
pen_center = None

if contours:
    # Find the largest contour by area, assuming it's the pen
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Filter out very small contours that might be noise (adjust threshold as needed)
    if cv2.contourArea(largest_contour) > 500: # Increased threshold for better pen detection
        pen_bbox_xywh = cv2.boundingRect(largest_contour) # Get (x, y, width, height)
        
        # Calculate the center of the detected pen
        x, y, w, h = pen_bbox_xywh
        pen_center = (x + w / 2, y + h / 2)

        # Draw pen bounding box and center on the annotated frame for visualization
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 255), 3) # Yellow rectangle for pen
        cv2.circle(annotated_frame, (int(pen_center[0]), int(pen_center[1])), 8, (0, 0, 255), -1) # Red dot for pen center
        print(f"Red pen detected at bounding box: {pen_bbox_xywh} with center: ({pen_center[0]:.0f}, {pen_center[1]:.0f})")
    else:
        print("No significant red object (pen) detected based on area threshold. Adjust '500' if needed.")
else:
    print("No red contours found in the image. Check image for red objects or adjust HSV ranges.")

# --- 6. Find the Closest Text Block ---
print("Finding closest text block...")

if pen_center is None:
    print("Cannot determine closest text block as the pen was not detected.")
else:
    min_distance = float('inf')
    closest_text_block_info = None

    detected_boxes = det_res[0].boxes # Access the boxes object from the first prediction result
    class_names = det_res[0].names    # Get the mapping of class IDs to names

    if len(detected_boxes) == 0:
        print("No document elements were detected by DocLayout-YOLO.")
    else:
        # Iterate through all detected elements
        for i in range(len(detected_boxes)):
            # Get bounding box coordinates in xyxy format (x1, y1, x2, y2)
            bbox_xyxy = detected_boxes.xyxy[i].cpu().numpy().astype(int) 
            class_id = int(detected_boxes.cls[i].item()) # Get class ID
            class_name = class_names[class_id]         # Get class name

            # We are interested in "text-like" blocks. 
            # DocLayout-YOLO typically outputs classes like 'text', 'title', 'list_item', 'paragraph' etc.
            # You might want to customize this list based on the exact classes you care about.
            text_related_classes = ["plain text"]

            if class_name in text_related_classes:
                x1, y1, x2, y2 = bbox_xyxy
                block_center_x = (x1 + x2) / 2
                block_center_y = (y1 + y2) / 2
                block_center = (block_center_x, block_center_y)

                distance = math.dist(pen_center, block_center)

                if distance < min_distance:
                    min_distance = distance
                    closest_text_block_info = {
                        'bbox': bbox_xyxy,
                        'class_name': class_name,
                        'center': block_center,
                        'distance_to_pen': min_distance
                    }
        
        # --- 7. Output the result and visualize the closest block ---
        if closest_text_block_info:
            print("\n--- Closest Text Block to Red Pen ---")
            print(f"Class Name: {closest_text_block_info['class_name']}")
            print(f"Bounding Box (x1, y1, x2, y2): {closest_text_block_info['bbox']}")
            print(f"Center: ({closest_text_block_info['center'][0]:.0f}, {closest_text_block_info['center'][1]:.0f})")
            print(f"Distance to Pen Center: {closest_text_block_info['distance_to_pen']:.2f} pixels")

            # Draw the bounding box of the closest text block in a distinct color (green)
            x1, y1, x2, y2 = closest_text_block_info['bbox']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3) # Green rectangle
            cv2.circle(annotated_frame, (int(closest_text_block_info['center'][0]), int(closest_text_block_info['center'][1])), 8, (0, 255, 0), -1) # Green dot for closest block center
            # Draw a green line connecting the pen center to the closest text block center
            cv2.line(annotated_frame, (int(pen_center[0]), int(pen_center[1])), 
                     (int(closest_text_block_info['center'][0]), int(closest_text_block_info['center'][1])), 
                     (0, 255, 0), 2) 
        else:
            print("No text-related blocks detected by DocLayout-YOLO or no pen detected.")

# --- 8. Save the final annotated image ---
output_image_path = "result_with_pen_and_closest_block.jpg"
cv2.imwrite(output_image_path, annotated_frame)
print(f"\nFinal annotated image saved as {output_image_path}")