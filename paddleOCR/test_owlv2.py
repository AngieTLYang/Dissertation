import os
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import time

# --- Setup paths ---
frames_dir = rf"C:\Users\88690\Desktop\Dissertation\paddleOCR\frames5"
output_img_dir = rf"C:\Users\88690\Desktop\Dissertation\paddleOCR\frame5_detect"
output_label_dir = rf"C:\Users\88690\Desktop\Dissertation\paddleOCR\frames5\labels"
os.makedirs(output_img_dir, exist_ok=True)
os.makedirs(output_label_dir, exist_ok=True)

# --- Load model and processor ---
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

# --- Define text queries ---
text_labels = [["a pen"]]
label_to_id = {"a pen": 0}  # Define class ID mapping

# --- Loop over each frame ---
frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.png'))])

for idx, frame_file in enumerate(frame_files[132:], start=133):
    frame_path = os.path.join(frames_dir, frame_file)
    image = Image.open(frame_path).convert("RGB")
    image_width, image_height = image.size

    timestart = time.time()

    # Process image
    inputs = processor(text=text_labels, images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Convert to readable results
    target_sizes = torch.tensor([(image.height, image.width)])
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=0.1,
        text_labels=text_labels,
    )

    timestop = time.time()

    result = results[0]
    boxes, scores, predicted_labels = result["boxes"], result["scores"], result["text_labels"]
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    yolo_lines = []

    for box, score, label in zip(boxes, scores, predicted_labels):
        box = [round(i, 2) for i in box.tolist()]
        x_min, y_min, x_max, y_max = box
        x_center = (x_min + x_max) / 2 / image_width
        y_center = (y_min + y_max) / 2 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        class_id = label_to_id.get(label, 0)
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        draw.rectangle(box, outline="red", width=2)
        draw.text((x_min, y_min), f"{label} ({round(score.item(), 2)})", fill="red", font=font)

    # Save image
    output_img_path = os.path.join(output_img_dir, frame_file)
    image.save(output_img_path)

    # Save annotation
    txt_filename = os.path.splitext(frame_file)[0] + ".txt"
    label_txt_path = os.path.join(output_label_dir, txt_filename)
    with open(label_txt_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

    print(f"[{idx+1}/{len(frame_files)}] Saved image + labels for: {frame_file} ({timestop - timestart:.2f}s)")

print("\n All frames processed and labeled in YOLO format.")
