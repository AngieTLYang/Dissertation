import os
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import time

# --- Input & output paths ---
image_path = rf"C:\Users\88690\Desktop\Dissertation\image.png"
output_img_path = rf"C:\Users\88690\Desktop\Dissertation\image_detect.png"
output_label_path = rf"C:\Users\88690\Desktop\Dissertation\image.txt"

# --- Load model and processor ---
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

# --- Define text queries ---
text_labels = [["a pen"]]
label_to_id = {"a pen": 0}  # Define class ID mapping

# --- Open image ---
image = Image.open(image_path).convert("RGB")
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

# Draw & save results
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

# Save output image
image.save(output_img_path)

# Save YOLO annotations
with open(output_label_path, 'w') as f:
    f.write('\n'.join(yolo_lines))

print(f"✅ Processed {image_path}")
print(f"🖼️ Saved detected image -> {output_img_path}")
print(f"📄 Saved YOLO labels -> {output_label_path}")
print(f"⏱️ Time taken: {timestop - timestart:.2f}s")
