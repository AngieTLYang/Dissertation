import os
from PIL import Image, ImageDraw, ImageFont
import torch
from transformers import Owlv2Processor, Owlv2ForObjectDetection

# Load model and processor once (keep in memory)
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")

# Define text labels
text_labels = [["a pen"]]
label_to_id = {"a pen": 0}  # map class name to ID

def detect_pens(image_path: str, save_img_path=None, save_label_path=None, threshold=0.3):
    """
    Run Owlv2 object detection on the image.

    Returns:
        pen_count (int): number of pens detected above threshold
        result_image (PIL.Image): optional image with bounding boxes drawn
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    inputs = processor(text=text_labels, images=image, return_tensors="pt")
    outputs = model(**inputs)

    target_sizes = torch.tensor([(image.height, image.width)])
    results = processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=threshold,
        text_labels=text_labels,
    )

    result = results[0]
    boxes, scores, predicted_labels = result["boxes"], result["scores"], result["text_labels"]

    # Draw image if needed
    if save_img_path is not None or save_label_path is not None:
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        yolo_lines = []

        for box, score, label in zip(boxes, scores, predicted_labels):
            box = [round(i, 2) for i in box.tolist()]
            x_min, y_min, x_max, y_max = box
            x_center = (x_min + x_max) / 2 / width
            y_center = (y_min + y_max) / 2 / height
            w = (x_max - x_min) / width
            h = (y_max - y_min) / height

            class_id = label_to_id.get(label, 0)
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            draw.rectangle(box, outline="red", width=2)
            draw.text((x_min, y_min), f"{label} ({round(score.item(),2)})", fill="red", font=font)

        if save_img_path:
            image.save(save_img_path)
        if save_label_path:
            with open(save_label_path, "w") as f:
                f.write("\n".join(yolo_lines))

    pen_count = sum([1 for label in predicted_labels if label == "a pen"])
    return pen_count
