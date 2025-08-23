import json
import cv2
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
# Load the pre-trained model
# model = YOLOv10("path/to/provided/model")
# Load the pre-trained model from Hugging Face Hub (or wherever supported)
# model = YOLOv10.from_pretrained("juliozhao/DocLayout-YOLO-DocStructBench")
filepath = hf_hub_download(repo_id="juliozhao/DocLayout-YOLO-DocStructBench", filename="doclayout_yolo_docstructbench_imgsz1024.pt")
model = YOLOv10(filepath)

# Perform prediction
image_path = rf"C:\Users\88690\Desktop\Dissertation\image processing\imagesFromPhone\photo_2025-08-22_18-57-35.jpg"
det_res = model.predict(
    image_path,
    imgsz=1024,      # Prediction image size
    conf=0.2,        # Confidence threshold
    device="cpu"     # Device to use (e.g., 'cuda:0' or 'cpu')
)

# Annotate and save the result
annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)

cv2.imwrite("result.jpg", annotated_frame)

# Extract results
result = det_res[0]
boxes = result.boxes.xyxy.cpu().numpy()     # bounding boxes
scores = result.boxes.conf.cpu().numpy()   # confidence scores
class_ids = result.boxes.cls.cpu().numpy() # class ids
names = result.names                       # id->label mapping

# Prepare JSON output
detections = []
for box, score, cls in zip(boxes, scores, class_ids):
    x1, y1, x2, y2 = box
    detections.append({
        "label": names[int(cls)],
        "confidence": float(score),
        "bbox": [float(x1), float(y1), float(x2), float(y2)]
    })

output = {
    "image": image_path,
    "model": "DocLayout-YOLO-DocStructBench",
    "detections": detections
}

# Save JSON
with open("detections.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print("✅ Results saved to detections.json")