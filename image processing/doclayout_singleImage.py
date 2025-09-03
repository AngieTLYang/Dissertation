import json
import cv2
import time
from doclayout_yolo import YOLOv10
from huggingface_hub import hf_hub_download
from typing import Union, Dict

# Load model once globally to avoid reloading on every call
MODEL_PATH = hf_hub_download(
    repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
    filename="doclayout_yolo_docstructbench_imgsz1024.pt"
)
MODEL = YOLOv10(MODEL_PATH)

def predict_document(
    image_path: str,
    imgsz: int = 1024,
    conf: float = 0.2,
    device: str = "cpu",
    save_image: bool = True,
    result_image_path: str = "doclayout_result.jpg"
) -> Dict:
    """
    Run document layout detection on a single image.

    Args:
        image_path (str): Path to input image.
        imgsz (int): Prediction image size.
        conf (float): Confidence threshold.
        device (str): Device to run the model ('cpu' or 'cuda:0').
        save_image (bool): Whether to save the annotated image.
        result_image_path (str): Path to save annotated image.

    Returns:
        dict: JSON-like dictionary with detections.
    """

    start_time = time.time()
    det_res = MODEL.predict(
        image_path,
        imgsz=imgsz,
        conf=conf,
        device=device
    )
    end_time = time.time()
    print(f"Prediction time: {end_time - start_time:.2f} seconds")

    # Annotate and save
    if save_image:
        annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
        cv2.imwrite(result_image_path, annotated_frame)

    # Extract results
    result = det_res[0]
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    names = result.names

    # Build JSON output
    detections = []
    for box, score, cls in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box
        detections.append({
            "label": names[int(cls)],
            "confidence": float(score),
            "bbox": [float(x1), float(y1), float(x2), float(y2)]
        })

    return detections
