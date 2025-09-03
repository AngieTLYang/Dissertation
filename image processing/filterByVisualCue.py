import os
import json
import cv2
import math
from PIL import Image

def process_doclayout_with_pens(img_path, owl_txt, doclayout_detections,
                                output_json="filtered_text_between_pens.json",
                                output_img="filtered_text_between_pens.jpg"):
    """
    Process DocLayout detections and OWLv2 pen detections,
    filter text between pens, and save outputs.

    Args:
        img_path (str): Path to the original image.
        owl_txt (str): Path to OWLv2 label file (.txt, YOLO format).
        doclayout_json (str): Path to DocLayout output JSON.
        output_json (str): Path to save filtered JSON.
        output_img (str): Path to save annotated image.

    Returns:
        dict: Filtered JSON content (same as saved file).
    """

    # --- Load original image ---
    image = Image.open(img_path)
    IMG_W, IMG_H = image.size
    img_cv = cv2.imread(img_path)

    # --- Load OWLv2 boxes ---
    def load_owl_boxes(txt_path, img_width, img_height):
        boxes = []
        with open(txt_path, "r") as f:
            for line in f:
                cls, x_c, y_c, w, h = map(float, line.strip().split())
                x1 = (x_c - w/2) * img_width
                y1 = (y_c - h/2) * img_height
                x2 = (x_c + w/2) * img_width
                y2 = (y_c + h/2) * img_height
                boxes.append([x1, y1, x2, y2])
        return boxes

    pen_boxes = load_owl_boxes(owl_txt, IMG_W, IMG_H)
    if len(pen_boxes) != 2:
        raise ValueError("Expected exactly 2 OWLv2 detections (two pens).")

    # --- Determine left/right pen ---
    if pen_boxes[0][0] < pen_boxes[1][0]:
        left_pen, right_pen = pen_boxes[0], pen_boxes[1]
    else:
        left_pen, right_pen = pen_boxes[1], pen_boxes[0]

    # --- Pen tips ---
    pen1_tip = (left_pen[2], (left_pen[1]+left_pen[3])/2)
    pen2_tip = (right_pen[0], (right_pen[1]+right_pen[3])/2)

    # --- Pen-tip rectangle ---
    TOL = 5
    x_left, x_right = min(pen1_tip[0], pen2_tip[0]), max(pen1_tip[0], pen2_tip[0])
    y_top, y_bottom = min(pen1_tip[1], pen2_tip[1]), max(pen1_tip[1], pen2_tip[1])

    # --- Distance helper ---
    LINE_TOL = 5
    def distance_point_to_line(px, py, x1, y1, x2, y2):
        A = px - x1
        B = py - y1
        C = x2 - x1
        D = y2 - y1
        dot = A * C + B * D
        len_sq = C * C + D * D
        param = dot / len_sq if len_sq != 0 else -1
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D
        dx = px - xx
        dy = py - yy
        return math.sqrt(dx*dx + dy*dy)

    detections = doclayout_detections

    # --- Filter detections ---
    filtered = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        rect_overlap = (x1 <= x_right + TOL and x2 >= x_left - TOL and
                        y1 <= y_bottom + TOL and y2 >= y_top - TOL)
        box_points = [(x1, y1), (x1, y2), (x2, y1), (x2, y2)]
        near_line = any(distance_point_to_line(px, py, *pen1_tip, *pen2_tip) <= LINE_TOL
                        for (px, py) in box_points)
        if rect_overlap or near_line:
            filtered.append(det)

    # --- Save JSON ---
    output = {
        "pens": pen_boxes,
        "pen_tip_line": [pen1_tip, pen2_tip],
        "pen_tip_rectangle": [x_left, y_top, x_right, y_bottom],
        "filtered_doclayout": filtered,
        "all_doclayout": detections
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # --- Visualization ---
    for (x1, y1, x2, y2) in pen_boxes:
        cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
    cv2.rectangle(img_cv, (int(x_left), int(y_top)), (int(x_right), int(y_bottom)), (255,0,0), 2)
    cv2.line(img_cv, (int(pen1_tip[0]), int(pen1_tip[1])),
                     (int(pen2_tip[0]), int(pen2_tip[1])),
                     (0,255,255), 2)
    for det in filtered:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img_cv, det["label"], (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    cv2.imwrite(output_img, img_cv)

    return output
