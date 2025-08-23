import cv2
import pytesseract
import json
from test_callGemini import ask_model 

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = rf"C:\Program Files\Tesseract-OCR\tesseract.exe"

def process_image_and_query(image_json_path, prompt="give me answer to the question"):
    # Load JSON with filtered bounding boxes
    with open(image_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load original image
    image = cv2.imread(data["image"])

    # Sort bounding boxes by top y-coordinate
    sorted_boxes = sorted(data["filtered_doclayout"], key=lambda d: d["bbox"][1])

    extracted_texts = []

    for i, det in enumerate(sorted_boxes):
        x1, y1, x2, y2 = map(int, det["bbox"])
        roi = image[y1:y2, x1:x2]

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        text = pytesseract.image_to_string(roi_rgb, lang="eng").strip()
        if text:
            extracted_texts.append(text)

    # Combine all OCR text
    combined_text = "\n".join(extracted_texts)

    # Prepare prompt for GenAI
    full_prompt = f"{prompt}\n\nContext:\n{combined_text}"

    # Call the imported function
    answer = ask_model(full_prompt)
    return answer


# --- Example usage ---
answer = process_image_and_query("filtered_text_between_pens.json")
print("=== LLM Answer ===")
print(answer)
