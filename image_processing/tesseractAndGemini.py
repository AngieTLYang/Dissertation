import sys
sys.path.append(rf"C:\Users\88690\Desktop\Dissertation\image_processing")
import cv2
import pytesseract
import jiwer
from PIL import Image
import json
from test_callGemini import ask_model 

def evaluate_ocr_text(ground_truth_path, ocr_text):
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        ground_truth = f.read()
    
    print("\nGround truth text:\n", ground_truth)

    word_error = jiwer.wer(ground_truth, ocr_text)
    char_error = jiwer.cer(ground_truth, ocr_text)

    return {
        "wer": word_error,
        "wer_accuracy": 1 - word_error,
        "cer": char_error,
        "cer_accuracy": 1 - char_error
    }
# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = rf"C:\Program Files\Tesseract-OCR\tesseract.exe"

def process_image_and_query(image_path, image_json_path, prompt="give me answer to the question", ground_truth_path = None):
    # Load JSON with filtered bounding boxes
    with open(image_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load original image
    image = cv2.imread(image_path)

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

    print(f"\nThis is text recognized by OCR:\n{combined_text}")

    # Evaluate OCR accuracy
    ocr_results = evaluate_ocr_text(ground_truth_path, combined_text)
    print(f"\nWER: {ocr_results['wer']:.2f}, WER Accuracy: {ocr_results['wer_accuracy']:.2f}")
    print(f"CER: {ocr_results['cer']:.2f}, CER Accuracy: {ocr_results['cer_accuracy']:.2f}")

    # Prepare prompt for GenAI
    full_prompt = f"{prompt}\n\nContext:\n{combined_text}"

    # Call the imported function
    answer = ask_model(full_prompt)
    return answer

