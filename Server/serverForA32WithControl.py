import socket
import threading
import os
import sys
import shutil
from datetime import datetime

# Add path to your OWL script
sys.path.append(rf"C:\Users\88690\Desktop\Dissertation\image_processing")
import owlv2_singleImage as owl  # assuming you wrapped OWL detection as a function
import doclayout_singleImage as dlo
import filterByVisualCue as fvc
import tesseractAndGemini as tg
# import yolo_singleImage as yolo
# Folder to save incoming images
save_folder = os.path.join(os.path.dirname(__file__), "imageFromPhone")
os.makedirs(save_folder, exist_ok=True)

# Globals
image_clients = []     # sockets for image streaming
control_clients = []   # sockets for control + text
latest_image_path = os.path.join(save_folder, "latest.jpg")
processing_image_path = os.path.join(save_folder, "processing.jpg")
#processing_lock = threading.Lock()
new_image_event = threading.Event()
pause_event = threading.Event()
pause_event.set()  # allow processing initially


# -------------------------
# Image handling
# -------------------------
def handle_image_client(client_socket, addr):
    print(f"[ImageClient] Connected: {addr}")
    image_clients.append(client_socket)
    try:
        while True:
            # Read length of incoming image
            length_bytes = client_socket.recv(4)
            if not length_bytes:
                break
            length = int.from_bytes(length_bytes, "big")

            # Read image data
            data = b""
            while len(data) < length:
                packet = client_socket.recv(length - len(data))
                if not packet:
                    break
                data += packet

            # Save latest image
            with open(latest_image_path, "wb") as f:
                f.write(data)
            print(f"[ImageClient] Received image ({len(data)} bytes) from {addr}, saved as latest.jpg")

            # Notify processing thread
            new_image_event.set()
    except Exception as e:
        print(f"[ImageClient] {addr} disconnected: {e}")
    finally:
        if client_socket in image_clients:
            image_clients.remove(client_socket)
        client_socket.close()


# -------------------------
# Control/Text handling
# -------------------------
def handle_control_client(client_socket, addr):
    print(f"[ControlClient] Connected: {addr}")
    control_clients.append(client_socket)
    try:
        while True:
            cmd = client_socket.recv(1024).decode().strip()
            if not cmd:
                break
            print(f"[ControlClient] Received command from {addr}: {cmd}")

            if cmd == "RESUME":
                pause_event.set()
    except Exception as e:
        print(f"[ControlClient] {addr} disconnected: {e}")


def send_to_control_clients(msg: str):
    for client in control_clients:
        try:
            client.send((msg + "\n").encode())
        except:
            pass


# -------------------------
# Image Processing Loop
# -------------------------
def process_images_loop():
    while True:
        new_image_event.wait()
        new_image_event.clear()

        # Wait here if paused
        pause_event.wait()

        # Copy latest image to processing file
        # with processing_lock: 
        shutil.copy2(latest_image_path, processing_image_path)

        try:
            print("Processing image with OWL...")
            pen_count = owl.detect_pens(processing_image_path, save_img_path = "owl_result.jpg", save_label_path = "owl_result.txt")
            print(f"Detected {pen_count} pens in the image.")

            if pen_count == 2:
                print("Exactly 2 pens detected, sending PAUSE command.")
                send_to_control_clients("PAUSE")
                pause_event.clear()  # pause loop until client sends RESUME
                resultDoclayout = dlo.predict_document(processing_image_path)
                output = fvc.process_doclayout_with_pens(img_path = processing_image_path, owl_txt = "owl_result.txt", doclayout_detections = resultDoclayout)
                answer = tg.process_image_and_query(image_path = processing_image_path, image_json_path = "filtered_text_between_pens.json")
                print(answer)
                send_to_control_clients(answer)

        except Exception as e:
            print(f"Error during OWL processing: {e}")

        # Check if a new image arrived during processing
        if os.path.exists(latest_image_path):
            if os.path.getmtime(latest_image_path) > os.path.getmtime(processing_image_path):
                print("New image received during processing, continuing loop...")
                new_image_event.set()

#'''
if __name__ == "__main__":
    processing_image_path = rf"C:\Users\88690\Desktop\Dissertation\imagesFromPhone\ground_truth_q02.jpg"
    ground_truth_path = rf"C:\Users\88690\Desktop\Dissertation\imagesFromPhone\ground_truth_q02.txt"
    print(f"Processing image with OWL... at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
    pen_count = owl.detect_pens(processing_image_path, save_img_path = "owl_result.jpg", save_label_path = "owl_result.txt")
    #pen_count = yolo.detect_pens(processing_image_path, save_img_path="yolo_result.jpg", save_label_path="yolo_result.txt")
    print(f"Detected {pen_count} pens in the image at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}.")

    if pen_count == 2:
        print("Exactly 2 pens detected, sending PAUSE command.")
        send_to_control_clients("PAUSE")
        pause_event.clear()  # pause loop until client sends RESUME
        print(f"\nProcessing image with Doclayout... at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        resultDoclayout = dlo.predict_document(processing_image_path)
        print(f"\nProcessing image with filterByVisualCue... at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        output = fvc.process_doclayout_with_pens(img_path = processing_image_path, owl_txt = "owl_result.txt", doclayout_detections = resultDoclayout)
        print(f"\nProcessing image with OCR and LLM... at {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        answer = tg.process_image_and_query(image_path = processing_image_path, image_json_path = "filtered_text_between_pens.json", ground_truth_path=ground_truth_path)
        print(f"\nThis answer is returned from Gemini: {answer}")
        send_to_control_clients(answer)
#'''

# -------------------------
# Server setup
# -------------------------
def start_image_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", 12345))
    server.listen(5)
    print("Image server running on port 12345...")
    while True:
        client_socket, addr = server.accept()
        threading.Thread(target=handle_image_client, args=(client_socket, addr), daemon=True).start()


def start_control_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("0.0.0.0", 12346))
    server.listen(5)
    print("Control server running on port 12346...")
    while True:
        client_socket, addr = server.accept()
        threading.Thread(target=handle_control_client, args=(client_socket, addr), daemon=True).start()


# -------------------------
# Start everything
# -------------------------
threading.Thread(target=start_image_server, daemon=True).start()
threading.Thread(target=start_control_server, daemon=True).start()
threading.Thread(target=process_images_loop, daemon=True).start()

# Optional manual control
while True:
    cmd = input("Enter command (PAUSE/RESUME/EXIT/TEXT <msg>): ").strip()
    if cmd.upper() == "EXIT":
        break
    elif cmd.upper() == "RESUME":
        pause_event.set()
        send_to_control_clients("RESUME")
    elif cmd.upper() == "PAUSE":
        pause_event.clear()
        send_to_control_clients("PAUSE")
    elif cmd.startswith("TEXT "):
        text = cmd[5:]
        send_to_control_clients(f"TEXT:{text}")


