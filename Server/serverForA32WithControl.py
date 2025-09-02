import socket
import threading
import os
import sys
import shutil
import time

# Add path to your OWL script
sys.path.append(rf"C:\Users\88690\Desktop\Dissertation\image processing")
import test_owlv2_singleImage as owl  # assuming you wrapped OWL detection as a function

# Folder to save incoming images
save_folder = os.path.join(os.path.dirname(__file__), "imageFromPhone")
os.makedirs(save_folder, exist_ok=True)

clients = []

latest_image_path = os.path.join(save_folder, "latest.jpg")
processing_image_path = os.path.join(save_folder, "processing.jpg")
processing_lock = threading.Lock()
new_image_event = threading.Event()


def handle_client(client_socket, addr):
    print(f"Client connected: {addr}")
    clients.append(client_socket)
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
            print(f"Received image ({len(data)} bytes) from {addr}, saved as latest.jpg")
            # Notify processing thread
            new_image_event.set()
    except Exception as e:
        print(f"Client {addr} disconnected: {e}")
    finally:
        clients.remove(client_socket)
        client_socket.close()


def send_command_to_clients(cmd: str):
    for client in clients:
        try:
            client.send(cmd.encode())
        except:
            pass


def process_images_loop():
    while True:
        # Wait until a new image is available
        new_image_event.wait()
        new_image_event.clear()

        # Copy latest image to processing file (so incoming images won't overwrite it)
        with processing_lock:
            shutil.copy2(latest_image_path, processing_image_path)

        # Call your OWL detection function
        try:
            print("Processing image with OWL...")
            # Assuming you wrapped your OWL detection as:
            # result = owl.detect_objects(image_path)
            # where result returns list of detected objects or counts
            result = owl.detect_pens(processing_image_path)

            # Check for at least 2 pens
            pen_count = result[0]
            print(f"Detected {pen_count} pens in the image.")

            if pen_count >= 2:
                print("✅ At least 2 pens detected, sending PAUSE command.")
                send_command_to_clients("PAUSE")
            else:
                print("⏩ Less than 2 pens, sending RESUME command.")
                send_command_to_clients("RESUME")

        except Exception as e:
            print(f"Error during OWL processing: {e}")

        # Check if new image came in during processing
        if os.path.exists(latest_image_path):
            if os.path.getmtime(latest_image_path) > os.path.getmtime(processing_image_path):
                print("New image received during processing, continuing loop...")
                new_image_event.set()


# -------------------------
# Server setup
# -------------------------
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 12345))
server.listen(5)
print("Server running on port 12345...")


def accept_clients():
    while True:
        client_socket, addr = server.accept()
        threading.Thread(target=handle_client, args=(client_socket, addr), daemon=True).start()


threading.Thread(target=accept_clients, daemon=True).start()
threading.Thread(target=process_images_loop, daemon=True).start()

# Optional command loop for manual commands
while True:
    cmd = input("Enter command (PAUSE/RESUME/EXIT): ").strip().upper()
    if cmd == "EXIT":
        break
    send_command_to_clients(cmd)


