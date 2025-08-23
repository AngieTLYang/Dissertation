import socket
import struct
import os

HOST = '0.0.0.0'
PORT = 12345

def recvall(sock, n):
    """Helper function to receive exactly n bytes from socket"""
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()
    print("Listening...")
    conn, addr = s.accept()
    with conn:
        print(f"Connected by {addr}")
        img_count = 0
        while True:
            length_bytes = recvall(conn, 4)
            if not length_bytes:
                print("Connection closed by client")
                break
            length = struct.unpack('!I', length_bytes)[0]

            image_data = recvall(conn, length)
            if not image_data:
                print("Connection closed by client during image data")
                break

            img_count += 1
            script_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(script_dir, f"received_{img_count}.jpg")
            with open(filename, "wb") as f:
                f.write(image_data)
            print(f"Received image {img_count} of {length} bytes")
