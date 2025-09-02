import socket
import struct

HOST = '0.0.0.0'  # listen on all interfaces
PORT = 12345

def receive_image(conn, addr):
    try:
        # First 4 bytes = length of incoming data
        length_bytes = conn.recv(4)
        if len(length_bytes) < 4:
            print(f"Incomplete length prefix from {addr}")
            return

        length = struct.unpack('!I', length_bytes)[0]
        print(f"Expecting {length} bytes from {addr}")

        # Receive the image data
        data = b''
        while len(data) < length:
            packet = conn.recv(length - len(data))
            if not packet:
                break
            data += packet

        if len(data) == length:
            filename = f"received_{addr[1]}.jpg"
            with open(filename, 'wb') as f:
                f.write(data)
            print(f"Saved {len(data)} bytes to {filename}")
        else:
            print(f"Received {len(data)}/{length} bytes from {addr}")

    except Exception as e:
        print(f"Error receiving data from {addr}: {e}")

def main():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((HOST, PORT))
        s.listen()
        print(f"Server listening on {HOST}:{PORT}")
        while True:
            conn, addr = s.accept()
            print(f"Connection from {addr}")
            with conn:
                receive_image(conn, addr)

if __name__ == "__main__":
    main()
