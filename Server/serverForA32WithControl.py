import socket
import threading

clients = []

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
            # Read image
            data = b""
            while len(data) < length:
                packet = client_socket.recv(length - len(data))
                if not packet:
                    break
                data += packet
            # Save image
            filename = f"received_{addr[1]}.jpg"
            with open(filename, "wb") as f:
                f.write(data)
            print(f"Saved image ({len(data)} bytes) to {filename}")
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

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(("0.0.0.0", 12345))
server.listen(5)
print("Server running on port 12345...")

def accept_clients():
    while True:
        client_socket, addr = server.accept()
        threading.Thread(target=handle_client, args=(client_socket, addr), daemon=True).start()

threading.Thread(target=accept_clients, daemon=True).start()

while True:
    cmd = input("Enter command (PAUSE/RESUME/EXIT): ").strip().upper()
    if cmd == "EXIT":
        break
    elif cmd in ("PAUSE", "RESUME"):
        send_command_to_clients(cmd)
