import socket

UDP_IP = "0.0.0.0"          # 本机任意 IP 接收
UDP_PORT = 1511             # NatNet 的数据端口

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"🔍 Listening for UDP packets on port {UDP_PORT}...")

while True:
    data, addr = sock.recvfrom(4096)  # 接收最多 4KB 数据
    print(f"✅ Received {len(data)} bytes from {addr}")
