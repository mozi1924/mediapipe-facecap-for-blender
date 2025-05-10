import socket
import json

class UDPTransmitter:
    def __init__(self, ip, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.target = (ip, port)
    
    def send(self, data):
        self.sock.sendto(json.dumps(data).encode(), self.target)
    
    def close(self):
        self.sock.close()