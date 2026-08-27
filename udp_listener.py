import socket
import struct

from aim_controller import AimController

class UDPListener:
    def __init__(self, ip:str, port:int):
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.ip, self.port))

    def state_update(self, aim_controller:AimController):
        while True:
            data, addr = self.sock.recvfrom(1024)

            if len(data) == 5:
                aim_controller.can_attack = data[0]
                aim_controller.weapon_cooldown = struct.unpack(">f", data[1:5])[0]

