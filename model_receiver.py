import socket
import numpy as np
import copy

FIXED_YAW = False

speed = 32 #64
rotation_speed = -12
scale_xy = 300    # Il 2 e' perche' il dragonfly deve partire dal centro
scale_z = 10


def go_forward(x, y, z, yaw):
    x += speed * np.cos(yaw * np.pi / 180)
    y += speed * np.sin(yaw * np.pi / 180)
    return x, y, z, yaw


def go_backward(x, y, z, yaw):
    x -= speed * np.cos(yaw * np.pi / 180)
    y -= speed * np.sin(yaw * np.pi / 180)
    return x, y, z, yaw


def turn_right(x, y, z, yaw):
    yaw = (yaw + rotation_speed) % 360
    return x, y, z, yaw


def turn_left(x, y, z, yaw):
    yaw = (yaw - rotation_speed) % 360
    return x, y, z, yaw


def turn_right_go_forward(x, y, z, yaw):
    x, y, z, yaw = turn_right(x, y, z, yaw)
    x, y, z, yaw = go_forward(x, y, z, yaw)
    return x, y, z, yaw


def turn_left_go_forward(x, y, z, yaw):
    x, y, z, yaw = turn_left(x, y, z, yaw)
    x, y, z, yaw = go_forward(x, y, z, yaw)
    return x, y, z, yaw


def turn_right_go_backward(x, y, z, yaw):
    x, y, z, yaw = turn_right(x, y, z, yaw)
    x, y, z, yaw = go_backward(x, y, z, yaw)
    return x, y, z, yaw


def turn_left_go_backward(x, y, z, yaw):
    x, y, z, yaw = turn_left(x, y, z, yaw)
    x, y, z, yaw = go_backward(x, y, z, yaw)
    return x, y, z, yaw


def idle(x, y, z, yaw):
    return x, y, z, yaw


def go_left(x, y, z, yaw):
    x += speed * np.cos((yaw - 90) * np.pi / 180)
    y += speed * np.sin((yaw - 90) * np.pi / 180)
    return x, y, z, yaw


def go_right(x, y, z, yaw):
    x += speed * np.cos((yaw + 90) * np.pi / 180)
    y += speed * np.sin((yaw + 90) * np.pi / 180)
    return x, y, z, yaw


def get_waypoint_from_action(x, y, z, yaw, action):
    if action == 0:
        x, y, z, yaw = go_forward(x, y, z, yaw)
    elif action == 1:
        x, y, z, yaw = turn_left(x, y, z, yaw)
    elif action == 2:
        x, y, z, yaw = turn_right(x, y, z, yaw)
    elif action == 3:
        x, y, z, yaw = turn_right_go_forward(x, y, z, yaw)
    elif action == 4:
        x, y, z, yaw = turn_left_go_forward(x, y, z, yaw)
    elif action == 5:
        x, y, z, yaw = turn_right_go_backward(x, y, z, yaw)
    elif action == 6:
        x, y, z, yaw = turn_left_go_backward(x, y, z, yaw)
    elif action == 7:
        x, y, z, yaw = go_backward(x, y, z, yaw)
    elif action == 8:
        x, y, z, yaw = idle(x, y, z, yaw)
    elif action == 9:
        x, y, z, yaw = go_left(x, y, z, yaw)
    elif action == 10:
        x, y, z, yaw = go_right(x, y, z, yaw)
    else:
        0 / 0
    return x, y, z, yaw


class ModelReceiver:

    def __init__(self, address='localhost', port=19876):
        self.offset_x, self.offset_y, self.offset_z, self.offset_yaw = None, None, None, None
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((address, port))

        server_socket.listen(1)

        self.sock, client_adress = server_socket.accept()

    def send_waypoint(self, x, y, z, yaw, action):
        x, y, z, yaw = get_waypoint_from_action(x, y, z, yaw, action)
        if self.offset_x is None:
            self.offset_x, self.offset_y, self.offset_z, self.offset_yaw = x, y, z, yaw
        yaw_print = yaw if yaw < 180 else yaw - 360
        print('estimate:', x, y, z, yaw_print)
        x = (x - self.offset_x) / scale_xy
        y = (y - self.offset_y) / scale_xy
        z = (z - self.offset_z + 1 * scale_z) / scale_z
        #yaw -= self.offset_yaw
        #yaw -= 90
        message = "{}_{}_{}_{}".format(x, y, z, yaw)
        print('e_rel:', x, y, z, yaw / 180 * np.pi)
        self.sock.sendall(message.encode())
        return yaw

    def recv_waypoint(self, yaw_est):
        message = self.sock.recv(1024).decode()
        waypoint = message.split("_")
        x, y, z, yaw = float(waypoint[0]), float(waypoint[1]), float(waypoint[2]), float(waypoint[3])
        if FIXED_YAW:
            yaw = yaw_est
        print('w_rel:', x, y, z, yaw / 180 * np.pi)
        x = x * scale_xy + self.offset_x
        y = y * scale_xy + self.offset_y
        z = z * scale_z + self.offset_z - 1 * scale_z
        #yaw += self.offset_yaw
        #yaw += 90
        yaw = yaw if yaw < 180 else yaw - 360
        return x, y, z, yaw

    def close(self):
        self.sock.close()
