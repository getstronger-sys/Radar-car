# comm/bluetooth.py
"""
import serial

class BluetoothComm:
    def __init__(self, port='COM4', baudrate=115200, timeout=1):
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        print(f"✅ Bluetooth connected on {port} @ {baudrate} bps")

    def receive_data(self):
        ""
        从串口接收一帧数据，格式示例：
        L:angle1,dist1;angle2,dist2;...|P:x,y,theta\\n
        ""
        try:
            line = self.ser.readline().decode().strip()
            if not line.startswith("L:") or "|P:" not in line:
                return [], {'x': 0, 'y': 0, 'theta': 0}

            lidar_raw, pose_raw = line.split("|P:")
            lidar_raw = lidar_raw[2:]  # remove "L:"
            scan_points = []
            for pair in lidar_raw.split(";"):
                if not pair:
                    continue
                angle, dist = map(float, pair.split(","))
                x = dist * 0.001 * math.cos(math.radians(angle))
                y = dist * 0.001 * math.sin(math.radians(angle))
                scan_points.append((x, y))

            x, y, theta = map(float, pose_raw.split(","))
            pose = {'x': x, 'y': y, 'theta': theta}
            return scan_points, pose

        except Exception as e:
            print(f"[BluetoothComm] Error receiving data: {e}")
            return [], {'x': 0, 'y': 0, 'theta': 0}

    def send_target(self, pose):
        ""
        发送目标位姿 (x,y,theta) 给机器人，格式：
        T:x,y,theta\\n
        ""
        try:
            command = f"T:{pose['x']:.2f},{pose['y']:.2f},{pose['theta']:.2f}\\n"
            self.ser.write(command.encode())
        except Exception as e:
            print(f"[BluetoothComm] Error sending target: {e}")

    def close(self):
        self.ser.close()
"""

import time
import numpy as np

class BluetoothCommMock:
    def __init__(self):
        self.pose = {'x': 2.5, 'y': 2.5, 'theta': 0.0}  # 起点
        self.path = []
        self.path_index = 0

    def receive_data(self):
        # 模拟每次返回激光数据和当前机器人位姿
        distances = [3000] * 360  # 模拟激光数据，没障碍
        return distances, self.pose

    def send_target(self, target_pose):
        # 收到目标点，存路径点
        if self.path_index >= len(self.path):
            self.path = []
            self.path_index = 0

        self.path.append(target_pose)

    def move_along_path(self):
        # 简单每调用一次，机器人移动一点向目标
        if self.path_index < len(self.path):
            target = self.path[self.path_index]
            # 简单直线一步一步走
            dx = target['x'] - self.pose['x']
            dy = target['y'] - self.pose['y']
            dist = (dx**2 + dy**2)**0.5
            step = 0.1  # 每步走0.1m

            if dist < step:
                self.pose = target
                self.path_index += 1
            else:
                self.pose['x'] += step * dx / dist
                self.pose['y'] += step * dy / dist

            # 更新朝向
            self.pose['theta'] = np.arctan2(dy, dx)

    def close(self):
        pass
