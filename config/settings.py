# 地图相关配置
import numpy as np

MAP_SIZE_M = 15.0  # 地图物理尺寸（米）
MAP_RESOLUTION = 0.3  # 分辨率（米/格子）
MAP_SIZE = int(MAP_SIZE_M / MAP_RESOLUTION)  # 地图格子数

# 起始点与出口点坐标（单位：米）
START_POSITION = {'x': 3.0, 'y': 0.0, 'theta': 0.0}
EXIT_POSITION = {'x': 11.7000, 'y': 14.7000, 'theta': 0.0}  # 自动更新终点

# 机器人参数
"""
把机器人简化成圆形（以ROBOT_RADIUS为半径的圆)
计算上更简单，碰撞检测和障碍物膨胀也更方便，因为圆形旋转对称，不用管朝向变化。
如果把机器人看成长方形，就需要考虑机器人的朝向（θ）
碰撞检测变成"旋转矩形与障碍物"的碰撞判断，计算更复杂。
"""
ROBOT_RADIUS = 0.005  # 包裹机器人外形的最小圆的半径 米
ROBOT_LENGTH = 0.3  # 从前端到后端的物理距离 米

# 控制与导航
USE_DWA = True  # 是否启用 DWA 局部避障
LOOP_DELAY = 0.1  # 主循环延时（秒）

# LiDAR 参数
LIDAR_SCAN_SIZE = 360
LIDAR_MAX_DISTANCE = 4000  # 毫米

# 蓝牙通信
BLUETOOTH_PORT = 'COM4'
BLUETOOTH_BAUDRATE = 115200

DWA_CONFIG = {
    'robot_radius': ROBOT_RADIUS,
    'map_resolution': MAP_RESOLUTION,
    'max_speed': 1.2,                # 提高最大速度，让轨迹评估更积极
    'min_speed': 0.0,
    'max_yawrate': 2.0,              # ✅ 提高转弯速度上限，更容易脱困
    'max_accel': 0.6,
    'max_dyawrate': 2.0,             # ✅ 同上，提升角加速度上限
    'dt': 0.1,
    'predict_time': 2.0,
    'v_reso': 0.08,                  # 稍微变粗，不影响效果
    'yawrate_reso': 0.035,          # ✅ 角速度步进调小：~2°，增加轨迹尝试多样性
    'to_goal_cost_gain': 12.0,      # ✅ 增大朝目标吸引力
    'speed_cost_gain': 0.5,         # ⚖️ 保持中等速度权重，不鼓励龟速
    'obstacle_cost_gain': 0.3,      # ✅ 降低避障畏惧感，使机器人敢于接近狭道
}
