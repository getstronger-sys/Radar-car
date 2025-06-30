# 地图相关配置
import numpy as np

MAP_SIZE = 5.0  # 地图边长（米）
MAP_RESOLUTION = 0.1  # 分辨率（米/格子）
MAP_SIZE_PIXELS = int(MAP_SIZE / MAP_RESOLUTION) # 计算地图格子数

# 起始点与出口点坐标（单位：米）
START_POSITION = {'x': 0.5, 'y': 0.5, 'theta': 0.0}
EXIT_POSITION = {'x': 4.2, 'y': 4.3, 'theta': 0.0}  # 可以在探索中动态识别

# 机器人参数
"""
把机器人简化成圆形（以ROBOT_RADIUS为半径的圆)
计算上更简单，碰撞检测和障碍物膨胀也更方便，因为圆形旋转对称，不用管朝向变化。
如果把机器人看成长方形，就需要考虑机器人的朝向（θ）
碰撞检测变成“旋转矩形与障碍物”的碰撞判断，计算更复杂。
"""
ROBOT_RADIUS = 0.05  # 包裹机器人外形的最小圆的半径 米
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
    'robot_radius': ROBOT_RADIUS,     # 机器人半径，单位：米。用于计算安全距离，障碍物膨胀时参考此值。
    'map_resolution': MAP_RESOLUTION, # 地图分辨率，单位：米/格子。每个栅格代表的实际尺寸，影响路径规划精度。
    'max_speed': 1.0,                 # 最大线速度，单位：米/秒。机器人允许达到的最高前进速度。
    'min_speed': 0.0,                 # 最小线速度，单位：米/秒。通常为0，表示机器人可以停止。
    'max_yawrate': 1.0,               # 最大角速度，单位：弧度/秒。机器人旋转速度的上限。
    'max_accel': 0.5,                 # 最大线加速度，单位：米/秒²。速度变化的最大速率，限制加减速幅度。
    'max_dyawrate': 1.0,              # 最大角加速度，单位：弧度/秒²。角速度变化的最大速率，限制旋转加减速。
    'dt': 0.1,                       # 时间步长，单位：秒。控制周期和轨迹预测的时间间隔。
    'predict_time': 2.0,             # 轨迹预测总时间，单位：秒。DWA算法在规划时会预测未来2秒的轨迹。
    'v_reso': 0.05,                  # 线速度采样分辨率，单位：米/秒。速度采样的步进大小，影响采样精细度。
    'yawrate_reso': 0.1,             # 角速度采样分辨率，单位：弧度/秒。角速度采样的步进大小。
    'to_goal_cost_gain': 1.0,        # 距离目标点代价的权重系数。数值越大，算法越倾向于朝目标方向移动。
    'speed_cost_gain': 1.0,          # 速度代价权重，鼓励机器人尽量以更高速度运动。
    'obstacle_cost_gain': 1.0,       # 障碍物代价权重，影响避障的重要性。数值越大，机器人越远离障碍物。
}
