import numpy as np
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import RPLidarA1  # 替换为实际传感器模型
from config.settings import MAP_SIZE_PIXELS, MAP_RESOLUTION, LIDAR_SCAN_SIZE, MAP_SIZE

class SLAMProcessor:
    def __init__(self):
        """
        初始化 SLAMProcessor，包括激光模型和 SLAM 实例，预分配地图字节数组。
        """

        # 使用实际雷达模型，自动处理角度和数据格式（匹配 RPLidar C1/A1）
        self.laser_model = RPLidarA1()  # ✅ 更推荐用这个替代 LaserModel

        # 初始化 SLAM 算法：RMHC（Real-time Monte Carlo）方法
        self.slam = RMHC_SLAM(
            self.laser_model,
            MAP_SIZE_PIXELS,  # 地图大小（像素）
            int(MAP_SIZE * 1000)  # 地图尺寸（毫米）
        )

        self.mapbytes = bytearray(MAP_SIZE_PIXELS * MAP_SIZE_PIXELS)  # 地图数据缓冲区
        self.pose = (0, 0, 0)  # 初始位姿 (x_mm, y_mm, theta_deg)
        self.prev_odom = None  # 上一帧里程计位姿（单位米）

    def update(self, scan, odom):
        """
        使用激光与里程计数据更新 SLAM 状态。

        参数:
        - scan: list[int]，360 个激光距离点（单位 mm）
        - odom: tuple(x, y, theta)，机器人当前估计位姿（单位米 + 弧度）

        返回:
        - pose: (x_mm, y_mm, theta_deg)
        - mapbytes: bytearray 类型地图数据
        """

        # === Step 1: 计算 odom 增量 ===
        if self.prev_odom is None:
            odo_delta = None  # 第一次更新无法计算增量
        else:
            dx = (odom[0] - self.prev_odom[0]) * 1000  # 米 → 毫米
            dy = (odom[1] - self.prev_odom[1]) * 1000
            dtheta = np.degrees(odom[2] - self.prev_odom[2])  # 弧度 → 角度
            odo_delta = (dx, dy, dtheta)

        self.prev_odom = odom  # 记录上一次里程计

        # === Step 2: SLAM 更新 ===
        if odo_delta:
            self.slam.update(scan, odo_delta)
        else:
            self.slam.update(scan)

        # === Step 3: 获取地图与估计位姿 ===
        self.pose = self.slam.getpos()          # 返回(x_mm, y_mm, theta_deg)
        self.slam.getmap(self.mapbytes)         # 写入地图缓冲区

        return self.pose, self.mapbytes

    def get_occupancy_grid(self, threshold=200):
        """
        将 SLAM 地图字节数据转换为二值栅格地图（0自由, 1障碍）

        参数:
            threshold: 阈值（默认200）大于该值视为障碍物

        返回:
            2D numpy 数组，0 表示自由，1 表示障碍
        """
        map_array = np.array(self.mapbytes, dtype=np.uint8).reshape((MAP_SIZE_PIXELS, MAP_SIZE_PIXELS))
        occupancy_grid = (map_array >= threshold).astype(np.uint8)
        return occupancy_grid

    def world_to_map(self, x, y):
        """
        将世界坐标 (米) 转换为地图像素坐标 (格子索引)
        ✅ 修复：添加中心偏移和边界检查
        """
        # 计算相对于地图中心的偏移
        center_offset = MAP_SIZE / 2
        gx = int((x + center_offset) / MAP_RESOLUTION)
        gy = int((y + center_offset) / MAP_RESOLUTION)
        
        # 边界检查
        gx = max(0, min(gx, MAP_SIZE_PIXELS - 1))
        gy = max(0, min(gy, MAP_SIZE_PIXELS - 1))
        
        return gx, gy

    def map_to_world(self, gx, gy):
        """
        将地图格子坐标转换为世界坐标 (米)
        ✅ 修复：添加中心偏移和边界检查
        """
        # 边界检查
        gx = max(0, min(gx, MAP_SIZE_PIXELS - 1))
        gy = max(0, min(gy, MAP_SIZE_PIXELS - 1))
        
        # 计算世界坐标（格子中心）
        center_offset = MAP_SIZE / 2
        x = gx * MAP_RESOLUTION - center_offset + MAP_RESOLUTION / 2
        y = gy * MAP_RESOLUTION - center_offset + MAP_RESOLUTION / 2
        
        return x, y

    def get_current_pose_meters(self):
        """
        获取当前位姿（单位：米）
        返回: (x_m, y_m, theta_rad)
        """
        x_mm, y_mm, theta_deg = self.pose
        x_m = x_mm / 1000.0
        y_m = y_mm / 1000.0
        theta_rad = np.radians(theta_deg)
        return (x_m, y_m, theta_rad)
