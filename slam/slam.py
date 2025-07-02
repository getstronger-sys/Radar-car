import time
import math
import numpy as np
import matplotlib.pyplot as plt
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser
from roboviz import MapVisualizer


class CarSLAM:
    def __init__(self, map_size_pixels=500, map_size_meters=10, laser_params=None):
        # 初始化激光模型（示例参数，需根据实际激光雷达调整）
        self.laser = Laser(scan_size=360, scan_rate_hz=100, detection_angle_degrees=360,
                           distance_no_detection_mm=10000, detection_margin=0, offset_mm=0)
        # 初始化SLAM对象
        self.slam = RMHC_SLAM(self.laser, map_size_pixels, map_size_meters)
        self.mapbytes = bytearray(map_size_pixels * map_size_pixels)
        # 初始化可视化工具
        self.viz = MapVisualizer(map_size_pixels, map_size_meters, 'Car SLAM Visualization')

    def update_position(self, lidar_scan, pose_change, x, y, theta):
        # 假设lidar_scan是当前激光雷达扫描数据（列表形式）
        # 更新SLAM（传入里程计数据）
        self.slam.update(lidar_scan, pose_change)
        # 获取更精确的位置
        x_mm, y_mm, theta_degrees = self.slam.getpos()
        # 更新地图
        self.slam.getmap(self.mapbytes)
        # 显示地图和当前位置（转换为米）
        self.viz.display(x_mm / 1000, y_mm / 1000, theta_degrees, self.mapbytes)
        return x_mm, y_mm, theta_degrees

    def simulate_straight_line(self, start_x_mm=1000, start_y_mm=5000, start_theta_deg=0, distance_mm=8000, step_mm=100):
        """在长方形场地内沿直线移动小车并更新SLAM地图"""
        # 场地尺寸: 20米 x 10米 (20000毫米 x 10000毫米)
        field_length_mm = 20000
        field_width_mm = 10000

        # 确保起点在场地内
        start_x_mm = max(1000, min(start_x_mm, field_length_mm - 1000))
        start_y_mm = max(1000, min(start_y_mm, field_width_mm - 1000))

        x = start_x_mm
        y = start_y_mm
        theta = start_theta_deg
        x = start_x_mm
        y = start_y_mm
        theta = start_theta_deg
        for _ in range(int(distance_mm / step_mm)):
            x += step_mm * np.cos(np.deg2rad(theta))
            y += step_mm * np.sin(np.deg2rad(theta))
            lidar_scan = generate_square_lidar_scan(x, y, theta)
            pose_change = (step_mm, 0, 0)  # 前进step_mm，无侧向移动，无旋转
            self.update_position(lidar_scan, pose_change, x, y, theta)
            time.sleep(0.1)

        # 延迟更新（保持可视化）
        plt.pause(0.5)

    def simulate_curved_path(self, start_x_mm=1000, start_y_mm=5000, start_theta_deg=0, radius_mm=5000, angle_deg=90, step_mm=100):
        """模拟小车沿圆弧路径运动"""
        # 转换角度为弧度
        angle_rad = math.radians(angle_deg)
        start_theta_rad = math.radians(start_theta_deg)
        
        # 计算圆弧长度和步数
        arc_length_mm = radius_mm * angle_rad
        num_steps = max(1, int(arc_length_mm / step_mm))
        
        # 计算圆心位置（左转）
        center_x = start_x_mm + radius_mm * math.cos(start_theta_rad + math.pi/2)
        center_y = start_y_mm + radius_mm * math.sin(start_theta_rad + math.pi/2)
        
        x, y, theta = start_x_mm, start_y_mm, start_theta_deg
        
        for i in range(num_steps):
            # 计算当前角度和位置
            current_angle_rad = start_theta_rad + (i / num_steps) * angle_rad + math.pi/2
            x = center_x - radius_mm * math.cos(current_angle_rad)
            y = center_y - radius_mm * math.sin(current_angle_rad)
            theta = start_theta_deg + (i / num_steps) * angle_deg
            
            # 生成激光扫描数据
            lidar_scan = generate_square_lidar_scan(x, y, theta)
            
            # 计算步长和角度变化
            delta_theta = angle_deg / num_steps
            pose_change = (step_mm, delta_theta, 0)
            
            # 更新SLAM位置
            self.update_position(lidar_scan, pose_change, x, y, theta)
            time.sleep(0.1)
        
        plt.pause(0.5)

def generate_square_lidar_scan(x_mm, y_mm, theta_deg, field_length_mm=20000, field_width_mm=10000, max_range_mm=10000):
    """模拟长方形场地中的激光雷达扫描数据"""
    scan = []
    theta_rad = math.radians(theta_deg)
    half_fov = math.radians(180)  # 假设激光雷达水平视场角为360度
    step = 2 * half_fov / 359  # 360个扫描点

    for i in range(360):
        angle = theta_rad - half_fov + i * step
        dx = math.cos(angle)
        dy = math.sin(angle)

        # 计算到长方形场地边界的距离
        t = float('inf')

        # 左边界 (x=0)
        if dx < 0:
            t_left = (0 - x_mm) / dx
            if t_left > 0:
                y_intersect = y_mm + dy * t_left
                if 0 <= y_intersect <= field_width_mm:
                    t = min(t, t_left)

        # 右边界 (x=field_length_mm)
        if dx > 0:
            t_right = (field_length_mm - x_mm) / dx
            if t_right > 0:
                y_intersect = y_mm + dy * t_right
                if 0 <= y_intersect <= field_width_mm:
                    t = min(t, t_right)

        # 下边界 (y=0)
        if dy < 0:
            t_bottom = (0 - y_mm) / dy
            if t_bottom > 0:
                x_intersect = x_mm + dx * t_bottom
                if 0 <= x_intersect <= field_length_mm:
                    t = min(t, t_bottom)

        # 上边界 (y=field_width_mm)
        if dy > 0:
            t_top = (field_width_mm - y_mm) / dy
            if t_top > 0:
                x_intersect = x_mm + dx * t_top
                if 0 <= x_intersect <= field_length_mm:
                    t = min(t, t_top)

        distance = t if t != float('inf') else max_range_mm
        scan.append(min(distance, max_range_mm))

    return scan


if __name__ == '__main__':
    # 创建SLAM实例
    slam = CarSLAM(map_size_pixels=800, map_size_meters=25)
    # 开始直线探索模拟
    # 组合运动：直线 -> 转弯 -> 直线
    slam.simulate_curved_path(radius_mm=3000, angle_deg=90)
    slam.simulate_straight_line(distance_mm=800)


