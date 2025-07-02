import time
import numpy as np
import matplotlib.pyplot as plt
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser
from roboviz import MapVisualizer

def generate_square_lidar_scan(pose_x_mm, pose_y_mm, pose_theta_deg, square_size_m=4, scan_size=360):
    """生成正方形场地的模拟激光雷达扫描数据（360度）"""
    scan = []
    theta_rad = np.deg2rad(pose_theta_deg)
    
    # 正方形边界坐标（米）
    half_size = square_size_m / 2
    x_min, x_max = -half_size, half_size
    y_min, y_max = -half_size, half_size
    
    for angle_deg in np.linspace(0, 360, scan_size, endpoint=False): 
        angle_rad = np.deg2rad(angle_deg) + theta_rad
        dx = np.cos(angle_rad)
        dy = np.sin(angle_rad)
        
        # 计算到各边的距离（米）
        t_xmin = (x_min - (pose_x_mm/1000)) / dx if dx != 0 else np.inf
        t_xmax = (x_max - (pose_x_mm/1000)) / dx if dx != 0 else np.inf
        t_ymin = (y_min - (pose_y_mm/1000)) / dy if dy != 0 else np.inf
        t_ymax = (y_max - (pose_y_mm/1000)) / dy if dy != 0 else np.inf
        
        # 取最近的有效交点
        valid_ts = [t for t in [t_xmin, t_xmax, t_ymin, t_ymax] if t > 0]
        if not valid_ts:
            # 无有效交点时返回最大检测距离（避免空数组错误）
            t = square_size_m * np.sqrt(2)
        else:
            t = np.min(valid_ts)
        distance_m = np.clip(t, 0.1, square_size_m*np.sqrt(2))  # 最小检测距离0.1米
        scan.append(int(distance_m * 1000))  # 转换为毫米
    
    return scan

class CarSLAM:
    def __init__(self, map_size_pixels=500, map_size_meters=10, laser_params=None):
        # 初始化激光模型（示例参数，需根据实际激光雷达调整）
        self.laser = Laser(scan_size=180, scan_rate_hz=100, detection_angle_degrees=180,
                          distance_no_detection_mm=10000, detection_margin=0, offset_mm=0)
        # 初始化SLAM对象
        self.slam = RMHC_SLAM(self.laser, map_size_pixels, map_size_meters)
        self.mapbytes = bytearray(map_size_pixels * map_size_pixels)
        # 初始化可视化工具
        self.viz = MapVisualizer(map_size_pixels, map_size_meters, 'Car SLAM Visualization')

    def update_position(self, lidar_scan, pose_change,x,y,theta):
        # 假设lidar_scan是当前激光雷达扫描数据（列表形式）
        # 更新SLAM（传入里程计数据）
        self.slam.update(lidar_scan, pose_change)
        # 获取更精确的位置
        x_mm, y_mm, theta_degrees = x,y,theta
        # 更新地图
        self.slam.getmap(self.mapbytes)
        # 显示地图和当前位置（转换为米）
        self.viz.display(x_mm/1000, y_mm/1000, theta_degrees, self.mapbytes)
        return x_mm, y_mm, theta_degrees


        
        
        # 延迟更新（保持可视化）
        plt.pause(0.5)
