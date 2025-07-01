"""
此模块实现了 SLAMWrapper 类，用于集成 BreezySLAM 库，实现基于激光雷达数据和位姿变化的 SLAM 功能。
提供地图更新、获取当前位姿和地图可视化等功能。
"""
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser

class SLAMWrapper:
    """
    封装 BreezySLAM 的 RMHC_SLAM 类，提供简单的接口用于更新 SLAM 状态、获取地图和当前位姿，以及可视化地图。

    属性:
        map_size_pixels (int): 地图的像素大小。
        laser (Laser): 激光雷达对象。
        slam (RMHC_SLAM): RMHC_SLAM 实例，用于执行 SLAM 算法。
    """
    def __init__(self, laser_params, map_size_pixels=500, map_size_meters=10):
        """
        初始化 SLAMWrapper 实例。

        参数:
            laser_params (dict): 激光雷达的参数，用于初始化 Laser 对象。
            map_size_pixels (int, 可选): 地图的像素大小，默认为 500。
            map_size_meters (int, 可选): 地图的实际大小（米），默认为 10。
        """
        self.map_size_pixels = map_size_pixels  # 保存地图尺寸
        self.laser = Laser(**laser_params)
        self.slam = RMHC_SLAM(self.laser, map_size_pixels, map_size_meters)
        
    def get_map(self):
        """
        获取当前的地图数据。

        返回:
            bytearray: 当前地图的字节数组，大小为 map_size_pixels * map_size_pixels。
        """
        mapbytes = bytearray(self.map_size_pixels * self.map_size_pixels)
        self.slam.getmap(mapbytes)
        return mapbytes
        
    def update(self, scan_distances, pose_change):
        """
        根据激光雷达扫描数据和位姿变化更新 SLAM 状态。

        参数:
            scan_distances (list): 激光雷达扫描距离列表，单位为毫米。
            pose_change (tuple): 位姿变化元组，格式为 (dx_mm, dy_mm, dtheta_degrees, dt_seconds)。
        """
        dx_mm, dy_mm, dtheta_degrees, dt_seconds = pose_change
        self.slam.update(scan_distances, (dx_mm, dy_mm, dtheta_degrees, dt_seconds))
        
    def get_position(self):
        """
        获取当前机器人的位姿。

        返回:
            tuple: 当前位姿，格式为 (x_mm, y_mm, theta_degrees)，单位分别为毫米和度。
        """
        return self.slam.getpos()
        
    def print_progress(self, iteration):
        """
        打印当前 SLAM 处理的迭代次数。

        参数:
            iteration (int): 当前的迭代次数。
        """
        print(f"SLAM处理中... 迭代次数: {iteration}")
        
    def visualize_map(self, map_data):
        """
        可视化地图数据，使用不同字符表示不同的障碍物密度。
        打印一个 20x20 的地图概览。

        参数:
            map_data (bytearray): 地图的字节数组数据。
        """
        print("当前地图概览(20x20):")
        # 打印20x20的地图，使用不同字符表示不同障碍物密度
        for i in range(0, min(20, self.map_size_pixels)):
            row = []
            for j in range(0, min(20, self.map_size_pixels)):
                byte = map_data[i*self.map_size_pixels + j]
                if byte > 220:
                    row.append('#')  # 高密度障碍物
                elif byte > 150:
                    row.append('*')  # 中等密度障碍物
                elif byte > 30:
                    row.append('.')  # 低密度障碍物
                else:
                    row.append(' ')
            print(''.join(row))


"""

下面是示例用法:
"""
    # 初始化激光雷达参数
laser_params = {
    'scan_size': 360,
    'scan_rate_hz': 10,
    'detection_angle_degrees': 240,
    'distance_no_detection_mm': 4000
}

# 创建SLAM实例
slam = SLAMWrapper(laser_params)

# 模拟数据更新循环
iteration = 0
while True:
    iteration += 1
    # 获取激光雷达数据(模拟)
    scan_distances = [1000] * 360  # 单位mm

    
    # 获取IMU位姿变化(模拟)
    pose_change = (10, 5, 1, 0.1)  # (dx_mm, dy_mm, dtheta_degrees, dt_seconds)
    
    # 更新SLAM
    slam.update(scan_distances, pose_change)
    
    # 获取当前位姿
    x, y, theta = slam.get_position()
    
    # 获取地图数据
    map_data = slam.get_map()
    
    # 显示处理进度和结果
    slam.print_progress(iteration)
    print(f"当前位姿: x={x}mm, y={y}mm, θ={theta}°")
    slam.visualize_map(map_data)
    
    # 控制输出频率
    import time
    time.sleep(1)

