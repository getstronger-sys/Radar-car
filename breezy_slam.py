# -*- coding: utf-8 -*-

from breezyslam.algorithms import RMHC_SLAM, Deterministic_SLAM
from breezyslam.sensors import Laser
import numpy as np
import warnings
import sys
sys.path.append('./PyRoboViz-master')
from roboviz import MapVisualizer

# 从配置文件导入参数
from config.map import MAP_SIZE, RESOLUTION

# 忽略BreezySLAM的警告
warnings.filterwarnings('ignore', message='.*No error gradient.*')

SLAM_MAP_SIZE_PIXELS = MAP_SIZE
SLAM_MAP_SCALE_METERS_PER_PIXEL = RESOLUTION
LIDAR_ANGLES_NUM = 360  # 激光束数量
LIDAR_RANGE = 4.0  # 激光雷达最大探测距离（米）

class BreezySLAMAdapter:
    """
    BreezySLAM的适配器类，用于与本项目的仿真环境对接。
    """

    def __init__(self, use_deterministic=False):
        try:
            # 创建激光雷达传感器模型
            scan_rate_hz = 10.0
            detection_angle_degrees = 360.0
            # distance_no_detection_mm为最大测距的4倍
            self.distance_no_detection_mm = int(LIDAR_RANGE * 1000 * 1)

            self.laser = Laser(
                LIDAR_ANGLES_NUM,
                scan_rate_hz,
                detection_angle_degrees,
                self.distance_no_detection_mm,
                0,  # detection_margin
                0  # offset_mm
            )

            # 计算地图参数 - 根据迷宫实际大小调整
            # 迷宫大约15x15米，设置20x20米的地图
            self.map_size_meters = SLAM_MAP_SIZE_PIXELS * SLAM_MAP_SCALE_METERS_PER_PIXEL
            self.map_size_pixels = SLAM_MAP_SIZE_PIXELS
            self.map_scale = SLAM_MAP_SCALE_METERS_PER_PIXEL

            # 创建SLAM算法实例
            if use_deterministic:
                print("使用Deterministic SLAM（无随机搜索）")
                self.slam = Deterministic_SLAM(
                    self.laser,
                    self.map_size_pixels,
                    self.map_size_meters,
                    map_quality=50,
                    hole_width_mm=600  # 恢复默认值
                )
            else:
                print("使用RMHC SLAM（随机搜索）")
                self.slam = RMHC_SLAM(
                    self.laser,
                    self.map_size_pixels,
                    self.map_size_meters,
                    map_quality=50,
                    hole_width_mm=600,
                    random_seed=42,
                    sigma_xy_mm=100,
                    sigma_theta_degrees=20,
                    max_search_iter=1000
                )

            # 地图数据
            self.map_bytes = bytearray(self.map_size_pixels * self.map_size_pixels)
            self.map_image = np.zeros((self.map_size_pixels, self.map_size_pixels), dtype=np.uint8)

            # 初始化标志
            self.initialized = False
            self.update_count = 0
            self.error_count = 0

            # 里程计历史
            self.last_odom = None

            print(f"SLAM适配器初始化成功 - 地图大小: {self.map_size_meters}m x {self.map_size_meters}m")
            print(f"地图比例: {self.map_scale:.3f} m/pixel")

        except Exception as e:
            print(f"SLAM适配器初始化失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def update(self, scan_distances_mm, odometry_mm):
        """
        使用新的激光雷达和里程计数据更新SLAM状态。
        """
        try:
            x_mm, y_mm, theta_deg = odometry_mm
            self.update_count += 1

            # 处理激光雷达数据
            processed_scan = []

            # 确保扫描数据长度正确
            if len(scan_distances_mm) != LIDAR_ANGLES_NUM:
                print(f"警告: 扫描数据长度不匹配 {len(scan_distances_mm)} != {LIDAR_ANGLES_NUM}")
                # 填充或截断
                if len(scan_distances_mm) < LIDAR_ANGLES_NUM:
                    scan_distances_mm = list(scan_distances_mm) + [self.distance_no_detection_mm] * (
                                LIDAR_ANGLES_NUM - len(scan_distances_mm))
                else:
                    scan_distances_mm = scan_distances_mm[:LIDAR_ANGLES_NUM]

            # 处理每个扫描点
            for dist in scan_distances_mm:
                if isinstance(dist, (int, float)) and 0 < dist < LIDAR_RANGE * 1000:
                    processed_scan.append(int(dist))
                elif isinstance(dist, (int, float)) and dist >= LIDAR_RANGE * 1000:
                    processed_scan.append(0)  # 没检测到墙，返回0
                else:
                    processed_scan.append(self.distance_no_detection_mm)

            # 调试激光雷达数据
            if self.update_count % 50 == 0:
                valid_distances = [d for d in processed_scan if d < 6000]
                invalid_count = len(processed_scan) - len(valid_distances)
                if valid_distances:
                    min_valid = min(valid_distances)
                    max_valid = max(valid_distances)
                    avg_valid = sum(valid_distances) / len(valid_distances)
                    print(f"    激光雷达: 有效距离{len(valid_distances)}个, 无效{invalid_count}个")
                    print(f"    有效距离范围: {min_valid}-{max_valid}mm, 平均{avg_valid:.1f}mm")
                else:
                    print(f"    激光雷达: 所有距离都无效({invalid_count}个)")

            # 计算位姿变化
            if self.last_odom is None:
                # 第一次更新
                pose_change = (0, 0, 0.1)
                self.last_odom = (x_mm, y_mm, theta_deg)
            else:
                # 计算相对于上次的变化
                last_x, last_y, last_theta = self.last_odom

                # 计算位置变化（欧几里得距离）
                dx = x_mm - last_x
                dy = y_mm - last_y
                dxy_mm = np.sqrt(dx * dx + dy * dy)

                # 角度变化
                dtheta = theta_deg - last_theta
                # 归一化到[-180, 180]
                while dtheta > 180: dtheta -= 360
                while dtheta < -180: dtheta += 360

                pose_change = (dxy_mm, dtheta, 0.1)
                self.last_odom = (x_mm, y_mm, theta_deg)

            # 更新SLAM（捕获可能的错误）
            try:
                self.slam.update(processed_scan, pose_change)
            except Exception as e:
                self.error_count += 1
                if self.error_count % 10 == 0:
                    print(f"SLAM更新错误 #{self.error_count}: {e}")

            # 获取地图
            try:
                self.slam.getmap(self.map_bytes)
                self.map_image = np.array(self.map_bytes).reshape((self.map_size_pixels, self.map_size_pixels))
            except Exception as e:
                print(f"获取地图错误: {e}")

            # 周期性输出状态
            if self.update_count % 50 == 0:
                slam_x, slam_y, slam_theta = self.slam.getpos()
                non_zero = np.count_nonzero(self.map_image)
                print(f"[{self.update_count}] SLAM位置: ({slam_x / 1000:.1f},{slam_y / 1000:.1f})m, "
                      f"地图非零像素: {non_zero}/{self.map_size_pixels * self.map_size_pixels}")
                print(f"    实际位置: ({x_mm / 1000:.1f},{y_mm / 1000:.1f})m, 角度: {theta_deg:.1f}°")

            return self.slam.getpos(), self.map_image

        except Exception as e:
            print(f"SLAM更新异常: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认值
            return (0, 0, 0), self.map_image

    def get_map(self):
        """获取并返回SLAM地图"""
        try:
            self.slam.getmap(self.map_bytes)
            np_map = np.array(self.map_bytes).reshape((self.map_size_pixels, self.map_size_pixels))
            return np_map
        except:
            return self.map_image

    def get_pose(self):
        """返回当前的机器人位姿"""
        try:
            return self.slam.getpos()
        except:
            return (0, 0, 0)

    def update_slam_map_visualization(self, slam_map):
        """SLAM地图平移对齐仿真起点"""
        if slam_map is not None:
            # 只做旋转，不翻转
            aligned_map = np.rot90(slam_map)
            # 获取仿真起点（米）
            sim_start_x, sim_start_y = self.maze_env.start_pos
            # SLAM地图分辨率
            map_resolution = 0.1  # 每像素0.1米
            # 计算仿真起点在SLAM地图中的像素坐标
            map_x = int(sim_start_x / map_resolution)
            map_y = int(sim_start_y / map_resolution)
            # extent参数设置地图的物理范围，使仿真起点在主图和SLAM图上重合
            extent = [
                -map_x, aligned_map.shape[1] - map_x,
                -map_y, aligned_map.shape[0] - map_y
            ]
            if self.slam_map_artist:
                self.slam_map_artist.set_data(aligned_map)
                self.slam_map_artist.set_extent(extent)
            else:
                self.slam_map_artist = self.ax3.imshow(aligned_map, cmap='gray', origin='lower', extent=extent)
            self.ax3.set_xlim(extent[0], extent[1])
            self.ax3.set_ylim(extent[2], extent[3])

def main():
    # 1. 加载地图
    grid_map = get_global_map()
    resolution = RESOLUTION
    start = START_POSITION
    robot_pose = [start['x'], start['y'], start['theta']]

    # 2. 初始化SLAM
    slam = BreezySLAMAdapter()

    # 3. 初始化PyRoboViz
    map_size_pixels = grid_map.shape[0]
    map_size_meters = map_size_pixels * resolution
    viz = MapVisualizer(map_size_pixels, map_size_meters, title='SLAM仿真', show_trajectory=True)

    # 4. 主循环
    for step in range(500):
        # ... 机器人运动、激光模拟、SLAM更新 ...
        viz.display(robot_pose[0], robot_pose[1], robot_pose[2], slam_map)