#!/usr/bin/env python3
"""
使用重构后的RoboVizSLAMViewer的完整SLAM演示
展示：实时建图、机器人位姿、激光点云、运动轨迹、导航路径
"""

import numpy as np
import time
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from viz.roboviz_slam_viz import RoboVizSLAMViewer
from config.settings import MAP_SIZE, MAP_SIZE_PIXELS, MAP_RESOLUTION

class RoboVizSLAMDemo:
    def __init__(self):
        """初始化SLAM演示"""
        # 初始化可视化器
        self.viewer = RoboVizSLAMViewer(
            title='RoboViz SLAM Demo - 实时建图与导航',
            map_size_pixels=MAP_SIZE_PIXELS,
            map_size_meters=MAP_SIZE
        )
        
        # 模拟地图数据
        self.map_bytes = bytearray(MAP_SIZE_PIXELS * MAP_SIZE_PIXELS)
        for i in range(len(self.map_bytes)):
            self.map_bytes[i] = 128  # 灰色（未知区域）
        
        # 机器人状态
        self.current_pose = (0.0, 0.0, 0.0)  # (x, y, theta)
        self.trajectory = []
        
        # 模拟迷宫环境
        self.maze_walls = self.create_maze_environment()
        
        # 导航路径（模拟A*规划结果）
        self.nav_path = self.generate_nav_path()
        
    def create_maze_environment(self):
        """创建模拟迷宫环境"""
        walls = []
        
        # 外边界
        walls.extend([
            [(0, 5), (5, 5)],  # 上边界
            [(0, 0), (5, 0)],  # 下边界  
            [(0, 0), (0, 5)],  # 左边界
            [(5, 0), (5, 5)]   # 右边界
        ])
        
        # 内部障碍物
        walls.extend([
            # 垂直墙
            [(1, 1), (1, 3)],
            [(3, 2), (3, 4)],
            # 水平墙
            [(1, 2), (3, 2)],
            [(2, 3), (4, 3)]
        ])
        
        return walls
    
    def generate_nav_path(self):
        """生成模拟导航路径"""
        # 模拟A*规划的路径
        path = [
            (0.5, 0.5),   # 起点
            (1.5, 0.5),   # 向右
            (1.5, 1.5),   # 向上
            (2.5, 1.5),   # 向右
            (2.5, 2.5),   # 向上
            (3.5, 2.5),   # 向右
            (3.5, 3.5),   # 向上
            (4.5, 3.5),   # 向右
            (4.5, 4.5)    # 目标点
        ]
        return path
    
    def simulate_lidar_scan(self, robot_pose):
        """模拟激光雷达扫描数据"""
        x, y, theta = robot_pose
        scan = []
        
        # 生成360度激光扫描
        for angle_deg in range(0, 360, 1):
            angle_rad = np.radians(angle_deg)
            global_angle = theta + angle_rad
            
            # 计算激光射线方向
            ray_dx = np.cos(global_angle)
            ray_dy = np.sin(global_angle)
            
            # 寻找最近的墙壁交点
            min_distance = 4000  # 最大距离4米
            
            for wall_start, wall_end in self.maze_walls:
                # 线段相交检测
                intersection = self.ray_wall_intersection(
                    (x, y), (ray_dx, ray_dy), wall_start, wall_end
                )
                
                if intersection:
                    dist = np.hypot(intersection[0] - x, intersection[1] - y)
                    if dist < min_distance:
                        min_distance = dist
            
            # 添加噪声
            noise = np.random.normal(0, 30)  # 30mm噪声
            min_distance += noise
            min_distance = max(0, min(min_distance, 4000))  # 限制范围
            
            scan.append(int(min_distance))
        
        return scan
    
    def ray_wall_intersection(self, ray_origin, ray_direction, wall_start, wall_end):
        """计算射线与墙壁的交点"""
        x1, y1 = ray_origin
        dx, dy = ray_direction
        x3, y3 = wall_start
        x4, y4 = wall_end
        
        # 参数方程求解
        denominator = dx * (y4 - y3) - dy * (x4 - x3)
        
        if abs(denominator) < 1e-6:
            return None  # 平行线
        
        t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / denominator
        
        if t < 0:
            return None  # 射线方向相反
        
        # 检查交点是否在墙壁线段上
        intersection_x = x1 + t * dx
        intersection_y = y1 + t * dy
        
        # 检查是否在墙壁范围内
        wall_min_x, wall_max_x = min(x3, x4), max(x3, x4)
        wall_min_y, wall_max_y = min(y3, y4), max(y3, y4)
        
        if (wall_min_x <= intersection_x <= wall_max_x and 
            wall_min_y <= intersection_y <= wall_max_y):
            return (intersection_x, intersection_y)
        
        return None
    
    def update_slam_map(self, robot_pose, lidar_scan):
        """更新SLAM地图（模拟）"""
        x, y, theta = robot_pose
        
        # 模拟地图更新过程
        for i, distance in enumerate(lidar_scan):
            if distance > 0 and distance < 4000:  # 有效距离
                angle_rad = np.radians(i)
                global_angle = theta + angle_rad
                
                # 计算激光点位置
                lidar_x = x + (distance / 1000.0) * np.cos(global_angle)
                lidar_y = y + (distance / 1000.0) * np.sin(global_angle)
                
                # 转换为地图坐标
                map_x = int((lidar_x + MAP_SIZE/2) / MAP_RESOLUTION)
                map_y = int((lidar_y + MAP_SIZE/2) / MAP_RESOLUTION)
                
                # 边界检查
                if 0 <= map_x < MAP_SIZE_PIXELS and 0 <= map_y < MAP_SIZE_PIXELS:
                    idx = map_y * MAP_SIZE_PIXELS + map_x
                    if idx < len(self.map_bytes):
                        # 根据距离设置地图值
                        if distance < 1000:  # 近距离认为是障碍物
                            self.map_bytes[idx] = 0  # 黑色（障碍物）
                        else:  # 远距离认为是自由空间
                            self.map_bytes[idx] = 255  # 白色（自由空间）
    
    def simulate_robot_motion(self, target_pose, dt=0.1):
        """模拟机器人运动到目标位置"""
        current_x, current_y, current_theta = self.current_pose
        target_x, target_y, target_theta = target_pose
        
        # 简单的直线运动
        dx = target_x - current_x
        dy = target_y - current_y
        distance = np.hypot(dx, dy)
        
        if distance > 0.1:  # 如果距离足够大
            # 计算朝向目标的角度
            target_angle = np.arctan2(dy, dx)
            angle_diff = target_angle - current_theta
            
            # 标准化角度到[-π, π]
            while angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            while angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            # 更新位姿
            step_size = 0.1  # 每步移动0.1米
            if distance > step_size:
                new_x = current_x + step_size * np.cos(target_angle)
                new_y = current_y + step_size * np.sin(target_angle)
            else:
                new_x, new_y = target_x, target_y
            
            new_theta = target_angle
            self.current_pose = (new_x, new_y, new_theta)
        
        return self.current_pose
    
    def run_demo(self):
        """运行SLAM演示"""
        print("🚀 RoboViz SLAM演示 - 实时建图与导航")
        print(f"   地图尺寸: {MAP_SIZE}x{MAP_SIZE} 米")
        print(f"   地图分辨率: {MAP_RESOLUTION} 米/格子")
        print(f"   地图像素: {MAP_SIZE_PIXELS}x{MAP_SIZE_PIXELS}")
        print("\n   可视化内容:")
        print("   - SLAM构建的占据网格地图")
        print("   - 机器人当前位姿（绿色圆点 + 红色箭头）")
        print("   - 运动轨迹（蓝色实线）")
        print("   - 导航路径（绿色虚线）")
        print("   - 激光雷达点云（红色点）")
        print("   - 障碍物（黑色区域）")
        print("   - 自由空间（白色区域）")
        print("   - 未知区域（灰色区域）")
        print("\n   请查看弹出的PyRoboViz窗口...")
        
        # 定义机器人探索路径
        exploration_path = [
            (0.5, 0.5, 0.0),      # 起点
            (1.5, 0.5, 0.0),      # 向右移动
            (1.5, 1.5, np.pi/2),  # 向上移动
            (2.5, 1.5, 0.0),      # 向右移动
            (2.5, 2.5, np.pi/2),  # 向上移动
            (3.5, 2.5, 0.0),      # 向右移动
            (3.5, 3.5, np.pi/2),  # 向上移动
            (4.5, 3.5, 0.0),      # 向右移动
            (4.5, 4.5, np.pi/2),  # 向上移动
        ]
        
        try:
            for i, target_pose in enumerate(exploration_path):
                print(f"\n📍 探索点 {i+1}/{len(exploration_path)}: {target_pose}")
                
                # 模拟机器人运动
                while True:
                    current_pose = self.simulate_robot_motion(target_pose)
                    
                    # 记录轨迹
                    self.trajectory.append(current_pose)
                    
                    # 模拟激光扫描
                    lidar_scan = self.simulate_lidar_scan(current_pose)
                    
                    # 更新SLAM地图
                    self.update_slam_map(current_pose, lidar_scan)
                    
                    # 更新可视化
                    self.viewer.update(
                        map_bytes=self.map_bytes,
                        pose=current_pose,
                        lidar_scan=lidar_scan,
                        trajectory=self.trajectory,
                        nav_path=self.nav_path
                    )
                    
                    # 检查是否到达目标
                    dist_to_target = np.hypot(
                        current_pose[0] - target_pose[0],
                        current_pose[1] - target_pose[1]
                    )
                    
                    if dist_to_target < 0.1:
                        break
                    
                    time.sleep(0.3)  # 控制更新频率
            
            print("\n✅ SLAM演示完成！")
            print("   地图已保存为: roboviz_final_slam_map.png")
            print("   按任意键退出...")
            input()
            
        except KeyboardInterrupt:
            print("\n⚠️  演示被用户中断")
        except Exception as e:
            print(f"\n❌ 演示出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 保存最终地图
            try:
                self.viewer.save_map('roboviz_final_slam_map.png')
            except:
                pass
            self.viewer.close()

def main():
    """主函数"""
    demo = RoboVizSLAMDemo()
    demo.run_demo()

if __name__ == "__main__":
    main() 