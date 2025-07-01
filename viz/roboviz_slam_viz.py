import numpy as np
import sys
import os

# 添加项目根目录到路径，解决导入问题
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roboviz import MapVisualizer

# 导入全局参数
from config.settings import MAP_SIZE_M, MAP_RESOLUTION
from config.map import MAP_SIZE

import matplotlib.patches as mpatches

class RoboVizSLAMViewer:
    def __init__(self, title='RoboViz SLAM Viewer', map_size_pixels=None, map_size_meters=None, use_mm=False):
        """
        增强版 PyRoboViz SLAM 可视化
        - 支持毫米单位大地图显示
        - 每帧画出激光射线
        - 机器人用三角形表示
        - 坐标轴单位和范围可自定义
        - 新增：前沿点可视化、流畅动画更新
        """
        self.use_mm = use_mm
        if use_mm:
            self.map_size_pixels = int(MAP_SIZE_M * 1000) if map_size_pixels is None else map_size_pixels
            self.map_size_meters = MAP_SIZE_M * 1000 if map_size_meters is None else map_size_meters
            self.resolution = 1.0  # 1mm/像素
        else:
            self.map_size_pixels = map_size_pixels or MAP_SIZE
            self.map_size_meters = map_size_meters or MAP_SIZE_M
            self.resolution = MAP_RESOLUTION
        self.title = title

        self.viz = MapVisualizer(
            map_size_pixels=self.map_size_pixels,
            map_size_meters=self.map_size_meters,
            title=title,
            show_trajectory=False
        )
        
        # 可视化元素
        self.traj_line = None
        self.path_line = None
        self.lidar_points = None
        self.laser_lines = []
        self.robot_patch = None
        self.obstacle_points = None
        self.frontier_points = None  # 新增：前沿点
        self.current_goal = None     # 新增：当前目标点
        self.status_text = None      # 新增：状态文本
        
        # 动画控制
        self.frame_count = 0
        self.last_update_time = 0
        
        # 设置matplotlib为交互模式，提高动画性能
        import matplotlib.pyplot as plt
        plt.ion()

    def update(self, map_bytes, pose, lidar_scan=None, trajectory=None, nav_path=None, 
               frontiers=None, current_goal=None, status_info=None):
        """
        更新可视化显示
        
        参数：
        - map_bytes: SLAM地图数据
        - pose: 机器人位姿 [x, y, theta]
        - lidar_scan: 激光扫描数据
        - trajectory: 轨迹点列表
        - nav_path: 导航路径
        - frontiers: 前沿点列表 [(x1, y1), (x2, y2), ...]
        - current_goal: 当前目标点 (x, y)
        - status_info: 状态信息字典
        """
        x, y, theta = pose
        theta_deg = np.degrees(theta)
        
        # 单位转换
        if self.use_mm:
            x_mm = x * 1000
            y_mm = y * 1000
        else:
            x_mm = x
            y_mm = y
            
        # 更新SLAM地图
        self.viz.display(x_mm, y_mm, theta_deg, map_bytes)

        # 清理旧的激光射线
        for line in self.laser_lines:
            try:
                line.remove()
            except Exception:
                pass
        self.laser_lines = []

        # 绘制激光射线
        if lidar_scan is not None:
            scan_angles = np.linspace(0, 2*np.pi, len(lidar_scan), endpoint=False)
            scan_dist = np.array(lidar_scan)
            if not self.use_mm:
                scan_dist = scan_dist / 1000.0  # 转米
            valid = (scan_dist > 0) & (scan_dist < (4000 if self.use_mm else 4.0))
            scan_angles = scan_angles[valid]
            scan_dist = scan_dist[valid]
            
            for r, a in zip(scan_dist, scan_angles):
                if self.use_mm:
                    end_x = x_mm + r * np.cos(theta + a)
                    end_y = y_mm + r * np.sin(theta + a)
                else:
                    end_x = x + r * np.cos(theta + a)
                    end_y = y + r * np.sin(theta + a)
                # 画射线 - 使用红色，更醒目
                line, = self.viz.ax.plot([x_mm, end_x], [y_mm, end_y], 
                                       color='red', linewidth=0.5, alpha=0.6, zorder=2)
                self.laser_lines.append(line)

        # 更新轨迹
        if self.traj_line:
            self.traj_line.remove()
            self.traj_line = None
        if trajectory and len(trajectory) > 1:
            if self.use_mm:
                traj_x = [p[0]*1000 for p in trajectory]
                traj_y = [p[1]*1000 for p in trajectory]
            else:
                traj_x = [p[0] for p in trajectory]
                traj_y = [p[1] for p in trajectory]
            self.traj_line, = self.viz.ax.plot(traj_x, traj_y, 'blue', 
                                             linewidth=1.5, alpha=0.8, zorder=4, 
                                             label='Robot Trajectory')

        # 更新导航路径
        if self.path_line:
            self.path_line.remove()
            self.path_line = None
        if nav_path and len(nav_path) > 1:
            if self.use_mm:
                path_x = [p[0]*1000 for p in nav_path]
                path_y = [p[1]*1000 for p in nav_path]
            else:
                path_x = [p[0] for p in nav_path]
                path_y = [p[1] for p in nav_path]
            self.path_line, = self.viz.ax.plot(path_x, path_y, 'green', 
                                             linewidth=2, alpha=0.7, zorder=3, 
                                             linestyle='--', label='Navigation Path')

        # 更新前沿点
        if self.frontier_points:
            self.frontier_points.remove()
            self.frontier_points = None
        if frontiers and len(frontiers) > 0:
            if self.use_mm:
                frontier_x = [f[0]*1000 for f in frontiers]
                frontier_y = [f[1]*1000 for f in frontiers]
            else:
                frontier_x = [f[0] for f in frontiers]
                frontier_y = [f[1] for f in frontiers]
            self.frontier_points = self.viz.ax.scatter(frontier_x, frontier_y, 
                                                     c='yellow', s=30, alpha=0.8, 
                                                     edgecolors='orange', linewidth=1,
                                                     label='Frontiers', zorder=5)

        # 更新当前目标点
        if self.current_goal:
            self.current_goal.remove()
            self.current_goal = None
        if current_goal:
            if self.use_mm:
                goal_x = current_goal[0] * 1000
                goal_y = current_goal[1] * 1000
            else:
                goal_x = current_goal[0]
                goal_y = current_goal[1]
            self.current_goal = self.viz.ax.scatter(goal_x, goal_y, 
                                                  c='magenta', s=100, alpha=0.9,
                                                  edgecolors='purple', linewidth=2,
                                                  marker='*', label='Current Goal', zorder=6)

        # 更新机器人三角形
        if self.robot_patch:
            self.robot_patch.remove()
            self.robot_patch = None
        # 机器人三角形参数
        tri_len = 200 if self.use_mm else 0.2  # 20cm
        tri_wid = 100 if self.use_mm else 0.1  # 10cm
        # 机器人朝向三角形顶点
        tip = (x_mm + tri_len * np.cos(theta), y_mm + tri_len * np.sin(theta))
        left = (x_mm + tri_wid * np.cos(theta + 2.5), y_mm + tri_wid * np.sin(theta + 2.5))
        right = (x_mm + tri_wid * np.cos(theta - 2.5), y_mm + tri_wid * np.sin(theta - 2.5))
        triangle = np.array([tip, left, right])
        self.robot_patch = mpatches.Polygon(triangle, color='red', zorder=10)
        self.viz.ax.add_patch(self.robot_patch)

        # 更新状态文本
        if self.status_text:
            self.status_text.remove()
            self.status_text = None
        if status_info:
            status_str = f"Frame: {self.frame_count}\n"
            for key, value in status_info.items():
                status_str += f"{key}: {value}\n"
            self.status_text = self.viz.ax.text(0.02, 0.98, status_str,
                                              transform=self.viz.ax.transAxes,
                                              verticalalignment='top',
                                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                                              fontsize=8, zorder=20)

        # 更新图例
        if not self.viz.ax.get_legend():
            self.viz.ax.legend(loc='upper right', fontsize=8)

        # 更新计数
        self.frame_count += 1

    def add_obstacle_points(self, obstacle_points):
        """
        添加障碍物点可视化
        
        参数：
        - obstacle_points: [(x1, y1), (x2, y2), ...] 障碍物点列表
        """
        if self.obstacle_points:
            self.obstacle_points.remove()
            self.obstacle_points = None
        
        if obstacle_points and len(obstacle_points) > 0:
            if self.use_mm:
                obs_x = [p[0]*1000 for p in obstacle_points]
                obs_y = [p[1]*1000 for p in obstacle_points]
            else:
                obs_x = [p[0] for p in obstacle_points]
                obs_y = [p[1] for p in obstacle_points]
            
            self.obstacle_points = self.viz.ax.scatter(obs_x, obs_y, 
                                                     c='black', s=20, alpha=0.8,
                                                     label='Obstacles', zorder=1)

    def save_map(self, filename='roboviz_slam_map.png'):
        """
        保存当前地图为图片
        """
        try:
            if hasattr(self.viz, 'fig'):
                self.viz.fig.savefig(filename, dpi=150, bbox_inches='tight')  # type: ignore
            elif hasattr(self.viz, 'figure'):
                self.viz.figure.savefig(filename, dpi=150, bbox_inches='tight')  # type: ignore
            elif hasattr(self.viz, 'ax'):
                self.viz.ax.figure.savefig(filename, dpi=150, bbox_inches='tight')  # type: ignore
            else:
                import matplotlib.pyplot as plt
                plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"✅ 地图已保存: {filename}")
        except Exception as e:
            print(f"❌ 保存地图失败: {e}")

    def close(self):
        """
        关闭可视化窗口
        """
        try:
            if hasattr(self.viz, 'close'):
                self.viz.close()
            else:
                import matplotlib.pyplot as plt
                plt.close('all')
        except Exception as e:
            print(f"关闭可视化窗口时出错: {e}")

    def show(self):
        """
        显示可视化窗口
        """
        try:
            if hasattr(self.viz, 'show'):
                self.viz.show()
            else:
                import matplotlib.pyplot as plt
                plt.show()
        except Exception as e:
            print(f"显示可视化窗口时出错: {e}")

    def get_frame_rate(self):
        """
        获取当前帧率
        """
        import time
        current_time = time.time()
        if self.last_update_time > 0:
            fps = 1.0 / (current_time - self.last_update_time)
        else:
            fps = 0
        self.last_update_time = current_time
        return fps


def test_roboviz_viewer():
    """
    测试RoboViz SLAM可视化器
    """
    import time
    
    # 创建可视化器
    viewer = RoboVizSLAMViewer(title='Test SLAM Viewer', use_mm=False)
    
    # 模拟数据
    map_bytes = bytearray(MAP_SIZE * MAP_SIZE)
    pose = [MAP_SIZE_M/2, MAP_SIZE_M/2, 0.0]  # 地图中心
    
    # 模拟激光扫描数据
    lidar_scan = [1000 + i * 50 for i in range(360)]
    
    # 模拟轨迹
    trajectory = []
    for i in range(10):
        x = MAP_SIZE_M/2 + i * 0.3
        y = MAP_SIZE_M/2 + i * 0.1
        theta = i * 0.1
        trajectory.append([x, y])
    
    # 模拟前沿点
    frontiers = [[1.0, 1.0], [2.0, 2.0], [3.0, 1.5]]
    
    # 模拟目标点
    current_goal = [MAP_SIZE_M-1, MAP_SIZE_M-1]
    
    # 更新可视化
    viewer.update(map_bytes, pose, lidar_scan, trajectory, None, frontiers, current_goal)
    
    print("✅ RoboViz SLAM可视化器测试完成")
    print(f"地图尺寸: {MAP_SIZE}x{MAP_SIZE} 像素")
    print(f"地图物理尺寸: {MAP_SIZE_M}x{MAP_SIZE_M} 米")
    print(f"分辨率: {MAP_RESOLUTION} 米/像素")
    
    # 保存测试图片
    viewer.save_map('test_slam_map.png')
    
    return viewer


if __name__ == "__main__":
    # 运行测试
    viewer = test_roboviz_viewer()
    viewer.show() 