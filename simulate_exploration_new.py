# 机器人自主探索模拟程序
# 该程序模拟机器人在未知环境中使用激光雷达进行自主探索和地图构建

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config.map import get_global_map, MAP_SIZE, MAP_RESOLUTION
from config.settings import START_POSITION, ROBOT_RADIUS
from exploration.frontier_detect import ExplorationManager, world_to_grid, grid_to_world, is_exploration_complete, \
    detect_frontiers, select_best_frontier
import heapq
import json
import os
from matplotlib.colors import ListedColormap
import re
from matplotlib.patches import Circle, Wedge, FancyArrow
import mplcursors
from scipy.interpolate import interp1d
from scipy.ndimage import distance_transform_edt, gaussian_filter
import matplotlib
import matplotlib.gridspec as gridspec

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False


# ==================== 平滑运动控制类 ====================
class SmoothMotionController:
    """平滑运动控制器，结合路径平滑和DWA局部规划"""

    def __init__(self, robot_radius=ROBOT_RADIUS, max_speed=1.0, max_angular_speed=2.0):
        self.robot_radius = robot_radius
        self.max_speed = max_speed
        self.max_angular_speed = max_angular_speed
        self.current_velocity = np.array([0.0, 0.0])  # [v, omega]
        self.smooth_path = []
        self.path_index = 0
        self.goal_threshold = 0.3  # 到达目标的阈值

    def smooth_path_with_bezier(self, path, num_points=50):
        """
        使用贝塞尔曲线平滑路径

        参数:
        - path: 原始路径点列表
        - num_points: 平滑后的点数

        返回:
        - smoothed_path: 平滑后的路径
        """
        if len(path) < 3:
            return path

        # 提取控制点
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]

        # 创建贝塞尔曲线控制点（简化版本）
        # 使用路径点作为控制点，但添加中间控制点使曲线更平滑
        control_points = []
        for i in range(len(path)):
            if i == 0 or i == len(path) - 1:
                control_points.append([x_coords[i], y_coords[i]])
            else:
                # 在相邻点之间插入控制点
                prev_x, prev_y = x_coords[i - 1], y_coords[i - 1]
                curr_x, curr_y = x_coords[i], y_coords[i]
                next_x, next_y = x_coords[i + 1], y_coords[i + 1]

                # 计算平滑的控制点
                smooth_x = curr_x + 0.1 * (prev_x + next_x - 2 * curr_x)
                smooth_y = curr_y + 0.1 * (prev_y + next_y - 2 * curr_y)
                control_points.append([smooth_x, smooth_y])

        # 生成贝塞尔曲线
        t = np.linspace(0, 1, num_points)
        smoothed_path = []

        for ti in t:
            x, y = self._bezier_point(control_points, ti)
            smoothed_path.append([x, y])

        return smoothed_path

    def _bezier_point(self, control_points, t):
        """计算贝塞尔曲线上的点"""
        n = len(control_points) - 1
        x, y = 0, 0

        for i, point in enumerate(control_points):
            coef = self._binomial(n, i) * (1 - t) ** (n - i) * t ** i
            x += coef * point[0]
            y += coef * point[1]

        return x, y

    def _binomial(self, n, k):
        """计算二项式系数"""
        if k > n:
            return 0
        if k == 0 or k == n:
            return 1
        return self._binomial(n - 1, k - 1) + self._binomial(n - 1, k)

    def smooth_path_with_spline(self, path, smoothing_factor=0.3):
        """
        使用样条插值平滑路径

        参数:
        - path: 原始路径点列表
        - smoothing_factor: 平滑因子

        返回:
        - smoothed_path: 平滑后的路径
        """
        if len(path) < 2:
            return path

        # 提取坐标
        x_coords = np.array([p[0] for p in path])
        y_coords = np.array([p[1] for p in path])

        # 计算路径长度参数
        distances = np.sqrt(np.diff(x_coords) ** 2 + np.diff(y_coords) ** 2)
        cumulative_distances = np.concatenate(([0], np.cumsum(distances)))

        # 判断插值类型
        if len(path) < 4:
            kind = 'linear'
        else:
            kind = 'cubic'

        # 创建样条插值器
        if len(cumulative_distances) > 1:
            fx = interp1d(cumulative_distances, x_coords, kind=kind, bounds_error=False, fill_value=0.0)
            fy = interp1d(cumulative_distances, y_coords, kind=kind, bounds_error=False, fill_value=0.0)

            # 生成更多点以获得平滑路径
            num_points = max(50, len(path) * 3)
            new_distances = np.linspace(0, cumulative_distances[-1], num_points)

            smoothed_x = fx(new_distances)
            smoothed_y = fy(new_distances)

            smoothed_path = list(zip(smoothed_x, smoothed_y))
        else:
            smoothed_path = path

        return smoothed_path

    def update_path(self, new_path, smooth_method='spline'):
        """更新路径并平滑"""
        if not new_path or len(new_path) < 2:
            self.smooth_path = []
            self.path_index = 0
            return

        # 选择平滑方法
        if smooth_method == 'bezier':
            self.smooth_path = self.smooth_path_with_bezier(new_path)
        else:  # 默认使用样条
            self.smooth_path = self.smooth_path_with_spline(new_path)

        self.path_index = 0
        print(f"路径已平滑: {len(new_path)} -> {len(self.smooth_path)} 个点")

    def get_next_target(self, current_pos, current_theta):
        """获取下一个目标点"""
        if not self.smooth_path or self.path_index >= len(self.smooth_path):
            return None

        return np.array(self.smooth_path[self.path_index])

    def update_robot_state(self, current_pos, current_theta, dt=0.1, speed_override=None):
        """
        更新机器人状态，实现平滑运动

        参数:
        - current_pos: 当前位置 [x, y]
        - current_theta: 当前朝向
        - dt: 时间步长
        - speed_override: 速度覆盖，用于返回阶段

        返回:
        - new_pos: 新位置
        - new_theta: 新朝向
        - reached_goal: 是否到达目标
        """
        if not self.smooth_path or self.path_index >= len(self.smooth_path):
            return current_pos, current_theta, True

        target = self.get_next_target(current_pos, current_theta)
        if target is None:
            return current_pos, current_theta, True

        # 计算到目标的距离和角度
        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]
        distance = np.sqrt(dx ** 2 + dy ** 2)
        target_angle = np.arctan2(dy, dx)

        # 检查是否到达当前目标点
        if distance < self.goal_threshold:
            self.path_index += 1
            if self.path_index >= len(self.smooth_path):
                return current_pos, current_theta, True
            target = self.get_next_target(current_pos, current_theta)
            if target is None:
                return current_pos, current_theta, True
            # 重新计算到新目标的参数
            dx = target[0] - current_pos[0]
            dy = target[1] - current_pos[1]
            distance = np.sqrt(dx ** 2 + dy ** 2)
            target_angle = np.arctan2(dy, dx)

        # 计算角度差
        angle_diff = target_angle - current_theta
        # 归一化角度差到[-π, π]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        # 简化的运动控制 - 更直接地跟随路径
        # 如果角度差太大，先转向
        if abs(angle_diff) > 0.3:  # 约17度
            # 只转向，不前进
            angular_speed = np.clip(angle_diff * 1.5, -self.max_angular_speed, self.max_angular_speed)
            new_theta = current_theta + angular_speed * dt
            new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))
            return current_pos, new_theta, False
        else:
            # 角度差较小，可以前进
            speed = min(self.max_speed, 2.0) if speed_override is None else min(self.max_speed, speed_override)
            angular_speed = np.clip(angle_diff * 1.0, -self.max_angular_speed, self.max_angular_speed)

            # 更新位置和朝向
            new_theta = current_theta + angular_speed * dt
            new_theta = np.arctan2(np.sin(new_theta), np.cos(new_theta))

            new_x = current_pos[0] + speed * np.cos(new_theta) * dt
            new_y = current_pos[1] + speed * np.sin(new_theta) * dt

            return np.array([new_x, new_y]), new_theta, False


# ==================== 输出目录设置 ====================
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================== 激光雷达参数配置 ====================
LIDAR_ANGLE_RES = 2  # 激光雷达角度分辨率：每2度一束激光
LIDAR_NUM = 360 // LIDAR_ANGLE_RES  # 激光束总数：360度除以角度分辨率
LIDAR_MAX_DIST = 10.0  # 激光雷达最大探测距离（米）

# ==================== 地图初始化 ====================
true_map = get_global_map()  # 获取真实地图：0表示空地，1表示障碍物
known_map = np.full_like(true_map, -1, dtype=float)  # 初始化已知地图：-1表示未知区域，0表示空地，1表示障碍物

# ==================== 机器人状态初始化 ====================
robot_pos = np.array([START_POSITION['x'], START_POSITION['y']])  # 机器人初始位置
robot_theta = START_POSITION['theta']  # 机器人初始朝向
robot_velocity = np.array([0.0, 0.0])  # 机器人速度 [v_x, v_y]
robot_angular_velocity = 0.0  # 机器人角速度
robot_acceleration = np.array([0.0, 0.0])  # 机器人加速度 [a_x, a_y]
trajectory = [robot_pos.copy()]  # 轨迹记录列表
velocity_history = [robot_velocity.copy()]  # 速度历史
acceleration_history = [robot_acceleration.copy()]  # 加速度历史

# ==================== 探索管理器初始化 ====================
explorer = ExplorationManager(map_resolution=MAP_RESOLUTION)  # 创建探索管理器实例

# ==================== 平滑运动控制器初始化 ====================
motion_controller = SmoothMotionController(
    robot_radius=ROBOT_RADIUS,
    max_speed=8.0,  # 最大线速度 8.0 m/s
    max_angular_speed=10.0  # 最大角速度 10.0 rad/s
)


# ==================== 激光雷达扫描模拟函数 ====================
def simulate_lidar(pos, theta, true_map):
    """
    模拟激光雷达扫描过程

    参数:
    pos: 机器人当前位置 [x, y]
    theta: 机器人当前朝向
    true_map: 真实地图（用于碰撞检测）

    返回:
    scan: 激光雷达扫描结果数组，每个元素表示对应角度的距离
    """
    scan = np.zeros(LIDAR_NUM)  # 初始化扫描结果数组

    # 对每个激光束进行扫描
    for i in range(LIDAR_NUM):
        angle = np.deg2rad(i * LIDAR_ANGLE_RES)  # 将角度索引转换为弧度
        a = theta + angle  # 计算激光束的绝对角度

        # 从机器人位置开始，沿着激光束方向逐步探测
        for r in np.arange(0, LIDAR_MAX_DIST, MAP_RESOLUTION / 2):
            # 计算激光束上的点坐标
            x = pos[0] + r * np.cos(a)
            y = pos[1] + r * np.sin(a)
            # 将世界坐标转换为网格坐标
            gx, gy = world_to_grid(x, y, MAP_RESOLUTION)

            # 检查是否超出地图边界
            if gx < 0 or gx >= MAP_SIZE or gy < 0 or gy >= MAP_SIZE:
                scan[i] = r  # 记录边界距离
                break
            # 检查是否遇到障碍物
            if true_map[gy, gx] == 1:
                scan[i] = r  # 记录障碍物距离
                break
        else:
            # 如果没有遇到障碍物或边界，记录最大距离
            scan[i] = LIDAR_MAX_DIST

    return scan


# ==================== 地图更新函数 ====================
def update_known_map(pos, scan, known_map):
    """
    根据激光雷达扫描结果更新已知地图

    参数:
    pos: 机器人当前位置
    scan: 激光雷达扫描结果
    known_map: 当前已知地图（将被更新）
    """
    for i, dist in enumerate(scan):
        angle = np.deg2rad(i * LIDAR_ANGLE_RES)  # 计算激光束角度
        a = robot_theta + angle  # 计算绝对角度

        # 更新激光束路径上的空闲区域
        for r in np.arange(0, min(dist, LIDAR_MAX_DIST), MAP_RESOLUTION / 2):
            x = pos[0] + r * np.cos(a)
            y = pos[1] + r * np.sin(a)
            gx, gy = world_to_grid(x, y, MAP_RESOLUTION)

            # 检查坐标是否在地图范围内
            if gx < 0 or gx >= MAP_SIZE or gy < 0 or gy >= MAP_SIZE:
                break

            # 如果该区域之前未知，标记为空地
            if known_map[gy, gx] == -1:
                known_map[gy, gx] = 0  # 标记为空地

        # 标记障碍物位置
        if dist < LIDAR_MAX_DIST:
            x = pos[0] + dist * np.cos(a)
            y = pos[1] + dist * np.sin(a)
            gx, gy = world_to_grid(x, y, MAP_RESOLUTION)
            if 0 <= gx < MAP_SIZE and 0 <= gy < MAP_SIZE:
                known_map[gy, gx] = 1  # 标记为障碍物


# ==================== A*路径规划算法 ====================
def astar(start, goal, occ_map):
    """
    使用A*算法进行路径规划

    参数:
    start: 起始位置 [x, y]（世界坐标）
    goal: 目标位置 [x, y]（世界坐标）
    occ_map: 占用栅格地图

    返回:
    path: 路径点列表（世界坐标），如果无法找到路径则返回None
    """
    # 将世界坐标转换为网格坐标
    sx, sy = world_to_grid(start[0], start[1], MAP_RESOLUTION)
    gx, gy = world_to_grid(goal[0], goal[1], MAP_RESOLUTION)

    # 初始化A*算法数据结构
    open_set = []  # 开放列表（待探索节点）
    heapq.heappush(open_set, (0, (sx, sy)))  # 将起始节点加入开放列表
    came_from = {}  # 记录路径来源
    g_score = {(sx, sy): 0}  # 从起始点到各点的实际代价

    # 8个方向的移动（上下左右+对角线）
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_set:
        # 取出f值最小的节点
        _, current = heapq.heappop(open_set)

        # 如果到达目标，重建路径
        if current == (gx, gy):
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            # 将网格坐标转换回世界坐标
            return [grid_to_world(x, y, MAP_RESOLUTION) for x, y in path]

        # 探索当前节点的邻居
        for dx, dy in dirs:
            nx, ny = current[0] + dx, current[1] + dy

            # 检查邻居节点是否有效（在地图范围内且不是障碍物）
            if 0 <= nx < MAP_SIZE and 0 <= ny < MAP_SIZE and occ_map[ny, nx] != 1:
                # 计算从起始点经过当前节点到邻居节点的代价
                tentative_g = g_score[current] + np.hypot(dx, dy)

                # 如果找到更好的路径，更新代价和路径
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    # 计算f值（g值 + 启发式值）
                    f = tentative_g + np.hypot(gx - nx, gy - ny)
                    heapq.heappush(open_set, (f, (nx, ny)))
                    came_from[(nx, ny)] = current

    return None  # 无法找到路径


# ==================== 可视化函数 ====================
def plot_map(
        true_map, known_map, robot_pos, trajectory,
        target=None, scan=None, robot_theta=None, return_path=None,
        ax=None, frontiers=None, smooth_path=None,
        robot_velocity=None, robot_acceleration=None
):
    if ax is None:
        ax = plt.gca()
    ax.clear()
    # 只画障碍物边界线
    ax.contour(true_map, levels=[0.5], colors='k', linewidths=0.5)
    # 绘制已知地图：未知区域灰色，已知空地白色，障碍物黑色
    display_map = np.full_like(known_map, 2)
    display_map[known_map == 0] = 0  # 已知空地（白色）
    display_map[known_map == 1] = 1  # 障碍物（黑色）
    display_map[known_map == -1] = 2  # 未知（灰色）
    cmap = ListedColormap(['white', 'black', 'gray'])
    ax.imshow(display_map, cmap=cmap, origin='lower', alpha=0.25)
    # 其余内容不变
    ax.plot([p[0] / MAP_RESOLUTION for p in trajectory], [p[1] / MAP_RESOLUTION for p in trajectory], 'g.-',
            linewidth=2)
    ax.plot(robot_pos[0] / MAP_RESOLUTION, robot_pos[1] / MAP_RESOLUTION, 'ro', markersize=8)
    # 显示机器人朝向
    if robot_theta is not None:
        arrow_length = 0.5
        arrow_x = robot_pos[0] / MAP_RESOLUTION + arrow_length * np.cos(robot_theta)
        arrow_y = robot_pos[1] / MAP_RESOLUTION + arrow_length * np.sin(robot_theta)
        ax.arrow(robot_pos[0] / MAP_RESOLUTION, robot_pos[1] / MAP_RESOLUTION,
                 arrow_length * np.cos(robot_theta), arrow_length * np.sin(robot_theta),
                 head_width=0.2, head_length=0.3, fc='red', ec='red', alpha=0.8)
    if target is not None:
        ax.plot(target[0] / MAP_RESOLUTION, target[1] / MAP_RESOLUTION, 'yx', markersize=12)
        ax.plot(target[0] / MAP_RESOLUTION, target[1] / MAP_RESOLUTION, marker='*', color='y', markersize=18)
    if scan is not None and robot_theta is not None:
        for i, dist in enumerate(scan):
            angle = np.deg2rad(i * LIDAR_ANGLE_RES)
            a = robot_theta + angle
            x_end = robot_pos[0] + dist * np.cos(a)
            y_end = robot_pos[1] + dist * np.sin(a)
            ax.plot(
                [robot_pos[0] / MAP_RESOLUTION, x_end / MAP_RESOLUTION],
                [robot_pos[1] / MAP_RESOLUTION, y_end / MAP_RESOLUTION],
                color='red', alpha=0.18, linewidth=0.7
            )
    if return_path is not None:
        ax.plot([p[0] / MAP_RESOLUTION for p in return_path], [p[1] / MAP_RESOLUTION for p in return_path], 'b.-',
                linewidth=2, alpha=0.8)
    # 显示平滑路径
    if smooth_path is not None and len(smooth_path) > 0:
        ax.plot([p[0] / MAP_RESOLUTION for p in smooth_path], [p[1] / MAP_RESOLUTION for p in smooth_path], 'm--',
                linewidth=1.5, alpha=0.7, label='Smooth Path')
    start_pos = np.array([START_POSITION['x'], START_POSITION['y']])
    ax.plot(start_pos[0] / MAP_RESOLUTION, start_pos[1] / MAP_RESOLUTION, marker='*', color='green', markersize=15)
    if frontiers is not None and len(frontiers) > 0:
        fx = [f[0] / MAP_RESOLUTION for f in frontiers]
        fy = [f[1] / MAP_RESOLUTION for f in frontiers]
        sc = ax.scatter(fx, fy, c='#00ff88', s=30, alpha=0.9, marker='o')
        mplcursors.cursor(sc).connect(
            "add", lambda sel: sel.annotation.set_text(
                f"({frontiers[sel.index][0]:.2f}, {frontiers[sel.index][1]:.2f}) m"
            )
        )
    ax.set_xlim(0, MAP_SIZE)
    ax.set_ylim(0, MAP_SIZE)
    circle = patches.Circle((robot_pos[0] / MAP_RESOLUTION, robot_pos[1] / MAP_RESOLUTION),
                            radius=ROBOT_RADIUS / MAP_RESOLUTION, fill=False, color='r', linestyle='--')
    ax.add_patch(circle)

    # 显示速度向量
    if robot_velocity is not None:
        vel_magnitude = np.linalg.norm(robot_velocity)
        if vel_magnitude > 0.01:  # 只有当速度足够大时才显示
            vel_scale = 2.0  # 速度向量缩放因子
            vel_x = robot_velocity[0] * vel_scale / MAP_RESOLUTION
            vel_y = robot_velocity[1] * vel_scale / MAP_RESOLUTION
            ax.arrow(robot_pos[0] / MAP_RESOLUTION, robot_pos[1] / MAP_RESOLUTION,
                     vel_x, vel_y, head_width=0.3, head_length=0.4,
                     fc='blue', ec='blue', alpha=0.8, linewidth=2)

    # 显示加速度向量
    if robot_acceleration is not None:
        acc_magnitude = np.linalg.norm(robot_acceleration)
        if acc_magnitude > 0.01:  # 只有当加速度足够大时才显示
            acc_scale = 5.0  # 加速度向量缩放因子
            acc_x = robot_acceleration[0] * acc_scale / MAP_RESOLUTION
            acc_y = robot_acceleration[1] * acc_scale / MAP_RESOLUTION
            ax.arrow(robot_pos[0] / MAP_RESOLUTION, robot_pos[1] / MAP_RESOLUTION,
                     acc_x, acc_y, head_width=0.2, head_length=0.3,
                     fc='green', ec='green', alpha=0.8, linewidth=1.5)

    # 添加图例
    if robot_velocity is not None:
        vel_norm = np.linalg.norm(robot_velocity)
        ax.text(0.98, 0.98, f'Velocity: {vel_norm:.2f} m/s',
                transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), color='blue')
    if robot_acceleration is not None:
        acc_norm = np.linalg.norm(robot_acceleration)
        ax.text(0.98, 0.92, f'Accel: {acc_norm:.2f} m/s$^2$',
                transform=ax.transAxes, fontsize=14, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), color='green')

    ax.set_title('Ground Truth Map')


class SLAMMapVisualizer:
    def __init__(self, ax, slam_map, resolution, robot_radius=0.15):
        self.ax = ax
        self.resolution = resolution
        self.ax.set_title('SLAM Mapping', fontsize=16)
        self.ax.set_facecolor('white')
        self.ax.set_xlim(0, slam_map.shape[1])
        self.ax.set_ylim(0, slam_map.shape[0])
        self.robot_circle = Circle(
            (0, 0),
            20 * robot_radius / resolution,
            color='red',
            alpha=1.0,
            linewidth=3,
            edgecolor='black',
            zorder=20
        )
        self.ax.add_patch(self.robot_circle)
        self.path_line, = self.ax.plot([], [], color='#00aaff', linewidth=3, alpha=0.8, zorder=5)

    def update(self, slam_map, robot_pos, trajectory, scan=None, robot_theta=None, **kwargs):
        robot_x_grid = robot_pos[0] / self.resolution
        robot_y_grid = robot_pos[1] / self.resolution
        self.robot_circle.center = (robot_x_grid, robot_y_grid)
        if len(trajectory) > 1:
            path_x = [p[0] / self.resolution for p in trajectory]
            path_y = [p[1] / self.resolution for p in trajectory]
            self.path_line.set_data(path_x, path_y)
        # 先画底图：已知空地白色，障碍物黑色，未知灰色，边界模糊
        display_map = np.full_like(slam_map, 2)
        display_map[slam_map == 0] = 0  # 已知空地（白色）
        display_map[slam_map == 1] = 1  # 障碍物（黑色）
        display_map[slam_map == -1] = 2  # 未知（灰色）
        cmap = ListedColormap(['white', 'black', 'gray'])
        blurred_map = gaussian_filter(display_map.astype(float), sigma=1.2)
        self.ax.imshow(blurred_map, cmap=cmap, origin='lower', alpha=0.7, vmin=0, vmax=2)
        # 移除旧contour
        try:
            for coll in getattr(self, '_contour_collections', []):
                coll.remove()
        except Exception as e:
            print("Contour remove error:", e)
        # 画新contour（只画边界线）
        try:
            contour = self.ax.contour(slam_map, levels=[0.5], colors='k', linewidths=0.5)
            getattr(contour, 'collections', []).append(self.path_line)
        except Exception as e:
            print("Contour draw error:", e)
        self.ax.set_title('SLAM Mapping')


# ==================== 轨迹线段生成函数 ====================
def generate_trajectory_segments(trajectory, output_path):
    """
    将轨迹点序列转为线段并保存为 json 文件
    参数:
        trajectory: 轨迹点列表 (N,2)
        output_path: 输出 json 文件路径
    """
    segments = []
    for i in range(len(trajectory) - 1):
        seg = {
            "start": [float(trajectory[i][0]), float(trajectory[i][1])],
            "end": [float(trajectory[i + 1][0]), float(trajectory[i + 1][1])]
        }
        segments.append(seg)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)
    print(f"已生成 {output_path}，格式与SEGMENTS一致")


# ==================== 出口检测相关函数 ====================
def load_map_data(file_path):
    try:
        map_data = np.loadtxt(file_path, delimiter=',')
        return map_data
    except Exception as e:
        print(f"加载地图数据失败: {e}")
        return None


def find_exit_candidates(map_data, start_pos, map_resolution=MAP_RESOLUTION, min_dist_from_entry=10, central_window=3):
    height, width = map_data.shape
    start_x, start_y = int(start_pos[0] / map_resolution), int(start_pos[1] / map_resolution)
    candidates = []
    for x in range(width):
        if map_data[0, x] == 0:
            candidates.append((x, 0))
        if map_data[height - 1, x] == 0:
            candidates.append((x, height - 1))
    for y in range(height):
        if map_data[y, 0] == 0:
            candidates.append((0, y))
        if map_data[y, width - 1] == 0:
            candidates.append((width - 1, y))
    filtered = []
    for x, y in candidates:
        dist = np.hypot(x - start_x, y - start_y)
        if dist >= min_dist_from_entry:
            filtered.append((x, y))
    scored = []
    for x, y in filtered:
        free_count = 0
        for dx in range(-central_window, central_window + 1):
            for dy in range(-central_window, central_window + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if map_data[ny, nx] == 0:
                        free_count += 1
        scored.append({'grid_pos': (x, y), 'central_score': free_count})
    print(f"边界空地候选: {len(candidates)}，丢弃入口附近后: {len(filtered)}")
    return scored


def rank_exit_candidates(exit_candidates, start_pos, map_data, map_resolution=MAP_RESOLUTION):
    if not exit_candidates:
        return []
    start_x, start_y = int(start_pos[0] / map_resolution), int(start_pos[1] / map_resolution)
    ranked = []
    for item in exit_candidates:
        x, y = item['grid_pos']
        dist = np.hypot(x - start_x, y - start_y)
        score = item['central_score'] / (dist + 1)
        ranked.append({
            'grid_pos': (x, y),
            'world_pos': (x * map_resolution, y * map_resolution),
            'distance': dist,
            'central_score': item['central_score'],
            'score': score
        })
    ranked.sort(key=lambda x: x['score'], reverse=True)
    return ranked


def visualize_exit_detection(map_data, start_pos, exit_candidates, map_resolution=MAP_RESOLUTION, output_path=None):
    plt.figure(figsize=(12, 10))
    cmap = ListedColormap(['white', 'black', 'gray'])
    display_map = map_data.copy()
    display_map[display_map == -1] = 2
    plt.imshow(display_map, cmap=cmap, origin='lower')
    start_x, start_y = int(start_pos[0] / map_resolution), int(start_pos[1] / map_resolution)
    plt.plot(start_x, start_y, 'go', markersize=15, label='Start Position', markeredgecolor='black', markeredgewidth=2)
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
    for i, exit_info in enumerate(exit_candidates[:6]):
        x, y = exit_info['grid_pos']
        color = colors[i % len(colors)]
        plt.plot(x, y, 'o', color=color, markersize=12, label=f'Exit {i + 1}: Score={exit_info["score"]:.2f}')
        plt.text(x + 1, y + 1, f'{i + 1}', color=color, fontsize=12, fontweight='bold')
    plt.title('Exit Detection Results', fontsize=16)
    plt.xlabel('X Grid', fontsize=12)
    plt.ylabel('Y Grid', fontsize=12)
    plt.legend(bbox_to_anchor=(1.25, 1), loc='upper left')
    plt.colorbar(ticks=[0, 1, 2], label='0:Free 1:Obstacle 2:Unknown')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()


def detect_exits(map_file, start_pos, map_resolution=MAP_RESOLUTION, output_dir=OUTPUT_DIR):
    print("=== 出口检测算法 ===")
    map_data = load_map_data(map_file)
    if map_data is None:
        return []
    print(f"地图尺寸: {map_data.shape}")
    boundary_points = find_exit_candidates(map_data, start_pos, map_resolution)
    if not boundary_points:
        print("未找到任何边界点")
        return []
    exit_candidates = rank_exit_candidates(boundary_points, start_pos, map_data, map_resolution)
    print("\n=== 检测结果 ===")
    for i, exit_info in enumerate(exit_candidates[:5]):
        print(f"出口 {i + 1}:")
        print(f"  网格坐标: {exit_info['grid_pos']}")
        print(f"  世界坐标: {exit_info['world_pos']}")
        print(f"  到起点距离: {exit_info['distance']:.1f} 格子")
        print(f"  中央性评分: {exit_info['central_score']}")
        print(f"  综合评分: {exit_info['score']:.3f}")
        print()
    print("4. 生成可视化结果...")
    visualize_exit_detection(map_data, start_pos, exit_candidates, map_resolution,
                             output_path=os.path.join(output_dir, 'exit_detection_result.png'))
    return exit_candidates


def update_exit_position_in_settings(exit_pos, settings_path='config/settings.py'):
    """
    更新config/settings.py中的EXIT_POSITION字段为新的出口坐标
    参数:
        exit_pos: [x, y] 世界坐标
        settings_path: 配置文件路径
    """
    with open(settings_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = []
    pattern = re.compile(r"^EXIT_POSITION\s*=.*")
    replaced = False
    for line in lines:
        if pattern.match(line):
            new_line = f"EXIT_POSITION = {{'x': {exit_pos[0]:.4f}, 'y': {exit_pos[1]:.4f}, 'theta': 0.0}}  # 自动更新终点\n"
            new_lines.append(new_line)
            replaced = True
        else:
            new_lines.append(line)
    if replaced:
        with open(settings_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
        print(f"已自动更新 config/settings.py 中的 EXIT_POSITION: {exit_pos}")
    else:
        print("未找到 EXIT_POSITION 字段，未做修改。")


# ==================== 仪表盘窗口初始化 ====================
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, height_ratios=[2, 1])
ax_true = fig.add_subplot(gs[0, 0])
ax_lidar = fig.add_subplot(gs[0, 1], polar=True)  # 雷达极坐标图放中间
ax_slam = fig.add_subplot(gs[0, 2])
ax_gauge_v = fig.add_subplot(gs[1, 0])
ax_gauge_a = fig.add_subplot(gs[1, 1])
ax_gauge_theta = fig.add_subplot(gs[1, 2])
fig.suptitle('Exploration Visualization & Dashboard', fontsize=18)
plt.tight_layout(rect=(0, 0, 1, 0.96))

# 仪表盘参数
GAUGE_REFRESH_STEP = 5  # 每多少步刷新一次仪表盘


# ==================== 仪表盘绘制函数 ====================
def draw_gauge(ax, value, vmin, vmax, label, unit, color='b', arrow_color=None):
    ax.cla()
    # 仪表盘半圆
    theta1, theta2 = 210, -30  # 仪表盘起止角度
    ax.add_patch(Wedge((0, 0), 1, theta1, theta2, facecolor='#f0f0f0', edgecolor='k', lw=1.5, zorder=1))
    # 刻度
    for t in np.linspace(theta1, theta2, 7):
        x = np.cos(np.deg2rad(t))
        y = np.sin(np.deg2rad(t))
        ax.plot([0.85 * x, x], [0.85 * y, y], color='k', lw=1)
    # 指针
    angle = theta1 + (theta2 - theta1) * (value - vmin) / (vmax - vmin)
    angle = np.clip(angle, theta2, theta1)
    x = 0.7 * np.cos(np.deg2rad(angle))
    y = 0.7 * np.sin(np.deg2rad(angle))
    ax.arrow(0, 0, x, y, width=0.03, head_width=0.08, head_length=0.12, fc=arrow_color or color,
             ec=arrow_color or color, zorder=3)
    # 英文文字
    ax.text(0, -0.25, f'{label}', fontsize=13, ha='center', va='center')
    ax.text(0, -0.50, f'{value:.2f} {unit}', fontsize=14, ha='center', va='center', color=color)  # 下移一点
    ax.set_xlim(-1, 1)
    ax.set_ylim(-0.6, 1)
    ax.axis('off')


# 朝向极坐标仪表盘
polar_theta = np.linspace(0, 2 * np.pi, 100)


def draw_theta_gauge(ax, theta):
    ax.cla()
    ax.plot(np.cos(polar_theta), np.sin(polar_theta), 'k-', lw=1.5)
    ax.arrow(0, 0, 0.8 * np.cos(theta), 0.8 * np.sin(theta), width=0.03, head_width=0.08, head_length=0.12, fc='orange',
             ec='orange', zorder=3)
    ax.text(0, -1.1, 'Heading', fontsize=13, ha='center', va='center')
    ax.text(0, -1.45, f'{np.degrees(theta):.1f}°', fontsize=14, ha='center', va='center', color='orange')  # 下移
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.6, 1.2)
    ax.axis('off')


# ==================== 雷达极坐标展示函数 ====================
def plot_lidar_polar(ax, scan, angle_res=2, max_dist=10.0):
    ax.clear()
    ax.set_title('Lidar Scan', fontsize=14)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_rlim(0, max_dist)
    # 角度
    angles = np.deg2rad(np.arange(0, 360, angle_res))
    # 绘制雷达点
    ax.plot(angles, scan, 'r.', markersize=3, alpha=0.7)
    # 不再绘制0度蓝线
    ax.grid(True, alpha=0.3)


# ==================== 主探索循环 ====================
step = 0  # 步数计数器
lidar_log = []  # 激光雷达数据日志
exploration_complete = False  # 探索完成标志
return_path = []  # 返回路径记录
return_initialized = False  # 返回路径平滑初始化标志

# 调试模式：显示运动状态
DEBUG_MODE = True
print("开始自主探索...")

visualizer = SLAMMapVisualizer(ax_slam, known_map, MAP_RESOLUTION, robot_radius=ROBOT_RADIUS)

mplcursors_inited = False

VISUALIZE_EVERY_N_STEPS = 20

while True:
    # 1. 模拟激光雷达扫描
    scan = simulate_lidar(robot_pos, robot_theta, true_map)

    # 2. 根据扫描结果更新已知地图
    update_known_map(robot_pos, scan, known_map)

    # 3. 记录激光雷达数据
    lidar_log.append({
        'step': step,
        'timestamp': round(step * 0.1, 3),  # 时间戳（假设每步0.1秒）
        'robot_pos': [float(robot_pos[0]), float(robot_pos[1]), float(robot_theta)],
        'scan': scan.tolist()
    })

    # 4. 获取下一个探索目标（前沿点）
    target = explorer.get_next_target(known_map, robot_pos)

    # 5. 检查探索是否完成
    if target is None and not exploration_complete:
        print('所有前沿已探索，地图构建完成！')
        exploration_complete = True
        # 开始返回起点
        print('开始返回起点...')
        start_pos = np.array([START_POSITION['x'], START_POSITION['y']])
        return_path = astar(robot_pos, start_pos, known_map)
        if return_path is None:
            print('无法找到返回起点的路径！')
            break
        print(f'找到返回路径，共{len(return_path)}个点')

    # 6. 探索阶段：使用A*算法规划路径
    if not exploration_complete:
        path = astar(robot_pos, target, known_map)

        # 7. 检查路径是否有效
        if path is None or len(path) < 2:
            print('无法到达目标，跳过')
            continue

        # 8. 简化的路径跟随 - 直接沿路径移动
        # 更新路径到运动控制器（用于可视化）
        motion_controller.update_path(path, smooth_method='spline')

        # 选择下一个路径点（跳过当前点）
        if len(path) > 1:
            # 检查是否已经接近当前目标
            current_target = np.array(path[0])
            dist_to_current = np.linalg.norm(robot_pos - current_target)

            if dist_to_current < 0.8:  # 距离当前目标0.8米内（进一步增大到达阈值，减少停留时间）
                # 移动到下一个目标点
                next_target_idx = min(1, len(path) - 1)
                next_target = np.array(path[next_target_idx])
                print(f'到达路径点 {0}，移动到下一个点 {next_target_idx}')
            else:
                # 继续朝当前目标移动
                next_target = current_target

            # 如果路径很长，可以跳过中间点，直接朝更远的目标移动
            if len(path) > 5 and dist_to_current > 2.0:
                # 选择更远的目标点，加快移动速度
                skip_idx = min(4, len(path) - 1)
                next_target = np.array(path[skip_idx])
                print(f'激进跳点，直接朝目标点 {skip_idx} 移动')

            # 计算到目标的方向
            dx = next_target[0] - robot_pos[0]
            dy = next_target[1] - robot_pos[1]
            target_distance = np.sqrt(dx ** 2 + dy ** 2)
            target_angle = np.arctan2(dy, dx)

            # 计算角度差
            angle_diff = target_angle - robot_theta
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

            # 简化的运动控制 - 进一步提高速度
            if abs(angle_diff) > 0.1:  # 约5.7度，进一步降低转向阈值
                # 转向时也可以前进
                angular_speed = np.clip(angle_diff * 4.0, -4.0, 4.0)  # 进一步提高转向速度
                robot_angular_velocity = angular_speed
                robot_theta += angular_speed * 0.1
                robot_theta = np.arctan2(np.sin(robot_theta), np.cos(robot_theta))
                # 转向时也前进更多
                speed = 1.5
                new_velocity = speed * np.array([np.cos(robot_theta), np.sin(robot_theta)])
                new_pos = robot_pos + new_velocity * 0.1
            else:
                # 可以前进
                speed = 4.0  # 再提高前进速度
                angular_speed = np.clip(angle_diff * 2.0, -2.0, 2.0)

                robot_angular_velocity = angular_speed
                robot_theta += angular_speed * 0.1
                robot_theta = np.arctan2(np.sin(robot_theta), np.cos(robot_theta))

                new_velocity = speed * np.array([np.cos(robot_theta), np.sin(robot_theta)])
                new_pos = robot_pos + new_velocity * 0.1

            # 计算加速度
            robot_acceleration = (new_velocity - robot_velocity) / 0.1
            robot_velocity = new_velocity
        else:
            new_pos = robot_pos.copy()

        # 9. 碰撞检测
        gx, gy = world_to_grid(new_pos[0], new_pos[1], MAP_RESOLUTION)
        if 0 <= gx < MAP_SIZE and 0 <= gy < MAP_SIZE and true_map[gy, gx] == 1:
            print('碰到障碍，尝试局部避障...')
            # 简单的局部避障：尝试向左或向右偏移
            for offset in [0.2, -0.2, 0.4, -0.4]:
                offset_pos = robot_pos + offset * np.array(
                    [np.cos(robot_theta + np.pi / 2), np.sin(robot_theta + np.pi / 2)])
                gx, gy = world_to_grid(offset_pos[0], offset_pos[1], MAP_RESOLUTION)
                if 0 <= gx < MAP_SIZE and 0 <= gy < MAP_SIZE and true_map[gy, gx] != 1:
                    new_pos = offset_pos
                    print(f'局部避障成功，偏移: {offset:.2f}m')
                    break
            else:
                print('局部避障失败，跳过此目标')
                continue

        # 10. 更新机器人位置和轨迹
        robot_pos = new_pos
        trajectory.append(robot_pos.copy())
        velocity_history.append(robot_velocity.copy())
        acceleration_history.append(robot_acceleration.copy())
        step += 1

        # 调试信息
        if DEBUG_MODE and step % 20 == 0:
            vel_mag = np.linalg.norm(robot_velocity)
            acc_mag = np.linalg.norm(robot_acceleration)
            print(f"步骤 {step}: 位置=({robot_pos[0]:.2f}, {robot_pos[1]:.2f}), "
                  f"速度={vel_mag:.2f} m/s, 加速度={acc_mag:.2f} m/s²")

        # 11. 获取所有前沿点
        frontiers = detect_frontiers(known_map, map_resolution=MAP_RESOLUTION)

        # 2. 当前目标（最优前沿点）
        best_frontier = target  # target 就是你主循环里选出来的目标

        # 3. 当前A*路径（如果有）
        # path = ...  # 你主循环里A*算出来的路径

        # --- 并排更新两个窗口 ---
        if step % VISUALIZE_EVERY_N_STEPS == 0:
            plot_map(
                true_map, known_map, robot_pos, trajectory, target,
                scan=scan, robot_theta=robot_theta, return_path=return_path,
                ax=ax_true, frontiers=frontiers, smooth_path=motion_controller.smooth_path,
                robot_velocity=robot_velocity, robot_acceleration=robot_acceleration
            )
            if not mplcursors_inited and frontiers is not None and len(frontiers) > 0:
                fx = [f[0] / MAP_RESOLUTION for f in frontiers]
                fy = [f[1] / MAP_RESOLUTION for f in frontiers]
                sc = ax_true.scatter(fx, fy, c='#00ff88', s=30, alpha=0.9, marker='o')
                mplcursors.cursor(sc).connect(
                    "add", lambda sel: sel.annotation.set_text(
                        f"({frontiers[sel.index][0]:.2f}, {frontiers[sel.index][1]:.2f}) m"
                    )
                )
                mplcursors_inited = True
            visualizer.update(
                slam_map=known_map,
                robot_pos=robot_pos,
                trajectory=trajectory,
                scan=scan,
                robot_theta=robot_theta,
                frontiers=frontiers,
                best_frontier=best_frontier,
                path=path
            )
            # 雷达极坐标实时展示
            plot_lidar_polar(ax_lidar, scan, angle_res=LIDAR_ANGLE_RES, max_dist=LIDAR_MAX_DIST)
            fig.canvas.draw()
            fig.canvas.flush_events()

            # 每GAUGE_REFRESH_STEP步刷新一次仪表盘
            if step % GAUGE_REFRESH_STEP == 0:
                draw_gauge(ax_gauge_v, np.linalg.norm(robot_velocity), 0, 5, 'Velocity', 'm/s', color='blue')
                draw_gauge(ax_gauge_a, np.linalg.norm(robot_acceleration), 0, 10, 'Acceleration', 'm/s$^2$',
                           color='green')
                draw_theta_gauge(ax_gauge_theta, robot_theta)
                plt.pause(0.001)

    # 12. 返回阶段：沿返回路径移动
    else:
        # --- 严格按照平滑路径逐点跟随返回起点 ---
        if not return_initialized:
            motion_controller.update_path(return_path, smooth_method='spline')
            return_initialized = True

        # 用平滑运动控制器推进
        robot_pos, robot_theta, reached_goal = motion_controller.update_robot_state(robot_pos, robot_theta, dt=0.1,
                                                                                    speed_override=4.0)
        trajectory.append(robot_pos.copy())
        # 速度和加速度估算
        if len(velocity_history) > 0:
            robot_velocity = (robot_pos - trajectory[-2]) / 0.1 if len(trajectory) > 1 else np.array([0.0, 0.0])
            robot_acceleration = (robot_velocity - velocity_history[-1]) / 0.1 if len(
                velocity_history) > 0 else np.array([0.0, 0.0])
        else:
            robot_velocity = np.array([0.0, 0.0])
            robot_acceleration = np.array([0.0, 0.0])
        velocity_history.append(robot_velocity.copy())
        acceleration_history.append(robot_acceleration.copy())
        step += 1

        # 可视化刷新
        if step % VISUALIZE_EVERY_N_STEPS == 0:
            plot_map(
                true_map, known_map, robot_pos, trajectory, target=None,
                scan=scan, robot_theta=robot_theta, return_path=return_path,
                ax=ax_true, frontiers=None, smooth_path=motion_controller.smooth_path,
                robot_velocity=robot_velocity, robot_acceleration=robot_acceleration
            )
            visualizer.update(
                slam_map=known_map,
                robot_pos=robot_pos,
                trajectory=trajectory,
                scan=scan,
                robot_theta=robot_theta,
                frontiers=None,
                best_frontier=None,
                path=return_path
            )
            # 雷达极坐标实时展示
            plot_lidar_polar(ax_lidar, scan, angle_res=LIDAR_ANGLE_RES, max_dist=LIDAR_MAX_DIST)
            fig.canvas.draw()
            fig.canvas.flush_events()

            # 仪表盘刷新
            if step % GAUGE_REFRESH_STEP == 0:
                draw_gauge(ax_gauge_v, np.linalg.norm(robot_velocity), 0, 5, 'Velocity', 'm/s', color='blue')
                draw_gauge(ax_gauge_a, np.linalg.norm(robot_acceleration), 0, 10, 'Acceleration', 'm/s$^2$',
                           color='green')
                draw_theta_gauge(ax_gauge_theta, robot_theta)
                plt.pause(0.001)

            # 判断是否到达平滑路径终点（即起点）
            if reached_goal:
                print('已精确返回起点！')
                break

# ==================== 保存探索结果 ====================
print("探索完成，正在保存结果...")

# 保存轨迹数据
np.save(os.path.join(OUTPUT_DIR, 'exploration_trajectory.npy'), np.array(trajectory))
np.savetxt(os.path.join(OUTPUT_DIR, 'exploration_trajectory.txt'), np.array(trajectory), fmt='%.4f', delimiter=',')

# 保存轨迹线段json
generate_trajectory_segments(trajectory, os.path.join(OUTPUT_DIR, 'trajectory_segments.json'))

# 保存最终地图
np.save(os.path.join(OUTPUT_DIR, 'exploration_final_map.npy'), known_map)
np.savetxt(os.path.join(OUTPUT_DIR, 'exploration_final_map.txt'), known_map, fmt='%.2f', delimiter=',')

# 保存激光雷达数据日志
with open(os.path.join(OUTPUT_DIR, 'exploration_lidar.json'), 'w') as f:
    json.dump(lidar_log, f, indent=2, ensure_ascii=False)

print('轨迹、返回路径、已知地图和激光数据已保存。')

# ==================== 最终可视化 ====================
plt.ioff()

# 绘制速度、加速度历史图表
if len(velocity_history) > 1:
    fig_vel, (ax_vel, ax_acc) = plt.subplots(2, 1, figsize=(12, 8))
    fig_vel.suptitle('机器人运动状态历史', fontsize=16)

    # 时间轴
    time_steps = np.arange(len(velocity_history)) * 0.1

    # 速度历史
    vel_magnitudes = [np.linalg.norm(v) for v in velocity_history]
    ax_vel.plot(time_steps, vel_magnitudes, 'b-', linewidth=2, label='线速度')
    ax_vel.set_ylabel('速度 (m/s)', fontsize=12)
    ax_vel.set_title('速度历史', fontsize=14)
    ax_vel.grid(True, alpha=0.3)
    ax_vel.legend()

    # 加速度历史
    acc_magnitudes = [np.linalg.norm(a) for a in acceleration_history]
    ax_acc.plot(time_steps, acc_magnitudes, 'r-', linewidth=2, label='线加速度')
    ax_acc.set_xlabel('时间 (s)', fontsize=12)
    ax_acc.set_ylabel('加速度 (m/s²)', fontsize=12)
    ax_acc.set_title('加速度历史', fontsize=14)
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'motion_history.png'), dpi=150, bbox_inches='tight')
    print('运动状态历史图表已保存为 motion_history.png')
    plt.show()

plt.show()

# ==================== 保存最终地图图片 ====================
try:
    map_data = np.loadtxt(os.path.join(OUTPUT_DIR, 'exploration_final_map.txt'), delimiter=',')
    cmap = ListedColormap(['white', 'black', 'gray'])
    display_map = map_data.copy()
    display_map[display_map == -1] = 2
    plt.figure(figsize=(6, 6))
    plt.imshow(display_map, cmap=cmap, origin='lower')
    plt.title('Exploration Final Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(ticks=[0, 1, 2], label='0:Free 1:Obstacle 2:Unknown')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'exploration_final_map.png'), dpi=150)
    plt.close()
    print('最终地图图片 exploration_final_map.png 已保存。')
except Exception as e:
    print('保存地图图片失败:', e)

# ==================== 出口检测示例 ====================
try:
    map_file = os.path.join(OUTPUT_DIR, 'exploration_final_map.txt')
    start_pos = [START_POSITION['x'], START_POSITION['y']]
    exits = detect_exits(map_file, start_pos, MAP_RESOLUTION, OUTPUT_DIR)
    if exits:
        best_exit = exits[0]
        update_exit_position_in_settings(best_exit['world_pos'])
except Exception as e:
    print('出口检测失败:', e) 