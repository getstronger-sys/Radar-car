import time
import math
import numpy as np
import matplotlib.pyplot as plt
from breezyslam.algorithms import RMHC_SLAM
from breezyslam.sensors import Laser
from roboviz import MapVisualizer
from config.map import get_global_map
from config.settings import START_POSITION, EXIT_POSITION, MAP_RESOLUTION
import heapq
from exploration.frontier_detect import ExplorationManager, world_to_grid, grid_to_world, is_exploration_complete, \
    detect_frontiers
from matplotlib.patches import Circle
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['toolbar'] = 'None'


class CarSLAM:
    def __init__(self, map_size_pixels=1024, map_size_meters=18.0, laser_params=None, grid_map=None):
        # 初始化激光模型（示例参数，需根据实际激光雷达调整）
        self.laser = Laser(scan_size=360, scan_rate_hz=100, detection_angle_degrees=360,
                           distance_no_detection_mm=10000, detection_margin=0, offset_mm=0)
        # 初始化SLAM对象
        self.slam = RMHC_SLAM(self.laser, map_size_pixels, map_size_meters)
        self.mapbytes = bytearray(map_size_pixels * map_size_pixels)
        # 新增：保存障碍物地图
        self.grid_map = grid_map
        self.map_size_pixels = map_size_pixels
        self.map_size_meters = map_size_meters

    def update_position(self, lidar_scan, pose_change, x, y, theta):
        # 假设lidar_scan是当前激光雷达扫描数据（列表形式）
        # 更新SLAM（传入里程计数据）
        self.slam.update(lidar_scan, pose_change)
        # 获取更精确的位置
        x_mm, y_mm, theta_degrees = self.slam.getpos()
        # 更新地图
        self.slam.getmap(self.mapbytes)
        return x_mm, y_mm, theta_degrees

    def simulate_straight_line(self, distance_mm=8000, step_mm=100):
        """沿直线移动小车并更新SLAM地图，起点为 START_POSITION"""
        # 起点（米转毫米）
        x = START_POSITION['x'] * 1000
        y = START_POSITION['y'] * 1000
        theta = START_POSITION['theta']
        for _ in range(int(distance_mm / step_mm)):
            x += step_mm * np.cos(np.deg2rad(theta))
            y += step_mm * np.sin(np.deg2rad(theta))
            lidar_scan = generate_lidar_scan_from_gridmap(x, y, theta, self.grid_map, MAP_RESOLUTION)
            pose_change = (step_mm, 0, 0)  # 前进step_mm，无侧向移动，无旋转
            self.update_position(lidar_scan, pose_change, x, y, theta)
            time.sleep(0.1)
        plt.pause(0.5)

    def simulate_curved_path(self, start_x_mm=1000, start_y_mm=5000, start_theta_deg=0, radius_mm=5000, angle_deg=90,
                             step_mm=100):
        """模拟小车沿圆弧路径运动"""
        # 转换角度为弧度
        angle_rad = math.radians(angle_deg)
        start_theta_rad = math.radians(start_theta_deg)

        # 计算圆弧长度和步数
        arc_length_mm = radius_mm * angle_rad
        num_steps = max(1, int(arc_length_mm / step_mm))

        # 计算圆心位置（左转）
        center_x = start_x_mm + radius_mm * math.cos(start_theta_rad + math.pi / 2)
        center_y = start_y_mm + radius_mm * math.sin(start_theta_rad + math.pi / 2)

        x, y, theta = start_x_mm, start_y_mm, start_theta_deg

        for i in range(num_steps):
            # 计算当前角度和位置
            current_angle_rad = start_theta_rad + (i / num_steps) * angle_rad + math.pi / 2
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


def generate_lidar_scan_from_gridmap(x_mm, y_mm, theta_deg, grid_map, map_resolution, max_range_mm=4000, scan_size=360):
    """
    基于栅格地图的激光雷达模拟。
    x_mm, y_mm: 机器人位置（毫米）
    theta_deg: 机器人朝向（度）
    grid_map: 2D numpy array，障碍物为1
    map_resolution: 每格米数
    max_range_mm: 最大探测距离（毫米）
    scan_size: 激光束数
    返回: 长度为 scan_size 的距离数组（毫米）
    """
    scan = []
    map_h, map_w = grid_map.shape
    x0 = x_mm / 1000 / map_resolution  # 转为格子坐标
    y0 = y_mm / 1000 / map_resolution
    for i in range(scan_size):
        angle = math.radians(theta_deg) + 2 * math.pi * i / scan_size
        dx = math.cos(angle)
        dy = math.sin(angle)
        # DDA/Bresenham
        for r in range(1, int(max_range_mm / (map_resolution * 1000))):
            x = x0 + dx * r
            y = y0 + dy * r
            xi, yi = int(round(x)), int(round(y))
            if xi < 0 or xi >= map_w or yi < 0 or yi >= map_h:
                distance = r * map_resolution * 1000
                break
            if grid_map[yi, xi] == 1:
                distance = r * map_resolution * 1000
                break
        else:
            distance = max_range_mm
        scan.append(distance)
    return scan


# ========== 常量定义 ==========
LIDAR_ANGLE_RES = 1  # 每1度一束激光，360束
LIDAR_NUM = 360
LIDAR_MAX_DIST = 20.0  # 激光最大探测距离（米，可根据需要调整）


# ========== A* 路径规划 ==========
def astar(start, goal, occ_map, map_resolution, map_size):
    sx, sy = world_to_grid(start[0], start[1], map_resolution)
    gx, gy = world_to_grid(goal[0], goal[1], map_resolution)
    open_set = []
    heapq.heappush(open_set, (0, (sx, sy)))
    came_from = {}
    g_score = {(sx, sy): 0}
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == (gx, gy):
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return [grid_to_world(x, y, map_resolution) for x, y in path]
        for dx, dy in dirs:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < map_size and 0 <= ny < map_size and occ_map[ny, nx] != 1:
                tentative_g = g_score[current] + np.hypot(dx, dy)
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + np.hypot(gx - nx, gy - ny)
                    heapq.heappush(open_set, (f, (nx, ny)))
                    came_from[(nx, ny)] = current
    return None


# ========== 地图更新 ==========
def update_known_map(pos, scan, known_map, robot_theta, map_resolution, lidar_angle_res, lidar_max_dist):
    for i, dist in enumerate(scan):
        angle = np.deg2rad(i * lidar_angle_res)
        a = robot_theta + angle
        for r in np.arange(0, min(dist, lidar_max_dist), map_resolution / 2):
            x = pos[0] + r * np.cos(a)
            y = pos[1] + r * np.sin(a)
            gx, gy = world_to_grid(x, y, map_resolution)
            if gx < 0 or gx >= known_map.shape[1] or gy < 0 or gy >= known_map.shape[0]:
                break
            if known_map[gy, gx] == -1:
                known_map[gy, gx] = 0
        if dist < lidar_max_dist:
            x = pos[0] + dist * np.cos(a)
            y = pos[1] + dist * np.sin(a)
            gx, gy = world_to_grid(x, y, map_resolution)
            if 0 <= gx < known_map.shape[1] and 0 <= gy < known_map.shape[0]:
                known_map[gy, gx] = 1


def is_frontier(known_map, gx, gy):
    if known_map[gy, gx] != 0:
        return False
    # 只要有一个邻居是未知区，就是frontier
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = gx + dx, gy + dy
        if 0 <= nx < known_map.shape[1] and 0 <= ny < known_map.shape[0]:
            if known_map[ny, nx] == -1:
                return True
    return False


# ========== 主循环 ==========
if __name__ == '__main__':
    try:
        import matplotlib.pyplot as plt
        from config.map import MAP_SIZE
        from config.settings import ROBOT_RADIUS

        true_map = get_global_map()
        known_map = np.full_like(true_map, -1, dtype=float)
        explorer = ExplorationManager(map_resolution=MAP_RESOLUTION)
        robot_pos = np.array([START_POSITION['x'], START_POSITION['y']])
        robot_theta = START_POSITION['theta']
        trajectory = [robot_pos.copy()]
        MAP_PIXELS = 1024
        MAP_METERS = 20.0
        PADDING = 7.0  # 米
        MAP_METERS = MAP_METERS + 2 * PADDING
        MAP_PIXELS = int(MAP_PIXELS + 2 * (PADDING / MAP_RESOLUTION))
        slam = CarSLAM(map_size_pixels=MAP_PIXELS, map_size_meters=MAP_METERS, grid_map=true_map)
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        # 先用起点位置做一次激光扫描，更新已知地图
        scan_mm = generate_lidar_scan_from_gridmap(
            robot_pos[0] * 1000, robot_pos[1] * 1000, np.rad2deg(robot_theta),
            true_map, MAP_RESOLUTION, max_range_mm=int(LIDAR_MAX_DIST * 1000), scan_size=LIDAR_NUM)
        scan = np.array(scan_mm) / 1000.0  # 转为米
        update_known_map(robot_pos, scan, known_map, robot_theta, MAP_RESOLUTION, LIDAR_ANGLE_RES, LIDAR_MAX_DIST)
        print('激光初始化后已知地图空地数：', np.sum(known_map == 0))
        test_frontiers = explorer.get_next_target(known_map, robot_pos)
        print('激光初始化后前沿点（目标）:', test_frontiers)
        print('进入主循环')
        step_size = 0.1  # 每次前进0.1米
        turn_step = np.deg2rad(5)  # 每次最多转5度
        while True:
            # 1. 用真实地图模拟激光
            scan_mm = generate_lidar_scan_from_gridmap(
                robot_pos[0] * 1000, robot_pos[1] * 1000, np.rad2deg(robot_theta),
                true_map, MAP_RESOLUTION, max_range_mm=int(LIDAR_MAX_DIST * 1000), scan_size=LIDAR_NUM)
            scan = np.array(scan_mm) / 1000.0  # 转为米
            # 2. 更新已知地图
            update_known_map(robot_pos, scan, known_map, robot_theta, MAP_RESOLUTION, LIDAR_ANGLE_RES, LIDAR_MAX_DIST)
            # 3. 探索逻辑
            # 获取所有frontier点
            frontiers = detect_frontiers(known_map, unknown_val=-1, free_threshold=0.2, map_resolution=MAP_RESOLUTION)
            target = None
            for f in frontiers:
                gx, gy = world_to_grid(f[0], f[1], MAP_RESOLUTION)
                if known_map[gy, gx] == 0:
                    target = f
                    break
            if target is None:
                print('探索完成！')
                break
            print(f'当前机器人位置: {robot_pos}, theta(rad): {robot_theta:.2f}')
            print(f'当前已知空地数: {np.sum(known_map == 0)}, 前沿点: {target}')
            path = astar(robot_pos, target, known_map, MAP_RESOLUTION, MAP_SIZE)
            if path is None or len(path) < 2:
                print(f'无法到达目标，path={path}，target={target}，robot_pos={robot_pos}')
                gx, gy = world_to_grid(target[0], target[1], MAP_RESOLUTION)
                print(f"A*失败，目标点格子({gx},{gy})，known_map值={known_map[gy, gx]}")
                print(
                    f"robot_pos={robot_pos}, robot格子={world_to_grid(robot_pos[0], robot_pos[1], MAP_RESOLUTION)}, known_map值={known_map[world_to_grid(robot_pos[1], robot_pos[0], MAP_RESOLUTION)]}")
                # 可视化known_map，标出robot和target
                ax1.clear()
                # 地图静止显示，origin='lower'
                ax1.imshow(true_map, cmap='gray_r', alpha=0.3, origin='lower')
                show_map = known_map.copy()
                show_map[show_map == -1] = 0.5
                ax1.imshow(show_map, cmap='Blues', alpha=0.5, origin='lower')
                # 轨迹和小车位置用世界坐标
                ax1.plot([p[0] / MAP_RESOLUTION for p in trajectory], [p[1] / MAP_RESOLUTION for p in trajectory], 'g.-', linewidth=2)
                ax1.plot(robot_pos[0] / MAP_RESOLUTION, robot_pos[1] / MAP_RESOLUTION, 'ro', markersize=8)
                if target is not None:
                    ax1.plot(target[0] / MAP_RESOLUTION, target[1] / MAP_RESOLUTION, 'yx', markersize=12)
                    ax1.plot(target[0] / MAP_RESOLUTION, target[1] / MAP_RESOLUTION, marker='*', color='y', markersize=18)
                ax1.set_xlim(0, MAP_SIZE)
                ax1.set_ylim(0, MAP_SIZE)
                circle = Circle((robot_pos[0] / MAP_RESOLUTION, robot_pos[1] / MAP_RESOLUTION), radius=ROBOT_RADIUS / MAP_RESOLUTION, fill=False, color='r', linestyle='--')
                ax1.add_patch(circle)
                ax1.set_title('真实地图/轨迹/目标/小车')
                # 右侧：SLAM建图
                ax2.clear()
                slam_map = np.array(slam.mapbytes, dtype=np.uint8).reshape(slam.map_size_pixels, slam.map_size_pixels)
                ax2.imshow(slam_map, cmap='gray', origin='lower')
                # SLAM估计的位置
                x_mm, y_mm, theta_deg = slam.slam.getpos()
                ax2.plot(x_mm / 1000 / slam.map_size_meters * slam.map_size_pixels,
                         y_mm / 1000 / slam.map_size_meters * slam.map_size_pixels, 'ro', markersize=8)
                ax2.set_title('SLAM建图与估计位置')
                plt.pause(0.01)
                continue
            # === 平滑运动到下一个A*点 ===
            for i in range(1, len(path)):
                target_pos = np.array(path[i])
                while True:
                    delta = target_pos - robot_pos
                    dist = np.linalg.norm(delta)
                    target_theta = np.arctan2(delta[1], delta[0])
                    dtheta = target_theta - robot_theta
                    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
                    if abs(dtheta) > 1e-2:
                        turn = np.clip(dtheta, -turn_step, turn_step)
                        robot_theta += turn
                        pose_change = (0, turn, 0)
                    else:
                        move = min(step_size, dist)
                        robot_pos += move * np.array([np.cos(robot_theta), np.sin(robot_theta)])
                        pose_change = (move * 1000, 0, 0)
                    # 激光模拟
                    scan_mm = generate_lidar_scan_from_gridmap(
                        robot_pos[0] * 1000, robot_pos[1] * 1000, np.rad2deg(robot_theta),
                        true_map, MAP_RESOLUTION, max_range_mm=int(LIDAR_MAX_DIST * 1000), scan_size=LIDAR_NUM)
                    scan = np.array(scan_mm) / 1000.0
                    update_known_map(robot_pos, scan, known_map, robot_theta, MAP_RESOLUTION, LIDAR_ANGLE_RES,
                                     LIDAR_MAX_DIST)
                    slam.update_position(scan_mm, pose_change, robot_pos[0] * 1000, robot_pos[1] * 1000,
                                         np.rad2deg(robot_theta))
                    trajectory.append(robot_pos.copy())
                    # 可视化
                    ax1.clear()
                    # 地图静止显示，origin='lower'
                    ax1.imshow(true_map, cmap='gray_r', alpha=0.3, origin='lower')
                    show_map = known_map.copy()
                    show_map[show_map == -1] = 0.5
                    ax1.imshow(show_map, cmap='Blues', alpha=0.5, origin='lower')
                    # 轨迹和小车位置用世界坐标
                    ax1.plot([p[0] / MAP_RESOLUTION for p in trajectory], [p[1] / MAP_RESOLUTION for p in trajectory], 'g.-', linewidth=2)
                    ax1.plot(robot_pos[0] / MAP_RESOLUTION, robot_pos[1] / MAP_RESOLUTION, 'ro', markersize=8)
                    if target is not None:
                        ax1.plot(target[0] / MAP_RESOLUTION, target[1] / MAP_RESOLUTION, 'yx', markersize=12)
                        ax1.plot(target[0] / MAP_RESOLUTION, target[1] / MAP_RESOLUTION, marker='*', color='y', markersize=18)
                    ax1.set_xlim(0, MAP_SIZE)
                    ax1.set_ylim(0, MAP_SIZE)
                    circle = Circle((robot_pos[0] / MAP_RESOLUTION, robot_pos[1] / MAP_RESOLUTION), radius=ROBOT_RADIUS / MAP_RESOLUTION, fill=False, color='r', linestyle='--')
                    ax1.add_patch(circle)
                    ax1.set_title('真实地图/轨迹/目标/小车')
                    # 右侧：SLAM建图
                    ax2.clear()
                    slam_map = np.array(slam.mapbytes, dtype=np.uint8).reshape(slam.map_size_pixels, slam.map_size_pixels)
                    ax2.imshow(slam_map, cmap='gray', origin='lower')
                    # SLAM估计的位置
                    x_mm, y_mm, theta_deg = slam.slam.getpos()
                    ax2.plot(x_mm / 1000 / slam.map_size_meters * slam.map_size_pixels,
                             y_mm / 1000 / slam.map_size_meters * slam.map_size_pixels, 'ro', markersize=8)
                    ax2.set_title('SLAM建图与估计位置')
                    plt.pause(0.01)
                    if abs(dtheta) <= 1e-2 and dist < step_size:
                        break
                # === Graph SLAM节点采样：每到达A*下一个点时采样 ===
                # 你可以在这里加Graph SLAM节点和边的采样逻辑
        plt.ioff()
        plt.show()
        # 居中窗口（仅限TkAgg后端）
        try:
            if matplotlib.get_backend() == 'TkAgg':
                plt.draw()
                fig = plt.gcf()
                mgr = plt.get_current_fig_manager()
                # 获取屏幕尺寸
                import tkinter as tk
                root = tk.Tk()
                screen_w = root.winfo_screenwidth()
                screen_h = root.winfo_screenheight()
                root.destroy()
                # 获取窗口尺寸
                w, h = fig.get_size_inches() * fig.dpi
                w, h = int(w), int(h)
                # 计算居中位置
                x = (screen_w - w) // 2
                y = (screen_h - h) // 2
                mgr.window.wm_geometry(f"{w}x{h}+{x}+{y}")
        except Exception as e:
            print('窗口居中失败:', e)
        input('按回车退出...')
    except Exception as e:
        import traceback

        print('全局异常:', e)
        traceback.print_exc()
        input('按回车退出...')