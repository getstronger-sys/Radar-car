import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from scipy.ndimage import binary_dilation
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.map import get_global_map, MAP_SIZE_M, MAP_RESOLUTION
from config.settings import START_POSITION, EXIT_POSITION
from planner.path_planner import plan_path_simple, smooth_path_with_obstacle_avoidance
from PythonRobotics.PathPlanning.DynamicWindowApproach.dynamic_window_approach import (
    dwa_control, Config as DWAConfig, motion as dwa_motion
)


def align_to_grid_center(pos, resolution):
    return {
        'x': (int(pos['x'] / resolution) + 0.5) * resolution,
        'y': (int(pos['y'] / resolution) + 0.5) * resolution
    }


def is_path_blocked(robot_state, target, grid_map, resolution):
    # 判断机器人到目标点的直线路径上是否有障碍物
    x0, y0 = int(robot_state[0] / resolution), int(robot_state[1] / resolution)
    x1, y1 = int(target[0] / resolution), int(target[1] / resolution)
    points = bresenham(x0, y0, x1, y1)
    for x, y in points:
        if grid_map[y, x] == 1:
            return True
    return False


def bresenham(x0, y0, x1, y1):
    # Bresenham整数直线算法
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x1, y1))
    return points


def simple_goto_control(robot_state, target, max_v=0.6, w_gain=2.5):
    dx = target[0] - robot_state[0]
    dy = target[1] - robot_state[1]
    dist = np.hypot(dx, dy)
    v = min(max_v, dist)  # 距离越近速度越小
    target_theta = np.arctan2(dy, dx)
    yaw = robot_state[2]
    angle_diff = np.arctan2(np.sin(target_theta - yaw), np.cos(target_theta - yaw))
    w = w_gain * angle_diff
    return np.array([v, w])


def run_astar_follow_with_dwa(grid_map, start, goal, max_iterations=1200, goal_threshold=0.18):
    raw_path = plan_path_simple(grid_map, start, goal, MAP_RESOLUTION)
    if not raw_path:
        print("⚠️  无法找到全局路径")
        return [], [], []
    path = smooth_path_with_obstacle_avoidance(raw_path, grid_map, MAP_RESOLUTION)
    if not path or len(path) < 2:
        path = raw_path
    obs_y, obs_x = np.where(grid_map == 1)
    ob = np.vstack([
        obs_x * MAP_RESOLUTION + MAP_RESOLUTION / 2,
        obs_y * MAP_RESOLUTION + MAP_RESOLUTION / 2
    ]).T if len(obs_x) > 0 else np.zeros((0, 2))
    if len(path) > 2:
        dx = path[2][0] - path[0][0]
        dy = path[2][1] - path[0][1]
        initial_theta = np.arctan2(dy, dx)
    else:
        initial_theta = 0.0
    robot_state = np.array([start['x'], start['y'], initial_theta, 0.1, 0.0])
    robot_states = [robot_state.copy()]
    control_history = []
    config = DWAConfig()
    config.robot_radius = 0.05
    config.max_speed = 1.5
    config.min_speed = 0.0
    config.max_accel = 0.8
    config.max_yaw_rate = 2.5
    config.v_resolution = 0.08
    config.yaw_rate_resolution = 2.0 * np.pi / 180.0
    config.dt = 0.1
    config.predict_time = 2.0
    config.to_goal_cost_gain = 14.0
    config.speed_cost_gain = 0.2
    config.obstacle_cost_gain = 0.18
    path_idx = 0
    for iteration in range(max_iterations):
        if path_idx >= len(path):
            break
        target = path[path_idx]
        dist_to_target = np.hypot(robot_state[0] - target[0], robot_state[1] - target[1])
        dist_to_goal = np.hypot(robot_state[0] - goal['x'], robot_state[1] - goal['y'])
        if dist_to_goal < goal_threshold:
            print(f"✅ 到达目标! 迭代次数: {iteration}")
            break
        # 检查路径点是否被障碍物阻挡
        blocked = is_path_blocked(robot_state, target, grid_map, MAP_RESOLUTION)
        if blocked:
            # 启动DWA局部避障
            u, _ = dwa_control(robot_state, config, [target[0], target[1]], ob)
        else:
            # 直接朝路径点运动
            u = simple_goto_control(robot_state, target)
        robot_state = dwa_motion(robot_state, u, config.dt)
        robot_states.append(robot_state.copy())
        control_history.append([u[0], u[1]])
        if dist_to_target < 0.3 and path_idx < len(path) - 1:
            path_idx += 1
    return robot_states, control_history, path


# 其余可视化和主程序部分保持不变
if __name__ == "__main__":
    grid_map = get_global_map()
    grid_map_orig = grid_map.copy()  # 保存原始障碍物地图
    map_size_m = MAP_SIZE_M
    resolution = MAP_RESOLUTION
    start = align_to_grid_center(START_POSITION, resolution)
    goal = align_to_grid_center(EXIT_POSITION, resolution)
    # 机器人半径
    robot_radius = 0.05  # 与DWA config一致
    # 计算膨胀核尺寸
    dilation_radius = int(np.ceil(robot_radius / resolution))
    if dilation_radius > 0:
        structure = np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1), dtype=bool)
        dilated_grid_map = binary_dilation(grid_map == 1, structure=structure).astype(np.uint8)
    else:
        dilated_grid_map = grid_map.copy()


    def is_free(pos, grid_map, resolution):
        gx = int(pos['x'] / resolution)
        gy = int(pos['y'] / resolution)
        return grid_map[gy, gx] == 0


    if not is_free(start, dilated_grid_map, resolution):
        raise ValueError("起点在障碍物内，请选择空地作为起点！")
    if not is_free(goal, dilated_grid_map, resolution):
        raise ValueError("终点在障碍物内，请选择空地作为终点！")
    robot_states, control_history, path = run_astar_follow_with_dwa(dilated_grid_map, start, goal)
    if robot_states:
        from matplotlib import pyplot as plt


        def plot_astar_dwa(grid_map, grid_map_orig, start, goal, path, robot_states):
            fig, ax = plt.subplots(figsize=(12, 10))
            ax.imshow(grid_map, cmap='Greys', origin='lower', extent=(0, map_size_m, 0, map_size_m), alpha=0.3)
            # 原始障碍物
            obs_y, obs_x = np.where((grid_map == 1) & (grid_map_orig == 1))
            ax.scatter(obs_x * resolution + resolution / 2, obs_y * resolution + resolution / 2, c='k', s=10,
                       label='Obstacles', alpha=0.7)
            # 膨胀障碍物
            dil_y, dil_x = np.where((grid_map == 1) & (grid_map_orig == 0))
            ax.scatter(dil_x * resolution + resolution / 2, dil_y * resolution + resolution / 2, c='#39FF14', s=10,
                       label='Inflated Obstacles', alpha=0.7)
            ax.scatter([start['x']], [start['y']], c='g', s=100, marker='o', label='Start')
            ax.scatter([goal['x']], [goal['y']], c='r', s=100, marker='*', label='Goal')
            if path:
                px, py = zip(*path)
                ax.plot(px, py, 'b-', linewidth=2, label='Global Path', alpha=0.7)
            if robot_states:
                traj_x = [state[0] for state in robot_states]
                traj_y = [state[1] for state in robot_states]
                ax.plot(traj_x, traj_y, 'm-', linewidth=3, label='Trajectory')
            ax.set_xlim(0, map_size_m)
            ax.set_ylim(0, map_size_m)
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_title('A* Path Following + DWA Local Obstacle Avoidance')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()


        plot_astar_dwa(dilated_grid_map, grid_map_orig, start, goal, path, robot_states)


        def animate_astar_dwa(grid_map, grid_map_orig, start, goal, path, robot_states, control_history):
            fig, ax = plt.subplots(figsize=(10, 8))

            def animate(frame):
                ax.clear()
                ax.imshow(grid_map, cmap='Greys', origin='lower', extent=(0, map_size_m, 0, map_size_m), alpha=0.3)
                # 原始障碍物
                obs_y, obs_x = np.where((grid_map == 1) & (grid_map_orig == 1))
                ax.scatter(obs_x * resolution + resolution / 2, obs_y * resolution + resolution / 2, c='k', s=10,
                           alpha=0.7)
                # 膨胀障碍物
                dil_y, dil_x = np.where((grid_map == 1) & (grid_map_orig == 0))
                ax.scatter(dil_x * resolution + resolution / 2, dil_y * resolution + resolution / 2, c='#39FF14', s=10,
                           alpha=0.7)
                ax.scatter([start['x']], [start['y']], c='g', s=100, marker='o')
                ax.scatter([goal['x']], [goal['y']], c='r', s=100, marker='*')
                if path:
                    px, py = zip(*path)
                    ax.plot(px, py, 'b-', linewidth=2, alpha=0.7)
                if frame < len(robot_states):
                    traj_x = [state[0] for state in robot_states[:frame + 1]]
                    traj_y = [state[1] for state in robot_states[:frame + 1]]
                    ax.plot(traj_x, traj_y, 'm-', linewidth=3)
                    current_state = robot_states[frame]
                    x, y, theta = current_state[:3]
                    robot_circle = patches.Circle((x, y), 0.4, fill=False, color='red', linewidth=2)
                    ax.add_patch(robot_circle)
                    arrow_length = 0.6
                    arrow_dx = arrow_length * np.cos(theta)
                    arrow_dy = arrow_length * np.sin(theta)
                    ax.arrow(x, y, arrow_dx, arrow_dy, head_width=0.05, head_length=0.05, fc='red', ec='red')
                    # 速度
                    if frame < len(control_history):
                        v, omega = control_history[frame]
                    else:
                        v, omega = 0.0, 0.0
                    # 加速度
                    if frame > 0 and frame < len(control_history):
                        v_prev, _ = control_history[frame - 1]
                        accel = (v - v_prev) / 0.1
                    else:
                        accel = 0.0
                    # 距离终点
                    dist_to_goal = np.hypot(x - goal['x'], y - goal['y'])
                    info = (
                        f"Step: {frame}\n"
                        f"Pos: ({x:.2f}, {y:.2f})\n"
                        f"Vel: {v:.2f} m/s\n"
                        f"Accel: {accel:.2f} $m/s^2$\n"
                        f"Dist to Goal: {dist_to_goal:.2f} m"
                    )
                    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=12,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                ax.set_xlim(0, map_size_m)
                ax.set_ylim(0, map_size_m)
                ax.set_xlabel('X [m]')
                ax.set_ylabel('Y [m]')
                ax.set_title(f'A*+DWA animation - Frame {frame}')
                ax.grid(True, alpha=0.3)
                return ax.get_children()

            anim = FuncAnimation(fig, animate, frames=len(robot_states), interval=100, repeat=True, blit=False)
            plt.show()
            return anim


        animate_astar_dwa(dilated_grid_map, grid_map_orig, start, goal, path, robot_states, control_history)
    else:
        print("❌ 仿真失败，无法可视化")