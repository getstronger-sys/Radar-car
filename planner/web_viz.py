import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from config.map import get_global_map, MAP_SIZE_M, MAP_RESOLUTION
from config.settings import START_POSITION, EXIT_POSITION
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
from scipy.ndimage import binary_dilation
from planner.path_planner import plan_path, smooth_path_with_obstacle_avoidance, plan_path_simple
from PythonRobotics.PathPlanning.DynamicWindowApproach.dynamic_window_approach import dwa_control, Config as DWAConfig, motion as dwa_motion
import matplotlib.patches as patches
from collections import deque

st.set_page_config(page_title="机器人导航可视化", layout="wide")
st.title("🤖 机器人导航/SLAM/路径规划 多功能可视化系统")

# ========== 地图与障碍物膨胀 ==========
grid_map = get_global_map()
grid_map_orig = grid_map.copy()
map_size_m = MAP_SIZE_M
resolution = MAP_RESOLUTION
robot_radius = 0.05
# 膨胀障碍物
kernel = int(np.ceil(robot_radius / resolution))
if kernel > 0:
    structure = np.ones((2 * kernel + 1, 2 * kernel + 1), dtype=bool)
    dilated_grid_map = binary_dilation(grid_map == 1, structure=structure).astype(np.uint8)
else:
    dilated_grid_map = grid_map.copy()

# ========== Streamlit 侧边栏参数 ==========
with st.sidebar:
    st.header("机器人起点/终点参数")
    robot_num = st.number_input("机器人数量", 1, 5, 1)
    robot_starts = []
    robot_goals = []
    for i in range(robot_num):
        st.markdown(f"**机器人{i+1}**")
        # 默认第一个机器人用全局起点终点，其余递增/递减
        if i == 0:
            sx_default = float(START_POSITION['x'])
            sy_default = float(START_POSITION['y'])
            gx_default = float(EXIT_POSITION['x'])
            gy_default = float(EXIT_POSITION['y'])
        else:
            sx_default = float(START_POSITION['x']) + i
            sy_default = float(START_POSITION['y']) + i
            gx_default = float(EXIT_POSITION['x']) - i
            gy_default = float(EXIT_POSITION['y']) - i
        sx = st.number_input(f"起点X{i+1}", 0.0, float(map_size_m), sx_default)
        sy = st.number_input(f"起点Y{i+1}", 0.0, float(map_size_m), sy_default)
        gx = st.number_input(f"终点X{i+1}", 0.0, float(map_size_m), gx_default)
        gy = st.number_input(f"终点Y{i+1}", 0.0, float(map_size_m), gy_default)
        robot_starts.append({'x': sx, 'y': sy})
        robot_goals.append({'x': gx, 'y': gy})

# ========== 页面Tab ==========
tab1, tab2, tab3 = st.tabs(["A*路径规划与平滑", "A*+DWA仿真", "可达区域"])

# ========== 1. A*路径规划与平滑 ==========
with tab1:
    st.header("A* 路径规划与平滑路径对比与分析")
    colors = ['b', 'g', 'm', 'c', 'y']
    # 1. 只显示A*路径
    fig_astar, ax_astar = plt.subplots(figsize=(10, 8))
    ax_astar.imshow(grid_map, cmap='Greys', origin='lower', extent=(0, map_size_m, 0, map_size_m), alpha=0.3)
    obs_y, obs_x = np.where(grid_map == 1)
    ax_astar.scatter(obs_x * resolution + resolution / 2, obs_y * resolution + resolution / 2, c='k', s=10, label='障碍物', alpha=0.7)
    path_lengths = []
    for i in range(robot_num):
        start, goal = robot_starts[i], robot_goals[i]
        path = plan_path(grid_map, start, goal, smooth_path_flag=False)
        if i == 0:
            ax_astar.scatter([start['x']], [start['y']], c='lime', s=100, marker='o', label='R1 Start')
            ax_astar.scatter([goal['x']], [goal['y']], c='red', s=100, marker='*', label='R1 Goal')
            if path:
                px, py = zip(*path)
                ax_astar.plot(px, py, '-', color='blue', linewidth=2, label='R1 Path')
                path_lengths.append(("R1 Path", len(path), np.sum(np.hypot(np.diff(px), np.diff(py)))))
        else:
            ax_astar.scatter([start['x']], [start['y']], c=colors[i%len(colors)], s=100, marker='o', label=f'R{i+1} Start')
            ax_astar.scatter([goal['x']], [goal['y']], c=colors[i%len(colors)], s=100, marker='*', label=f'R{i+1} Goal')
            if path:
                px, py = zip(*path)
                ax_astar.plot(px, py, '-', color=colors[i%len(colors)], linewidth=2, label=f'R{i+1} Path')
                path_lengths.append((f'R{i+1} Path', len(path), np.sum(np.hypot(np.diff(px), np.diff(py)))))
    ax_astar.set_xlim(0, map_size_m)
    ax_astar.set_ylim(0, map_size_m)
    ax_astar.set_title("A* 路径规划结果（无障碍物膨胀）")
    ax_astar.legend()
    st.pyplot(fig_astar)

    # 2. 只显示平滑路径
    fig_smooth, ax_smooth = plt.subplots(figsize=(10, 8))
    ax_smooth.imshow(grid_map, cmap='Greys', origin='lower', extent=(0, map_size_m, 0, map_size_m), alpha=0.3)
    obs_y, obs_x = np.where(grid_map == 1)
    ax_smooth.scatter(obs_x * resolution + resolution / 2, obs_y * resolution + resolution / 2, c='k', s=10, label='障碍物', alpha=0.7)
    smooth_lengths = []
    for i in range(robot_num):
        start, goal = robot_starts[i], robot_goals[i]
        path = plan_path(grid_map, start, goal, smooth_path_flag=False)
        smoothed = smooth_path_with_obstacle_avoidance(path, grid_map, resolution)
        if i == 0:
            ax_smooth.scatter([start['x']], [start['y']], c='lime', s=100, marker='o', label='R1 Start')
            ax_smooth.scatter([goal['x']], [goal['y']], c='red', s=100, marker='*', label='R1 Goal')
            if smoothed and len(smoothed) > 2:
                spx, spy = zip(*smoothed)
                ax_smooth.plot(spx, spy, '--', color='magenta', linewidth=3, alpha=0.9, label='R1 Smoothed')
                smooth_lengths.append(("R1 Smoothed", len(smoothed), np.sum(np.hypot(np.diff(spx), np.diff(spy)))))
        else:
            ax_smooth.scatter([start['x']], [start['y']], c=colors[i%len(colors)], s=100, marker='o', label=f'R{i+1} Start')
            ax_smooth.scatter([goal['x']], [goal['y']], c=colors[i%len(colors)], s=100, marker='*', label=f'R{i+1} Goal')
            if smoothed and len(smoothed) > 2:
                spx, spy = zip(*smoothed)
                ax_smooth.plot(spx, spy, '--', color=colors[i%len(colors)], linewidth=3, alpha=0.7, label=f'R{i+1} Smoothed')
                smooth_lengths.append((f'R{i+1} Smoothed', len(smoothed), np.sum(np.hypot(np.diff(spx), np.diff(spy)))))
    ax_smooth.set_xlim(0, map_size_m)
    ax_smooth.set_ylim(0, map_size_m)
    ax_smooth.set_title("平滑路径结果（无障碍物膨胀）")
    ax_smooth.legend()
    st.pyplot(fig_smooth)

    # 3. 路径长度对比表格
    if path_lengths and smooth_lengths:
        st.subheader("路径长度对比")
        import pandas as pd
        df = pd.DataFrame(path_lengths + smooth_lengths, columns=["类型", "点数", "路径长度(m)"])
        st.dataframe(df)

# ========== 2. A*+DWA仿真与动画 ==========
with tab2:
    st.header("A*+DWA 路径跟踪仿真与动画（单机器人演示）")
    def align_to_grid_center(pos, resolution):
        return {
            'x': (int(pos['x'] / resolution) + 0.5) * resolution,
            'y': (int(pos['y'] / resolution) + 0.5) * resolution
        }
    start = align_to_grid_center(START_POSITION, resolution)
    goal = align_to_grid_center(EXIT_POSITION, resolution)
    def is_free(pos, grid_map, resolution):
        gx = int(pos['x'] / resolution)
        gy = int(pos['y'] / resolution)
        return grid_map[gy, gx] == 0
    if not is_free(start, dilated_grid_map, resolution) or not is_free(goal, dilated_grid_map, resolution):
        st.error("起点或终点在障碍物内，请调整参数！")
    else:
        def run_astar_follow_with_dwa(grid_map, start, goal, max_iterations=1200, goal_threshold=0.18):
            raw_path = plan_path_simple(grid_map, start, goal, resolution)
            if not raw_path:
                return [], [], []
            path = smooth_path_with_obstacle_avoidance(raw_path, grid_map, resolution)
            if not path or len(path) < 2:
                path = raw_path
            obs_y, obs_x = np.where(grid_map == 1)
            ob = np.vstack([
                obs_x * resolution + resolution / 2,
                obs_y * resolution + resolution / 2
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
            config.robot_radius = robot_radius
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
                    break
                # 检查路径点是否被障碍物阻挡
                def is_path_blocked(robot_state, target, grid_map, resolution):
                    x0, y0 = int(robot_state[0] / resolution), int(robot_state[1] / resolution)
                    x1, y1 = int(target[0] / resolution), int(target[1] / resolution)
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
                    for x, y in points:
                        if grid_map[y, x] == 1:
                            return True
                    return False
                blocked = is_path_blocked(robot_state, target, grid_map, resolution)
                if blocked:
                    u, _ = dwa_control(robot_state, config, [target[0], target[1]], ob)
                else:
                    dx = target[0] - robot_state[0]
                    dy = target[1] - robot_state[1]
                    target_theta = np.arctan2(dy, dx)
                    yaw = robot_state[2]
                    angle_diff = np.arctan2(np.sin(target_theta - yaw), np.cos(target_theta - yaw))
                    w = 2.5 * angle_diff
                    u = np.array([0.6, w])
                robot_state = dwa_motion(robot_state, u, config.dt)
                robot_states.append(robot_state.copy())
                control_history.append([u[0], u[1]])
                if dist_to_target < 0.3 and path_idx < len(path) - 1:
                    path_idx += 1
            return robot_states, control_history, path
        robot_states, control_history, path = run_astar_follow_with_dwa(dilated_grid_map, start, goal)
        if robot_states:
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(dilated_grid_map, cmap='Greys', origin='lower', extent=(0, map_size_m, 0, map_size_m), alpha=0.3)
            obs_y, obs_x = np.where((dilated_grid_map == 1) & (grid_map_orig == 1))
            ax.scatter(obs_x * resolution + resolution / 2, obs_y * resolution + resolution / 2, c='k', s=10, label='原始障碍物', alpha=0.7)
            dil_y, dil_x = np.where((dilated_grid_map == 1) & (grid_map_orig == 0))
            ax.scatter(dil_x * resolution + resolution / 2, dil_y * resolution + resolution / 2, c='#39FF14', s=10, label='膨胀障碍物', alpha=0.7)
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
            ax.set_title('A*+DWA 路径跟踪仿真')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            # 动画（可选）
            if st.button('播放A*+DWA动画'):
                from matplotlib.animation import FuncAnimation
                import time
                import io
                fig_anim, ax_anim = plt.subplots(figsize=(10, 8))
                def animate(frame):
                    ax_anim.clear()
                    ax_anim.imshow(dilated_grid_map, cmap='Greys', origin='lower', extent=(0, map_size_m, 0, map_size_m), alpha=0.3)
                    obs_y, obs_x = np.where((dilated_grid_map == 1) & (grid_map_orig == 1))
                    ax_anim.scatter(obs_x * resolution + resolution / 2, obs_y * resolution + resolution / 2, c='k', s=10, alpha=0.7)
                    dil_y, dil_x = np.where((dilated_grid_map == 1) & (grid_map_orig == 0))
                    ax_anim.scatter(dil_x * resolution + resolution / 2, dil_y * resolution + resolution / 2, c='#39FF14', s=10, alpha=0.7)
                    ax_anim.scatter([start['x']], [start['y']], c='g', s=100, marker='o')
                    ax_anim.scatter([goal['x']], [goal['y']], c='r', s=100, marker='*')
                    if path:
                        px, py = zip(*path)
                        ax_anim.plot(px, py, 'b-', linewidth=2, alpha=0.7)
                    if frame < len(robot_states):
                        traj_x = [state[0] for state in robot_states[:frame+1]]
                        traj_y = [state[1] for state in robot_states[:frame+1]]
                        ax_anim.plot(traj_x, traj_y, 'm-', linewidth=3)
                        current_state = robot_states[frame]
                        x, y, theta = current_state[:3]
                        robot_circle = patches.Circle((x, y), 0.4, fill=False, color='red', linewidth=2)
                        ax_anim.add_patch(robot_circle)
                        arrow_length = 0.6
                        arrow_dx = arrow_length * np.cos(theta)
                        arrow_dy = arrow_length * np.sin(theta)
                        ax_anim.arrow(x, y, arrow_dx, arrow_dy, head_width=0.05, head_length=0.05, fc='red', ec='red')
                        # 速度
                        if frame < len(control_history):
                            v, omega = control_history[frame]
                        else:
                            v, omega = 0.0, 0.0
                        # 加速度
                        if frame > 0 and frame < len(control_history):
                            v_prev, _ = control_history[frame-1]
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
                        ax_anim.text(0.02, 0.98, info, transform=ax_anim.transAxes, fontsize=12,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
                    ax_anim.set_xlim(0, map_size_m)
                    ax_anim.set_ylim(0, map_size_m)
                    ax_anim.set_xlabel('X [m]')
                    ax_anim.set_ylabel('Y [m]')
                    ax_anim.set_title(f'A*+DWA 动画')
                    ax_anim.grid(True, alpha=0.3)
                    return ax_anim.get_children()
                anim = FuncAnimation(fig_anim, animate, frames=len(robot_states), interval=100, repeat=True, blit=False)
                gif_path = 'dwa_animation.gif'
                anim.save(gif_path, writer='pillow', fps=10)
                with open(gif_path, 'rb') as f:
                    gif_bytes = f.read()
                st.image(gif_bytes, caption='A*+DWA 动画', use_container_width=True)
                plt.close(fig_anim)
                import os
                if os.path.exists(gif_path):
                    os.remove(gif_path)
        else:
            st.warning("仿真失败，无法可视化")

# ========== 3. 可达区域可视化 ==========
with tab3:
    st.header("起点可达区域可视化（Flood Fill）")
    start = robot_starts[0] if robot_starts else {'x': 0.5, 'y': 0.5}
    h, w = grid_map.shape
    visited = np.zeros_like(grid_map, dtype=bool)
    sx = int(start['x'] / resolution)
    sy = int(start['y'] / resolution)
    if grid_map[sy, sx] != 0:
        st.warning("起点在障碍物内，无法可视化可达区域！")
    else:
        queue = deque([(sx, sy)])
        visited[sy, sx] = True
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < w and 0 <= ny < h and not visited[ny, nx] and grid_map[ny, nx] == 0):
                    visited[ny, nx] = True
                    queue.append((nx, ny))
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(grid_map, cmap='Greys', origin='lower', extent=(0, map_size_m, 0, map_size_m), alpha=0.3)
        # 可达区域用浅蓝色显示
        reachable_y, reachable_x = np.where(visited)
        ax.scatter(reachable_x * resolution + resolution / 2, reachable_y * resolution + resolution / 2, c='cyan', s=15, alpha=0.6, label='Reachable Area')
        ax.scatter([start['x']], [start['y']], c='g', s=100, marker='o', label='Start')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title('Reachable Area from Start Point')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

# ========== 4. 上传/下载 ==========
st.sidebar.header("数据上传/下载")
uploaded = st.sidebar.file_uploader("上传地图/轨迹文件")
if uploaded:
    st.sidebar.success("文件已上传！")
st.sidebar.button("下载仿真结果")

st.sidebar.info("本页面为多功能演示模板，可根据需要扩展DWA仿真、SLAM地图、机器人状态等内容。")
