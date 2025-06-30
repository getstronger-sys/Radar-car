import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# 添加项目根目录路径，使得可以导入 planner 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planner.path_planner import plan_path, smooth_path
from planner.dwa_planner import DWAPlanner

# ========== 1. 构建地图 ==========
map_size = 50            # 地图为 50x50 的栅格地图
map_size_m = 5.0         # 实际物理尺寸为 5.0m x 5.0m
resolution = map_size_m / map_size  # 每个栅格表示的实际长度（米）

# 初始化空地图（0 表示空地）
grid_map = np.zeros((map_size, map_size), dtype=np.uint8)


# ========== 2. DWA配置 ==========
dwa_config = {
    'max_speed': 5,        # 最大线速度 (m/s) - 降低速度
    'min_speed': 0.0,        # 最小线速度 (m/s)
    'max_yawrate': 5,      # 最大角速度 (rad/s) - 降低角速度
    'max_accel': 5,         # 提高加速度
    'max_dyawrate': 2,     # 最大角加速度 (rad/s²) - 降低角加速度
    'v_reso': 0.02,          # 线速度分辨率 (m/s) - 提高分辨率
    'yawrate_reso': 0.05,    # 角速度分辨率 (rad/s) - 提高分辨率
    'dt': 0.1,               # 时间步长 (s)
    'predict_time': 3.0,     # 预测时间 (s) - 增加预测时间
    'to_goal_cost_gain': 1.0,    # 降低目标代价权重
    'speed_cost_gain': 0.05,     # 速度代价权重 - 降低权重
    'obstacle_cost_gain': 1.5,   # 障碍物代价权重 - 增加权重
    'robot_radius': 0.02,        # 机器人半径 (m) - 减小半径
    'map_resolution': resolution  # 地图分辨率
}

# ========== 3. 机器人运动模型 ==========
def motion_model(state, control, dt):
    """
    机器人运动模型
    
    参数:
    - state: [x, y, theta] 当前状态
    - control: [v, omega] 控制输入
    - dt: 时间步长
    
    返回:
    - new_state: [x, y, theta] 新状态
    """
    x, y, theta = state
    v, omega = control
    
    # 简单的差分驱动模型
    x_new = x + v * np.cos(theta) * dt
    y_new = y + v * np.sin(theta) * dt
    theta_new = theta + omega * dt
    
    # 角度归一化到 [-π, π]
    theta_new = np.arctan2(np.sin(theta_new), np.cos(theta_new))
    
    return [x_new, y_new, theta_new]

# ========== 4. 可视化函数 ==========
def plot_dwa_simulation(grid_map, start, goal, path, dwa_trajectory, robot_states, 
                       control_history, animation_mode=False):
    """
    可视化DWA仿真结果
    
    参数:
    - grid_map: 栅格地图
    - start: 起始位置
    - goal: 目标位置
    - path: 全局路径
    - dwa_trajectory: DWA轨迹历史
    - robot_states: 机器人状态历史
    - control_history: 控制输入历史
    - animation_mode: 是否为动画模式
    """
    if animation_mode:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig, ax = plt.subplots(figsize=(12, 10))
        # 创建子图布局
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[3, 1])
        ax = fig.add_subplot(gs[0, 0])  # 主轨迹图
        ax_control = fig.add_subplot(gs[1, 0])  # 控制输入图
        ax_metrics = fig.add_subplot(gs[0, 1])  # 指标图

    # 显示背景地图
    ax.imshow(grid_map, cmap='Greys', origin='lower',
              extent=(0, map_size_m, 0, map_size_m), alpha=0.3)

    # 绘制障碍物点
    obs_y, obs_x = np.where(grid_map == 1)
    ax.scatter(obs_x * resolution + resolution / 2,
               obs_y * resolution + resolution / 2,
               c='k', s=10, label='Obstacles', alpha=0.7)

    # 起点和终点
    ax.scatter([start['x']], [start['y']], c='g', s=100, marker='o', label='Start')
    ax.scatter([goal['x']], [goal['y']], c='r', s=100, marker='*', label='Goal')

    # 全局路径
    if path:
        px, py = zip(*path)
        ax.plot(px, py, 'b-', linewidth=2, label='Global Path', alpha=0.7)

    # DWA轨迹
    if dwa_trajectory:
        traj_x = [state[0] for state in dwa_trajectory]
        traj_y = [state[1] for state in dwa_trajectory]
        ax.plot(traj_x, traj_y, 'm-', linewidth=3, label='DWA Trajectory')

    # 机器人当前位置（最后一个状态）
    if robot_states:
        current_state = robot_states[-1]
        x, y, theta = current_state
        
        # 绘制机器人（圆形）
        robot_circle = patches.Circle((x, y), dwa_config['robot_radius'], 
                                     fill=False, color='red', linewidth=2, label='Robot')
        ax.add_patch(robot_circle)
        
        # 绘制机器人朝向
        arrow_length = dwa_config['robot_radius'] * 1.5
        arrow_dx = arrow_length * np.cos(theta)
        arrow_dy = arrow_length * np.sin(theta)
        ax.arrow(x, y, arrow_dx, arrow_dy, head_width=0.05, head_length=0.05, 
                fc='red', ec='red')

    # 设置图像范围与标签
    ax.set_xlim(0, map_size_m)
    ax.set_ylim(0, map_size_m)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('DWA Path Planning with Dynamics')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if not animation_mode:
        # 控制输入图
        if control_history:
            times = np.arange(len(control_history)) * dwa_config['dt']
            velocities = [control[0] for control in control_history]
            angular_velocities = [control[1] for control in control_history]
            
            ax_control.plot(times, velocities, 'b-', label='Linear Velocity', linewidth=2)
            ax_control.set_ylabel('Velocity [m/s]')
            ax_control.set_xlabel('Time [s]')
            ax_control.legend()
            ax_control.grid(True, alpha=0.3)
            
            # 角速度（双轴）
            ax_control_twin = ax_control.twinx()
            ax_control_twin.plot(times, angular_velocities, 'r-', label='Angular Velocity', linewidth=2)
            ax_control_twin.set_ylabel('Angular Velocity [rad/s]', color='r')
            ax_control_twin.tick_params(axis='y', labelcolor='r')

        # 指标图
        if robot_states and path:
            # 计算到目标的距离
            goal_distances = []
            for state in robot_states:
                dist = np.hypot(state[0] - goal['x'], state[1] - goal['y'])
                goal_distances.append(dist)
            
            times = np.arange(len(robot_states)) * dwa_config['dt']
            ax_metrics.plot(times, goal_distances, 'g-', linewidth=2)
            ax_metrics.set_ylabel('Distance to Goal [m]')
            ax_metrics.set_xlabel('Time [s]')
            ax_metrics.set_title('Performance Metrics')
            ax_metrics.grid(True, alpha=0.3)
            
            # 添加最终距离信息
            final_distance = goal_distances[-1] if goal_distances else 0
            ax_metrics.text(0.05, 0.95, f'Final Distance: {final_distance:.3f}m', 
                           transform=ax_metrics.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()

def animate_dwa_simulation(grid_map, start, goal, path, robot_states, control_history):
    """
    创建DWA仿真的动画，并实时显示位置、速度、加速度、距离终点
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    def animate(frame):
        ax.clear()
        
        # 显示背景地图
        ax.imshow(grid_map, cmap='Greys', origin='lower',
                  extent=(0, map_size_m, 0, map_size_m), alpha=0.3)
        
        # 绘制障碍物
        obs_y, obs_x = np.where(grid_map == 1)
        ax.scatter(obs_x * resolution + resolution / 2,
                   obs_y * resolution + resolution / 2,
                   c='k', s=10, alpha=0.7)
        
        # 起点和终点
        ax.scatter([start['x']], [start['y']], c='g', s=100, marker='o')
        ax.scatter([goal['x']], [goal['y']], c='r', s=100, marker='*')
        
        # 全局路径
        if path:
            px, py = zip(*path)
            ax.plot(px, py, 'b-', linewidth=2, alpha=0.7)
        
        # 机器人轨迹（到当前帧）
        if frame < len(robot_states):
            traj_x = [state[0] for state in robot_states[:frame+1]]
            traj_y = [state[1] for state in robot_states[:frame+1]]
            ax.plot(traj_x, traj_y, 'm-', linewidth=3)
            
            # 当前机器人位置
            current_state = robot_states[frame]
            x, y, theta = current_state
            
            # 绘制机器人
            robot_circle = patches.Circle((x, y), dwa_config['robot_radius'], 
                                         fill=False, color='red', linewidth=2)
            ax.add_patch(robot_circle)
            
            # 绘制朝向
            arrow_length = dwa_config['robot_radius'] * 1.5
            arrow_dx = arrow_length * np.cos(theta)
            arrow_dy = arrow_length * np.sin(theta)
            ax.arrow(x, y, arrow_dx, arrow_dy, head_width=0.05, head_length=0.05, 
                    fc='red', ec='red')
            
            # ==== 实时数据显示 ====
            # 速度
            if frame < len(control_history):
                v, omega = control_history[frame]
            else:
                v, omega = 0.0, 0.0
            # 加速度
            if frame > 0 and frame < len(control_history):
                v_prev, _ = control_history[frame-1]
                accel = (v - v_prev) / dwa_config['dt']
            else:
                accel = 0.0
            # 距离终点
            dist_to_goal = np.hypot(x - goal['x'], y - goal['y'])
            # 文本显示
            info = (
                f"Step: {frame}\n"
                f"Pos: ({x:.2f}, {y:.2f})\n"
                f"Vel: {v:.2f} m/s\n"
                f"Accel: {accel:.2f} m/s²\n"
                f"Dist to Goal: {dist_to_goal:.2f} m"
            )
            ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        ax.set_xlim(0, map_size_m)
        ax.set_ylim(0, map_size_m)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_title(f'DWA Simulation - Frame {frame}')
        ax.grid(True, alpha=0.3)
        
        return ax.get_children()
    
    anim = FuncAnimation(fig, animate, frames=len(robot_states), 
                        interval=100, repeat=True, blit=False)
    plt.show()
    return anim

# ========== 5. DWA仿真主函数 ==========
def run_dwa_simulation(grid_map, start, goal, max_iterations=1000, goal_threshold=0.15):
    """
    运行DWA仿真
    
    参数:
    - grid_map: 栅格地图
    - start: 起始位置 {'x': float, 'y': float}
    - goal: 目标位置 {'x': float, 'y': float}
    - max_iterations: 最大迭代次数
    - goal_threshold: 到达目标的阈值
    
    返回:
    - robot_states: 机器人状态历史
    - control_history: 控制输入历史
    - path: 全局路径
    """
    # 初始化DWA规划器
    dwa_planner = DWAPlanner(dwa_config)
    
    # 获取全局路径
    path = plan_path(grid_map, start, goal, smooth_path_flag=True)
    if not path:
        print("⚠️  无法找到全局路径")
        return [], [], []
    
    # 初始化机器人状态 - 设置初始朝向指向目标
    initial_theta = np.arctan2(goal['y'] - start['y'], goal['x'] - start['x'])
    robot_state = [start['x'], start['y'], initial_theta]  # [x, y, theta]
    robot_velocity = [0.0, 0.0]  # [v, omega]
    
    # 记录历史
    robot_states = [robot_state.copy()]
    control_history = []
    
    print(f"🚀 开始DWA仿真")
    print(f"   起点: ({start['x']:.2f}, {start['y']:.2f})")
    print(f"   终点: ({goal['x']:.2f}, {goal['y']:.2f})")
    print(f"   初始朝向: {initial_theta:.3f} rad")
    print(f"   全局路径点数: {len(path)}")
    print(f"   目标阈值: {goal_threshold:.3f}m")
    
    # 记录最佳距离
    best_distance = float('inf')
    stuck_counter = 0
    last_position = robot_state[:2]
    
    # 仿真循环
    for iteration in range(max_iterations):
        # 计算到目标的距离
        distance_to_goal = np.hypot(robot_state[0] - goal['x'], robot_state[1] - goal['y'])
        
        # 更新最佳距离
        if distance_to_goal < best_distance:
            best_distance = distance_to_goal
        
        # 检查是否到达目标
        if distance_to_goal < goal_threshold:
            print(f"✅ 到达目标! 迭代次数: {iteration}")
            break
        
        # 检查是否卡住（位置没有变化）
        current_position = robot_state[:2]
        position_change = np.hypot(current_position[0] - last_position[0], 
                                  current_position[1] - last_position[1])
        
        if position_change < 0.01:  # 如果位置变化很小
            stuck_counter += 1
            if stuck_counter > 50:  # 如果连续50次迭代都卡住
                print(f"⚠️  机器人可能卡住，停止仿真")
                break
        else:
            stuck_counter = 0
            last_position = current_position
        
        # DWA规划
        goal_array = [goal['x'], goal['y']]
        v, omega = dwa_planner.plan(robot_state, robot_velocity, goal_array, grid_map)
        
        # 记录控制输入
        control_history.append([v, omega])
        
        # 更新机器人状态
        robot_state = motion_model(robot_state, [v, omega], dwa_config['dt'])
        robot_velocity = [v, omega]
        
        # 记录状态
        robot_states.append(robot_state.copy())
        
        # 检查是否超出地图边界
        if (robot_state[0] < 0 or robot_state[0] > map_size_m or 
            robot_state[1] < 0 or robot_state[1] > map_size_m):
            print(f"⚠️  机器人超出地图边界，停止仿真")
            break
        
        # 每50次迭代打印一次进度
        if iteration % 50 == 0:
            print(f"   迭代 {iteration}: 距离目标 {distance_to_goal:.3f}m, 控制 [{v:.3f}, {omega:.3f}]")
    
    print(f"📊 仿真完成")
    print(f"   总迭代次数: {len(robot_states)}")
    print(f"   最终距离: {distance_to_goal:.3f}m")
    print(f"   最佳距离: {best_distance:.3f}m")
    print(f"   是否到达目标: {'是' if distance_to_goal < goal_threshold else '否'}")
    
    return robot_states, control_history, path

# ========== 6. 主程序 ==========
if __name__ == "__main__":
    # 设置起点和终点
    start = {'x': 2.0, 'y': 0.5}
    goal = {'x': 4.5, 'y': 4.5}
    
    # 检查起点是否在障碍物内
    gx = int(start['x'] / resolution)
    gy = int(start['y'] / resolution)
    print("起点格子坐标:", gx, gy, "值:", grid_map[gy, gx])
    
    # 运行DWA仿真
    robot_states, control_history, path = run_dwa_simulation(grid_map, start, goal)
    
    if robot_states:
        # 静态可视化
        plot_dwa_simulation(grid_map, start, goal, path, robot_states, 
                           robot_states, control_history, animation_mode=False)
        
        # 动画可视化（可选）
        animate_dwa_simulation(grid_map, start, goal, path, robot_states, control_history)
    else:
        print("❌ 仿真失败，无法可视化")

def is_free(pos, grid_map, resolution):
    gx = int(pos['x'] / resolution)
    gy = int(pos['y'] / resolution)
    return grid_map[gy, gx] == 0

if not is_free(start, grid_map, resolution):
    raise ValueError("起点在障碍物内，请选择空地作为起点！")
if not is_free(goal, grid_map, resolution):
    raise ValueError("终点在障碍物内，请选择空地作为终点！") 