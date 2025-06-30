import sys, os
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# 可选：设置中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="机器人导航可视化", layout="wide")
st.title("🤖 机器人导航/SLAM/路径规划 多功能可视化系统")

# ========== 1. 地图与障碍物 ==========
with st.sidebar:
    st.header("地图参数")
    map_size = st.slider("地图尺寸 (格)", 20, 100, 50)
    map_size_m = st.slider("物理尺寸 (米)", 2, 10, 5)
    resolution = map_size_m / map_size

# 生成地图（后续要改成传入的地图）
grid_map = np.zeros((map_size, map_size), dtype=np.uint8)
grid_map[map_size//4:map_size//2, map_size//4:map_size//2] = 1
grid_map[10:15, 35:40] = 1

# ========== 2. 多机器人管理 ==========
st.sidebar.header("机器人管理")
robot_num = st.sidebar.number_input("机器人数量", 1, 5, 1)
robot_starts = []
robot_goals = []
for i in range(robot_num):
    st.sidebar.markdown(f"**机器人{i+1}**")
    sx = st.sidebar.number_input(f"起点X{i+1}", 0.0, float(map_size_m), float(0.5 + i))
    sy = st.sidebar.number_input(f"起点Y{i+1}", 0.0, float(map_size_m), float(0.5 + i))
    gx = st.sidebar.number_input(f"终点X{i+1}", 0.0, float(map_size_m), float(map_size_m-0.5-i))
    gy = st.sidebar.number_input(f"终点Y{i+1}", 0.0, float(map_size_m), float(map_size_m-0.5-i))
    robot_starts.append({'x': sx, 'y': sy})
    robot_goals.append({'x': gx, 'y': gy})

# ========== 3. 页面Tab ==========
tab1, tab2, tab3 = st.tabs(["A*路径规划", "DWA动力学仿真", "SLAM地图"])

# ========== 4. A*路径规划 ==========
with tab1:
    st.header("A* 路径规划与轨迹可视化")
    from planner.path_planner import plan_path
    fig, ax = plt.subplots()
    ax.imshow(grid_map, cmap='Greys', origin='lower', extent=(0, map_size_m, 0, map_size_m), alpha=0.3)
    colors = ['b', 'g', 'm', 'c', 'y']
    for i in range(robot_num):
        start, goal = robot_starts[i], robot_goals[i]
        path = plan_path(grid_map, start, goal, smooth_path_flag=True)
        ax.scatter([start['x']], [start['y']], c=colors[i%len(colors)], s=100, marker='o', label=f'R{i+1} Start')
        ax.scatter([goal['x']], [goal['y']], c=colors[i%len(colors)], s=100, marker='*', label=f'R{i+1} Goal')
        if path:
            px, py = zip(*path)
            ax.plot(px, py, '-', color=colors[i%len(colors)], linewidth=2, label=f'R{i+1} Path')
    ax.set_xlim(0, map_size_m)
    ax.set_ylim(0, map_size_m)
    ax.set_title("A* 路径规划结果")
    ax.legend()
    st.pyplot(fig)

# ========== 5. DWA动力学仿真 ==========
with tab2:
    st.header("DWA 动力学仿真（单机器人演示）")
    from planner.dwa_planner import DWAPlanner
    dwa_config = {
        'max_speed': 0.5,
        'min_speed': 0.0,
        'max_yawrate': 1.0,
        'max_accel': 0.5,
        'max_dyawrate': 0.5,
        'v_reso': 0.02,
        'yawrate_reso': 0.05,
        'dt': 0.1,
        'predict_time': 3.0,
        'to_goal_cost_gain': 1.0,
        'speed_cost_gain': 0.1,
        'obstacle_cost_gain': 0.5,
        'robot_radius': 0.02,
        'map_resolution': resolution
    }
    start = robot_starts[0]
    goal = robot_goals[0]
    dwa_planner = DWAPlanner(dwa_config)
    robot_state = [start['x'], start['y'], 0.0]
    robot_velocity = [0.0, 0.0]
    robot_states = [robot_state.copy()]
    velocities = [robot_velocity.copy()]
    accels = [[0.0, 0.0]]
    dists = [np.hypot(robot_state[0] - goal['x'], robot_state[1] - goal['y'])]
    for step in range(100):
        v, omega = dwa_planner.plan(robot_state, robot_velocity, [goal['x'], goal['y']], grid_map)
        new_state = [
            robot_state[0] + v * np.cos(robot_state[2]) * dwa_config['dt'],
            robot_state[1] + v * np.sin(robot_state[2]) * dwa_config['dt'],
            robot_state[2] + omega * dwa_config['dt']
        ]
        new_velocity = [v, omega]
        velocities.append(new_velocity)
        accel = [
            (new_velocity[0] - robot_velocity[0]) / dwa_config['dt'],
            (new_velocity[1] - robot_velocity[1]) / dwa_config['dt']
        ]
        accels.append(accel)
        dist = np.hypot(new_state[0] - goal['x'], new_state[1] - goal['y'])
        dists.append(dist)
        robot_state = new_state
        robot_velocity = new_velocity
        robot_states.append(robot_state.copy())
        # 实时展示
        st.write(f"**Step {step+1}**: 坐标=({robot_state[0]:.2f}, {robot_state[1]:.2f}), 速度=({v:.2f}, {omega:.2f}), 加速度=({accel[0]:.2f}, {accel[1]:.2f}), 距离终点={dist:.2f}m")
        if dist < 0.15:
            break
    fig2, ax2 = plt.subplots()
    ax2.imshow(grid_map, cmap='Greys', origin='lower', extent=(0, map_size_m, 0, map_size_m), alpha=0.3)
    ax2.scatter([start['x']], [start['y']], c='g', s=100, marker='o', label='Start')
    ax2.scatter([goal['x']], [goal['y']], c='r', s=100, marker='*', label='Goal')
    traj_x = [s[0] for s in robot_states]
    traj_y = [s[1] for s in robot_states]
    ax2.plot(traj_x, traj_y, 'm-', linewidth=2, label='DWA Trajectory')
    # 标注当前状态
    ax2.scatter([robot_states[-1][0]], [robot_states[-1][1]], c='b', s=80, marker='x', label='Current')
    ax2.set_xlim(0, map_size_m)
    ax2.set_ylim(0, map_size_m)
    ax2.set_title("DWA仿真轨迹")
    ax2.legend()
    st.pyplot(fig2)
    # 展示最后一步的详细信息
    st.success(f"最终位置: ({robot_states[-1][0]:.2f}, {robot_states[-1][1]:.2f}), 速度: {velocities[-1][0]:.2f}, {velocities[-1][1]:.2f}, 加速度: {accels[-1][0]:.2f}, {accels[-1][1]:.2f}, 距离终点: {dists[-1]:.2f}m")

# ========== 6. SLAM地图（占位） ==========
with tab3:
    st.header("SLAM 地图展示（可扩展）")
    st.info("此处可集成SLAM模块输出的地图、轨迹、粒子云等。")

# ========== 7. 上传/下载 ==========
st.sidebar.header("数据上传/下载")
uploaded = st.sidebar.file_uploader("上传地图/轨迹文件")
if uploaded:
    st.sidebar.success("文件已上传！")
st.sidebar.button("下载仿真结果")

st.sidebar.info("本页面为多功能演示模板，可根据需要扩展DWA仿真、SLAM地图、机器人状态等内容。")
