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
from matplotlib.patches import Circle
import mplcursors

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
trajectory = [robot_pos.copy()]  # 轨迹记录列表

# ==================== 探索管理器初始化 ====================
explorer = ExplorationManager(map_resolution=MAP_RESOLUTION)  # 创建探索管理器实例


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
        ax=None, frontiers=None
):
    if ax is None:
        ax = plt.gca()
    ax.clear()
    # 只画障碍物边界线
    ax.contour(true_map, levels=[0.5], colors='k', linewidths=0.5)
    # 其余内容不变
    ax.plot([p[0] / MAP_RESOLUTION for p in trajectory], [p[1] / MAP_RESOLUTION for p in trajectory], 'g.-',
            linewidth=2)
    ax.plot(robot_pos[0] / MAP_RESOLUTION, robot_pos[1] / MAP_RESOLUTION, 'ro', markersize=8)
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
    ax.set_title('Ground Truth Map')


class SLAMMapVisualizer:
    def __init__(self, ax, slam_map, resolution, robot_radius=0.15):
        self.ax = ax
        self.resolution = resolution
        self.ax.set_title('SLAM Mapping', fontsize=16)
        self.ax.set_facecolor('white')  # 右图底色为白色
        # 不用imshow，只画contour
        self._contour_collections = []
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
        self.ax.set_xlim(0, slam_map.shape[1])
        self.ax.set_ylim(0, slam_map.shape[0])

    def update(self, slam_map, robot_pos, trajectory, scan=None, robot_theta=None, **kwargs):
        robot_x_grid = robot_pos[0] / self.resolution
        robot_y_grid = robot_pos[1] / self.resolution
        self.robot_circle.center = (robot_x_grid, robot_y_grid)
        if len(trajectory) > 1:
            path_x = [p[0] / self.resolution for p in trajectory]
            path_y = [p[1] / self.resolution for p in trajectory]
            self.path_line.set_data(path_x, path_y)
        # 移除旧contour
        try:
            for coll in getattr(self, '_contour_collections', []):
                coll.remove()
        except Exception as e:
            print("Contour remove error:", e)
        # 画新contour（只画边界线）
        try:
            contour = self.ax.contour(slam_map, levels=[0.5], colors='k', linewidths=0.5)
            self._contour_collections = getattr(contour, 'collections', [])
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


# ==================== 主探索循环 ====================
plt.ion()  # 开启交互模式，用于实时显示
fig, (ax_true, ax_slam) = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Exploration Visualization', fontsize=18)
step = 0  # 步数计数器
lidar_log = []  # 激光雷达数据日志
exploration_complete = False  # 探索完成标志
return_path = []  # 返回路径记录

print("开始自主探索...")

visualizer = SLAMMapVisualizer(ax_slam, known_map, MAP_RESOLUTION, robot_radius=ROBOT_RADIUS)

mplcursors_inited = False

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
        return_step = 0

    # 6. 探索阶段：使用A*算法规划路径
    if not exploration_complete:
        path = astar(robot_pos, target, known_map)

        # 7. 检查路径是否有效
        if path is None or len(path) < 2:
            print('无法到达目标，跳过')
            continue

        # 8. 执行移动（只走一步）
        next_pos = np.array(path[1])

        # 9. 碰撞检测
        gx, gy = world_to_grid(next_pos[0], next_pos[1], MAP_RESOLUTION)
        if true_map[gy, gx] == 1:
            print('碰到障碍，跳过')
            continue

        # 10. 更新机器人位置和轨迹
        robot_pos = next_pos
        trajectory.append(robot_pos.copy())
        step += 1

        # 11. 获取所有前沿点
        frontiers = detect_frontiers(known_map, map_resolution=MAP_RESOLUTION)

        # 2. 当前目标（最优前沿点）
        best_frontier = target  # target 就是你主循环里选出来的目标

        # 3. 当前A*路径（如果有）
        # path = ...  # 你主循环里A*算出来的路径

        # --- 并排更新两个窗口 ---
        if step % 2 == 0:
            plot_map(
                true_map, known_map, robot_pos, trajectory, target,
                scan=scan, robot_theta=robot_theta, return_path=return_path,
                ax=ax_true, frontiers=frontiers
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
        fig.canvas.draw()
        fig.canvas.flush_events()

    # 12. 返回阶段：沿返回路径移动
    else:
        if return_step < len(return_path) - 1:
            next_pos = np.array(return_path[return_step + 1])
            gx, gy = world_to_grid(next_pos[0], next_pos[1], MAP_RESOLUTION)
            if true_map[gy, gx] == 1:
                print('返回途中碰到障碍，跳过')
                return_step += 1
                continue
            robot_pos = next_pos
            trajectory.append(robot_pos.copy())  # 记录轨迹
            return_step += 1
            step += 1
            # --- 并排更新两个窗口 ---
            if step % 2 == 0:
                plot_map(
                    true_map, known_map, robot_pos, trajectory, target=None,
                    scan=scan, robot_theta=robot_theta, return_path=return_path,
                    ax=ax_true, frontiers=frontiers
                )
            frontiers = detect_frontiers(known_map, map_resolution=MAP_RESOLUTION)
            best_frontier = None
            visualizer.update(
                slam_map=known_map,
                robot_pos=robot_pos,
                trajectory=trajectory,
                scan=scan,
                robot_theta=robot_theta,
                frontiers=frontiers,
                best_frontier=best_frontier,
                path=return_path
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
        else:
            print('已成功返回起点！')
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