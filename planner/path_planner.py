import numpy as np
import sys
import os

# 添加PythonRobotics库路径
current_dir = os.path.dirname(os.path.abspath(__file__))
pythonrobotics_path = os.path.join(current_dir, '..', 'PythonRobotics', 'PathPlanning', 'AStar')
sys.path.insert(0, pythonrobotics_path)

# 导入全局参数
from config.settings import MAP_SIZE_M, MAP_RESOLUTION, ROBOT_RADIUS
from config.map import MAP_SIZE

try:
    # 动态导入PythonRobotics模块
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "a_star",
        os.path.join(pythonrobotics_path, "a_star.py")
    )
    if spec is not None and spec.loader is not None:
        a_star_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(a_star_module)

        AStarPlanner = a_star_module.AStarPlanner
        PYTHONROBOTICS_AVAILABLE = True
    else:
        raise ImportError("无法加载PythonRobotics模块")
except Exception as e:
    print(f"警告: 无法导入PythonRobotics A*算法: {e}")
    PYTHONROBOTICS_AVAILABLE = False


def validate_path(path, grid_map, map_resolution=MAP_RESOLUTION):
    """
    验证路径是否有效（不穿过障碍物）

    参数:
    - path: [(x1, y1), (x2, y2), ...] 路径点列表
    - grid_map: 2D numpy数组，地图栅格
    - map_resolution: 地图分辨率

    返回:
    - is_valid: 路径是否有效
    """
    if len(path) < 2:
        return True

    try:
        h, w = grid_map.shape
        map_size_m = MAP_SIZE_M
        cell_size = map_size_m / max(h, w)

        for i in range(len(path) - 1):
            # 检查路径段
            x1, y1 = path[i]
            x2, y2 = path[i + 1]

            # 在路径段上采样多个点进行检查
            num_samples = max(5, int(np.hypot(x2 - x1, y2 - y1) / cell_size))

            for j in range(num_samples + 1):
                t = j / num_samples
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)

                # 转换为格子坐标
                gx = int(x / cell_size)
                gy = int(y / cell_size)

                # 检查边界
                if gx < 0 or gx >= w or gy < 0 or gy >= h:
                    print(f"⚠️  路径点 ({x:.2f}, {y:.2f}) 超出地图边界")
                    return False

                # 检查是否在障碍物内
                if grid_map[gy, gx] == 1:
                    print(f"⚠️  路径点 ({x:.2f}, {y:.2f}) 在障碍物内")
                    return False

        return True

    except Exception as e:
        print(f"路径验证错误: {e}")
        return False


def smooth_path(path, smoothing_factor=0.1, grid_map=None):
    """
    使用样条曲线平滑路径

    参数:
    - path: [(x1, y1), (x2, y2), ...] 原始路径
    - smoothing_factor: 平滑因子，0-1之间
    - grid_map: 地图栅格，用于验证平滑后的路径

    返回:
    - smoothed_path: 平滑后的路径
    """
    if len(path) < 3:
        return path

    try:
        # 提取x和y坐标
        x_coords = [p[0] for p in path]
        y_coords = [p[1] for p in path]

        # 简单的移动平均平滑
        smoothed_x = []
        smoothed_y = []

        for i in range(len(path)):
            if i == 0 or i == len(path) - 1:
                # 保持起点和终点不变
                smoothed_x.append(x_coords[i])
                smoothed_y.append(y_coords[i])
            else:
                # 对中间点进行平滑
                prev_x = x_coords[i - 1]
                curr_x = x_coords[i]
                next_x = x_coords[i + 1]

                prev_y = y_coords[i - 1]
                curr_y = y_coords[i]
                next_y = y_coords[i + 1]

                # 加权平均
                smooth_x = (1 - smoothing_factor) * curr_x + (smoothing_factor / 2) * (prev_x + next_x)
                smooth_y = (1 - smoothing_factor) * curr_y + (smoothing_factor / 2) * (prev_y + next_y)

                smoothed_x.append(smooth_x)
                smoothed_y.append(smooth_y)

        smoothed_path = list(zip(smoothed_x, smoothed_y))

        # 如果提供了地图，验证平滑后的路径
        if grid_map is not None and not validate_path(smoothed_path, grid_map):
            print("⚠️  平滑后的路径无效，返回原始路径")
            return path

        return smoothed_path

    except Exception as e:
        print(f"路径平滑错误: {e}")
        return path


def smooth_path_with_obstacle_avoidance(path, grid_map, resolution, initial_smoothing=0.2, min_smoothing=0.01, max_iter=20, verbose=True):
    """
    带障碍物约束的路径平滑（弹性带思想+自动调节平滑因子）
    - 若平滑后路径点落在障碍物内，则投影到最近的自由格子
    - 自动减小平滑因子，直到找到有效平滑路径或达到最小平滑度

    参数：
        path: [(x, y), ...] 原始路径
        grid_map: 2D numpy数组，0为自由，1为障碍
        resolution: 地图分辨率
        initial_smoothing: 初始平滑因子
        min_smoothing: 最小平滑因子
        max_iter: 最大迭代次数
        verbose: 是否打印调试信息
    返回：
        smoothed_path: 平滑且避障的路径
    """
    from scipy.ndimage import distance_transform_edt

    if len(path) < 3:
        return path

    h, w = grid_map.shape
    # 计算障碍物距离场（每个格子到最近障碍物的距离，单位：格）
    obstacle_mask = (grid_map == 1)
    dist_field = distance_transform_edt(~obstacle_mask) * resolution

    smoothing = initial_smoothing
    for attempt in range(max_iter):
        smoothed_x = []
        smoothed_y = []
        for i in range(len(path)):
            if i == 0 or i == len(path) - 1:
                smoothed_x.append(path[i][0])
                smoothed_y.append(path[i][1])
            else:
                prev_x, prev_y = path[i - 1]
                curr_x, curr_y = path[i]
                next_x, next_y = path[i + 1]
                # 加权平均
                smooth_x = (1 - smoothing) * curr_x + (smoothing / 2) * (prev_x + next_x)
                smooth_y = (1 - smoothing) * curr_y + (smoothing / 2) * (prev_y + next_y)
                # 投影到最近自由格
                gx = int(smooth_x / resolution)
                gy = int(smooth_y / resolution)
                if 0 <= gx < w and 0 <= gy < h and grid_map[gy, gx] == 0:
                    smoothed_x.append(smooth_x)
                    smoothed_y.append(smooth_y)
                else:
                    # 投影到最近自由格
                    # 找到最近的自由格子
                    free_gy, free_gx = np.unravel_index(np.argmax(dist_field), dist_field.shape)
                    # 也可以用最近的自由点
                    min_dist = float('inf')
                    best_x, best_y = curr_x, curr_y
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            nx, ny = gx + dx, gy + dy
                            if 0 <= nx < w and 0 <= ny < h and grid_map[ny, nx] == 0:
                                dist = (nx * resolution - smooth_x) ** 2 + (ny * resolution - smooth_y) ** 2
                                if dist < min_dist:
                                    min_dist = dist
                                    best_x = nx * resolution + resolution / 2
                                    best_y = ny * resolution + resolution / 2
                    smoothed_x.append(best_x)
                    smoothed_y.append(best_y)
        smoothed_path = list(zip(smoothed_x, smoothed_y))
        # 检查平滑路径有效性
        if validate_path(smoothed_path, grid_map, resolution):
            if verbose:
                print(f"✅ 避障平滑成功，平滑因子: {smoothing:.3f}, 尝试次数: {attempt+1}")
            return smoothed_path
        else:
            if verbose:
                print(f"⚠️  平滑因子 {smoothing:.3f} 下路径无效，自动减小平滑度...")
            smoothing = max(smoothing * 0.5, min_smoothing)
    if verbose:
        print("❌ 避障平滑失败，返回原始路径")
    return path


def plan_path(grid_map, start_pos, goal_pos, smooth_path_flag=True):
    """
    使用PythonRobotics库的A*算法进行路径规划

    参数:
    - grid_map: 2D numpy数组，地图栅格，0为自由，1为障碍
    - start_pos: dict {'x': float, 'y': float}，起始位置（米）
    - goal_pos: dict {'x': float, 'y': float}，目标位置（米）
    - smooth_path_flag: 是否平滑路径

    返回:
    - path: [(x1, y1), (x2, y2), ...] 路径点列表（米），无路径时返回空列表
    """
    if PYTHONROBOTICS_AVAILABLE:
        path = plan_path_pythonrobotics(grid_map, start_pos, goal_pos)
    else:
        path = plan_path_simple(grid_map, start_pos, goal_pos, MAP_RESOLUTION)

    # 如果找到路径且需要平滑
    if path and smooth_path_flag and len(path) > 2:
        original_length = len(path)
        path = smooth_path(path, grid_map=grid_map)
        print(f"✅ 路径已平滑: {original_length} -> {len(path)} 个点")

    return path


def plan_path_pythonrobotics(grid_map, start_pos, goal_pos):
    """使用PythonRobotics库的A*算法"""
    try:
        # 获取地图尺寸
        h, w = grid_map.shape
        map_size_m = MAP_SIZE_M  # 地图实际大小（米）
        resolution = MAP_RESOLUTION  # 使用全局分辨率

        # 提取障碍物坐标 - 修复坐标转换
        obstacle_indices = np.where(grid_map == 1)
        if len(obstacle_indices[0]) > 0:
            # 正确的坐标转换：x对应列，y对应行
            ox = obstacle_indices[1] * resolution  # x坐标（列）
            oy = obstacle_indices[0] * resolution  # y坐标（行）
        else:
            # 如果没有障碍物，创建一些边界障碍物
            border_points = []
            for i in range(w):
                border_points.extend([(i * resolution, 0), (i * resolution, (h - 1) * resolution)])
            for j in range(h):
                border_points.extend([(0, j * resolution), ((w - 1) * resolution, j * resolution)])
            ox = np.array([p[0] for p in border_points])
            oy = np.array([p[1] for p in border_points])

        # 扩展地图边界以包含起点和终点
        min_x = min(np.min(ox) if len(ox) > 0 else 0, start_pos['x'], goal_pos['x']) - 0.5
        max_x = max(np.max(ox) if len(ox) > 0 else map_size_m, start_pos['x'], goal_pos['x']) + 0.5
        min_y = min(np.min(oy) if len(oy) > 0 else 0, start_pos['y'], goal_pos['y']) - 0.5
        max_y = max(np.max(oy) if len(oy) > 0 else map_size_m, start_pos['y'], goal_pos['y']) + 0.5

        # 创建A*规划器
        a_star = AStarPlanner(ox, oy, resolution, ROBOT_RADIUS)  # 使用全局机器人半径

        # 规划路径
        rx, ry = a_star.planning(start_pos['x'], start_pos['y'], goal_pos['x'], goal_pos['y'])

        if rx is None or len(rx) == 0:
            print("❌ A*算法未找到路径")
            return []

        # 转换为路径格式
        path = list(zip(rx, ry))
        print(f"✅ A*算法找到路径: {len(path)} 个点")
        return path

    except Exception as e:
        print(f"❌ A*算法错误: {e}")
        return []


def plan_path_simple(grid_map, start_pos, goal_pos, map_resolution=MAP_RESOLUTION):
    """
    简单的A*路径规划实现
    """
    def heuristic(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def distance(a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

    def get_neighbors(pos, w, h):
        x, y = pos
        neighbors = []
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                neighbors.append((nx, ny))
        return neighbors

    def point_to_grid(p):
        return (int(p[0] / map_resolution), int(p[1] / map_resolution))

    def grid_to_point(gx, gy):
        return (gx * map_resolution + map_resolution / 2, gy * map_resolution + map_resolution / 2)

    # 转换为格子坐标
    start_grid = point_to_grid((start_pos['x'], start_pos['y']))
    goal_grid = point_to_grid((goal_pos['x'], goal_pos['y']))

    h, w = grid_map.shape

    # 检查起点和终点是否在障碍物内
    if grid_map[start_grid[1], start_grid[0]] == 1:
        print(f"❌ 起点 ({start_pos['x']:.2f}, {start_pos['y']:.2f}) 在障碍物内")
        return []
    if grid_map[goal_grid[1], goal_grid[0]] == 1:
        print(f"❌ 终点 ({goal_pos['x']:.2f}, {goal_pos['y']:.2f}) 在障碍物内")
        return []

    # A*算法
    open_set = {start_grid}
    closed_set = set()
    came_from = {}
    g_score = {start_grid: 0}
    f_score = {start_grid: heuristic(start_grid, goal_grid)}

    while open_set:
        current = min(open_set, key=lambda x: f_score.get(x, float('inf')))

        if current == goal_grid:
            # 重建路径
            path = []
            while current in came_from:
                path.append(grid_to_point(current[0], current[1]))
                current = came_from[current]
            path.append(grid_to_point(start_grid[0], start_grid[1]))
            path.reverse()
            return path

        open_set.remove(current)
        closed_set.add(current)

        for neighbor in get_neighbors(current, w, h):
            if neighbor in closed_set or grid_map[neighbor[1], neighbor[0]] == 1:
                continue

            tentative_g_score = g_score[current] + distance(current, neighbor)

            if neighbor not in open_set:
                open_set.add(neighbor)
            elif tentative_g_score >= g_score.get(neighbor, float('inf')):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g_score
            f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal_grid)

    print("❌ 未找到路径")
    return []


def plan_path_with_validation(grid_map, start_pos, goal_pos, smooth_path_flag=True):
    """
    带验证的路径规划
    """
    # 规划路径
    path = plan_path(grid_map, start_pos, goal_pos, smooth_path_flag)

    if path:
        # 验证路径
        if validate_path(path, grid_map):
            print("✅ 路径验证通过")
            return path
        else:
            print("❌ 路径验证失败")
            return []
    else:
        print("❌ 路径规划失败")
        return []


if __name__ == "__main__":
    # 测试路径规划
    from config.map import get_global_map
    from config.settings import START_POSITION, EXIT_POSITION

    # 获取地图
    grid_map = get_global_map()
    print(f"地图尺寸: {grid_map.shape}")

    # 测试路径规划
    path = plan_path_with_validation(grid_map, START_POSITION, EXIT_POSITION)
    if path:
        print(f"找到路径: {len(path)} 个点")
        print(f"路径: {path[:5]}...{path[-5:] if len(path) > 10 else path}")
    else:
        print("未找到路径")