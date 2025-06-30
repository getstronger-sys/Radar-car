import numpy as np
import sys
import os

# 添加PythonRobotics库路径
current_dir = os.path.dirname(os.path.abspath(__file__))
pythonrobotics_path = os.path.join(current_dir, '..', 'PythonRobotics', 'PathPlanning', 'AStar')
sys.path.insert(0, pythonrobotics_path)

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

def validate_path(path, grid_map, map_resolution=0.1):
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
        map_size_m = 5.0
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
                prev_x = x_coords[i-1]
                curr_x = x_coords[i]
                next_x = x_coords[i+1]
                
                prev_y = y_coords[i-1]
                curr_y = y_coords[i]
                next_y = y_coords[i+1]
                
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
        path = plan_path_simple(grid_map, start_pos, goal_pos)
    
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
        map_size_m = 5.0  # 地图实际大小（米）
        resolution = map_size_m / max(h, w)  # 计算分辨率
        
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
                border_points.extend([(i * resolution, 0), (i * resolution, (h-1) * resolution)])
            for j in range(h):
                border_points.extend([(0, j * resolution), ((w-1) * resolution, j * resolution)])
            ox = np.array([p[0] for p in border_points])
            oy = np.array([p[1] for p in border_points])
        
        # 扩展地图边界以包含起点和终点
        min_x = min(np.min(ox) if len(ox) > 0 else 0, start_pos['x'], goal_pos['x']) - 0.5
        max_x = max(np.max(ox) if len(ox) > 0 else map_size_m, start_pos['x'], goal_pos['x']) + 0.5
        min_y = min(np.min(oy) if len(oy) > 0 else 0, start_pos['y'], goal_pos['y']) - 0.5
        max_y = max(np.max(oy) if len(oy) > 0 else map_size_m, start_pos['y'], goal_pos['y']) + 0.5
        
        # 添加边界障碍物以确保地图完整性
        border_ox = []
        border_oy = []
        
        # 添加扩展后的边界障碍物
        for x in np.arange(min_x, max_x + resolution, resolution):
            border_ox.extend([x, x])
            border_oy.extend([min_y, max_y])
        for y in np.arange(min_y, max_y + resolution, resolution):
            border_ox.extend([min_x, max_x])
            border_oy.extend([y, y])
        
        # 合并原始障碍物和边界障碍物
        all_ox = np.concatenate([ox, border_ox])
        all_oy = np.concatenate([oy, border_oy])
        
        print(f"🔍 障碍物信息: {len(all_ox)} 个障碍物点")
        print(f"   地图尺寸: {w}x{h}, 分辨率: {resolution:.3f}m")
        print(f"   地图边界: x[{min_x:.2f}, {max_x:.2f}], y[{min_y:.2f}, {max_y:.2f}]")
        print(f"   起点: ({start_pos['x']:.2f}, {start_pos['y']:.2f})")
        print(f"   终点: ({goal_pos['x']:.2f}, {goal_pos['y']:.2f})")
        
        # 初始化A*规划器
        a_star = AStarPlanner(all_ox, all_oy, resolution, 0.1)  # 机器人半径0.1米
        
        # 执行路径规划
        rx, ry = a_star.planning(start_pos['x'], start_pos['y'], 
                                goal_pos['x'], goal_pos['y'])
        
        # 转换路径格式并确保正确的方向
        if len(rx) > 0:
            path = list(zip(rx, ry))
            
            # 检查路径方向，确保起点和终点正确
            if len(path) > 1:
                start_dist = np.hypot(path[0][0] - start_pos['x'], path[0][1] - start_pos['y'])
                end_dist = np.hypot(path[-1][0] - goal_pos['x'], path[-1][1] - goal_pos['y'])
                
                # 如果路径方向反了，反转路径
                if start_dist > end_dist:
                    path = path[::-1]
                    print(f"⚠️  路径方向已修正: 起点 {path[0]} -> 终点 {path[-1]}")
            
            print(f"✅ 找到路径: {len(path)} 个点")
            return path
        else:
            print(f"⚠️  未找到路径: 起点({start_pos['x']:.2f}, {start_pos['y']:.2f}) -> 终点({goal_pos['x']:.2f}, {goal_pos['y']:.2f})")
            return []
            
    except Exception as e:
        print(f"PythonRobotics A*路径规划错误: {e}")
        return []

def plan_path_simple(grid_map, start_pos, goal_pos):
    """简单的A*路径规划（备用方案）"""
    import heapq
    
    def heuristic(a, b):
        """启发函数：欧氏距离"""
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

    def distance(a, b):
        """计算两点间实际代价"""
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        if dx == 1 and dy == 1:
            return 1.414  # √2
        else:
            return 1.0

    def get_neighbors(pos, w, h):
        """获取8邻域邻居"""
        x, y = pos
        neighbors = []
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h:
                neighbors.append((nx, ny))
        return neighbors

    def astar_path(start, goal, occupancy_grid):
        """A*路径规划主函数"""
        grid = occupancy_grid
        h, w = grid.shape
        map_size_m = 5.0
        cell_size = map_size_m / h

        def point_to_grid(p):
            """将实际米坐标转换为格子索引"""
            gx = int(p['x'] / cell_size)
            gy = int(p['y'] / cell_size)
            gx = max(0, min(gx, w - 1))
            gy = max(0, min(gy, h - 1))
            return gx, gy

        def grid_to_point(gx, gy):
            """将格子索引转换为实际米坐标"""
            x = gx * cell_size + cell_size / 2
            y = gy * cell_size + cell_size / 2
            return (x, y)

        start_g = point_to_grid(start)
        goal_g = point_to_grid(goal)

        # 检查起点和终点
        if grid[start_g[1], start_g[0]] > 0.5 or grid[goal_g[1], goal_g[0]] > 0.5:
            return []

        # A*算法
        open_set = []
        heapq.heappush(open_set, (0 + heuristic(start_g, goal_g), 0, start_g))
        came_from = {}
        cost_so_far = {start_g: 0}
        closed_set = set()

        while open_set:
            _, cost, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            closed_set.add(current)

            if current == goal_g:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return [grid_to_point(x, y) for (x, y) in path]

            for n in get_neighbors(current, w, h):
                if n in closed_set or grid[n[1], n[0]] > 0.5:
                    continue
                    
                new_cost = cost + distance(current, n)
                
                if n not in cost_so_far or new_cost < cost_so_far[n]:
                    cost_so_far[n] = new_cost
                    priority = new_cost + heuristic(n, goal_g)
                    heapq.heappush(open_set, (priority, new_cost, n))
                    came_from[n] = current

        return []

    return astar_path(start_pos, goal_pos, grid_map)