import numpy as np
from scipy.ndimage import label
from scipy.ndimage import binary_dilation


def detect_frontiers(occupancy_grid, unknown_val=-1, free_threshold=0.2, map_resolution=0.01):
    """
    前沿检测函数

    参数：
    - occupancy_grid: 2D numpy数组，栅格地图，取值范围通常：
        已知空闲 < free_threshold (如 0.2)
        已知障碍 > 0.5
        未知区域通常为 -1（或其他负值）
    - unknown_val: 未知区域的标记值，默认-1
    - free_threshold: 空闲阈值，默认0.2，低于此认为空闲
    - map_resolution: 地图分辨率，单位米/格子

    返回：
    - frontiers_world: list of (x, y) 坐标列表，单位米，前沿目标点（格子中心坐标）
    """

    h, w = occupancy_grid.shape

    # 创建掩码：
    # 空闲掩码：占用概率 < free_threshold
    free_mask = (occupancy_grid >= 0) & (occupancy_grid < free_threshold)

    # 未知掩码
    unknown_mask = (occupancy_grid == unknown_val)

    # 前沿定义：空闲格子邻接至少一个未知格子
    frontiers_mask = np.zeros_like(occupancy_grid, dtype=bool)

    # 8邻域方向偏移
    neighbors_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if free_mask[y, x]:
                # 检查8邻域是否有未知格子
                for dx, dy in neighbors_offsets:
                    nx, ny = x + dx, y + dy
                    if unknown_mask[ny, nx]:
                        frontiers_mask[y, x] = True
                        break

    # 连通区域标记，得到前沿点聚类（方便后续选目标）
    labeled, num_features = label(frontiers_mask)  # type: ignore

    frontiers_world = []

    for i in range(1, num_features + 1):
        # 找出该连通区域的所有前沿格子索引
        ys, xs = np.where(labeled == i)

        # 计算该前沿簇的质心（平均格子坐标）
        cx = int(np.mean(xs))
        cy = int(np.mean(ys))

        # 转换成地图实际米坐标，格子中心点坐标
        x_m = cx * map_resolution + map_resolution / 2
        y_m = cy * map_resolution + map_resolution / 2

        frontiers_world.append((x_m, y_m))

    return frontiers_world


def select_closest_frontier(frontiers, robot_pos):
    """
    从前沿点列表中选择距离机器人当前位置最近的前沿点

    参数：
    - frontiers: list of (x, y) 坐标，单位米
    - robot_pos: (x, y) 机器人当前位置，单位米

    返回：
    - (x, y) 最近前沿点坐标，若无前沿返回None
    """
    if not frontiers:
        return None

    dists = [np.hypot(f[0] - robot_pos[0], f[1] - robot_pos[1]) for f in frontiers]
    min_index = np.argmin(dists)
    return frontiers[min_index]


def select_best_frontier_for_exploration(frontiers, robot_pos, occupancy_grid, map_resolution):
    """
    为探索任务选择最佳前沿点（综合考虑距离、信息增益和探索效率）
    
    参数：
    - frontiers: list of (x, y) 坐标，单位米
    - robot_pos: (x, y) 机器人当前位置，单位米
    - occupancy_grid: 占用栅格地图
    - map_resolution: 地图分辨率，米/格子
    
    返回：
    - (x, y) 最佳前沿点坐标，若无前沿返回None
    """
    if not frontiers:
        return None
    
    best_frontier = None
    best_score = float('inf')
    
    for frontier in frontiers:
        # 1. 距离代价（越小越好）
        distance = np.hypot(frontier[0] - robot_pos[0], frontier[1] - robot_pos[1])
        distance_cost = distance / 10.0  # 归一化到0-1范围
        
        # 2. 信息增益（周围未知区域大小）
        info_gain = calculate_frontier_information_gain(frontier, occupancy_grid, map_resolution)
        info_cost = 1.0 - (info_gain / 500.0)  # 归一化，信息增益越大越好
        
        # 3. 前沿点质量（前沿点周围空闲区域大小）
        frontier_quality = calculate_frontier_quality(frontier, occupancy_grid, map_resolution)
        quality_cost = 1.0 - (frontier_quality / 100.0)  # 归一化，质量越高越好
        
        # 4. 综合评分（加权平均）
        # 距离权重0.4，信息增益权重0.4，质量权重0.2
        total_score = (0.4 * distance_cost + 
                      0.4 * info_cost + 
                      0.2 * quality_cost)
        
        if total_score < best_score:
            best_score = total_score
            best_frontier = frontier
    
    return best_frontier


def calculate_frontier_information_gain(frontier_pos, occupancy_grid, map_resolution):
    """
    计算前沿点的信息增益（周围未知区域大小）
    
    参数：
    - frontier_pos: (x, y) 前沿点坐标（米）
    - occupancy_grid: 占用栅格地图
    - map_resolution: 地图分辨率
    
    返回：
    - information_gain: 信息增益值
    """
    # 转换到格子坐标
    gx, gy = world_to_grid(frontier_pos[0], frontier_pos[1], map_resolution)
    
    # 计算该前沿点周围未知区域的大小
    unknown_count = 0
    radius = 15  # 搜索半径（格子数）
    
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = gx + dx, gy + dy
            if (0 <= nx < occupancy_grid.shape[1] and
                0 <= ny < occupancy_grid.shape[0] and
                occupancy_grid[ny, nx] == -1):  # 未知区域
                unknown_count += 1
    
    return unknown_count


def calculate_frontier_quality(frontier_pos, occupancy_grid, map_resolution):
    """
    计算前沿点质量（周围空闲区域大小）
    
    参数：
    - frontier_pos: (x, y) 前沿点坐标（米）
    - occupancy_grid: 占用栅格地图
    - map_resolution: 地图分辨率
    
    返回：
    - quality: 前沿点质量值
    """
    # 转换到格子坐标
    gx, gy = world_to_grid(frontier_pos[0], frontier_pos[1], map_resolution)
    
    # 计算该前沿点周围空闲区域的大小
    free_count = 0
    radius = 8  # 搜索半径（格子数）
    
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = gx + dx, gy + dy
            if (0 <= nx < occupancy_grid.shape[1] and
                0 <= ny < occupancy_grid.shape[0] and
                occupancy_grid[ny, nx] == 0):  # 空闲区域
                free_count += 1
    
    return free_count


def select_frontier_with_obstacle_avoidance(frontiers, robot_pos, occupancy_grid, map_resolution):
    """
    选择考虑障碍物避障的前沿点
    
    参数：
    - frontiers: list of (x, y) 坐标，单位米
    - robot_pos: (x, y) 机器人当前位置，单位米
    - occupancy_grid: 占用栅格地图
    - map_resolution: 地图分辨率
    
    返回：
    - (x, y) 最佳前沿点坐标，若无前沿返回None
    """
    if not frontiers:
        return None
    
    valid_frontiers = []
    
    for frontier in frontiers:
        # 检查从机器人到前沿点的路径是否安全
        if is_path_safe(robot_pos, frontier, occupancy_grid, map_resolution):
            valid_frontiers.append(frontier)
    
    if not valid_frontiers:
        # 如果没有安全的前沿点，返回最近的前沿点
        return select_closest_frontier(frontiers, robot_pos)
    
    # 从安全的前沿点中选择最佳的一个
    return select_best_frontier_for_exploration(valid_frontiers, robot_pos, occupancy_grid, map_resolution)


def is_path_safe(start_pos, end_pos, occupancy_grid, map_resolution, safety_margin=0.5):
    """
    检查从起点到终点的路径是否安全
    
    参数：
    - start_pos: (x, y) 起点坐标（米）
    - end_pos: (x, y) 终点坐标（米）
    - occupancy_grid: 占用栅格地图
    - map_resolution: 地图分辨率
    - safety_margin: 安全边距（米）
    
    返回：
    - bool: 路径是否安全
    """
    # 计算路径长度
    path_length = np.hypot(end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])
    
    # 路径采样点数量
    num_samples = max(10, int(path_length / map_resolution))
    
    for i in range(num_samples + 1):
        # 插值计算路径上的点
        t = i / num_samples
        x = start_pos[0] + t * (end_pos[0] - start_pos[0])
        y = start_pos[1] + t * (end_pos[1] - start_pos[1])
        
        # 转换为格子坐标
        gx, gy = world_to_grid(x, y, map_resolution)
        
        # 检查边界
        if (gx < 0 or gx >= occupancy_grid.shape[1] or 
            gy < 0 or gy >= occupancy_grid.shape[0]):
            return False
        
        # 检查障碍物（包括安全边距）
        margin_cells = int(safety_margin / map_resolution)
        for dy in range(-margin_cells, margin_cells + 1):
            for dx in range(-margin_cells, margin_cells + 1):
                nx, ny = gx + dx, gy + dy
                if (0 <= nx < occupancy_grid.shape[1] and
                    0 <= ny < occupancy_grid.shape[0] and
                    occupancy_grid[ny, nx] == 1):  # 障碍物
                    return False
    
    return True


def calculate_information_gain(frontier_pos, occupancy_grid, map_resolution, unknown_val=-1):
    """
    计算前沿点的信息增益

    参数：
    - frontier_pos: (x, y) 前沿点坐标（米）
    - occupancy_grid: 占用栅格地图
    - map_resolution: 地图分辨率
    - unknown_val: 未知区域标记值

    返回：
    - information_gain: 信息增益值
    """
    # 转换到格子坐标
    gx, gy = world_to_grid(frontier_pos[0], frontier_pos[1], map_resolution)

    # 计算该前沿点周围未知区域的大小
    unknown_count = 0
    radius = 10  # 搜索半径（格子数）

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = gx + dx, gy + dy
            if (0 <= nx < occupancy_grid.shape[1] and
                    0 <= ny < occupancy_grid.shape[0] and
                    occupancy_grid[ny, nx] == unknown_val):
                unknown_count += 1

    return unknown_count


def select_best_frontier(frontiers, robot_pos, occupancy_grid, map_resolution,
                         distance_weight=0.7, info_weight=0.3):
    """
    选择最佳前沿点（综合考虑距离和信息增益）

    参数：
    - frontiers: 前沿点列表
    - robot_pos: 机器人位置
    - occupancy_grid: 占用栅格地图
    - map_resolution: 地图分辨率
    - distance_weight: 距离权重
    - info_weight: 信息增益权重

    返回：
    - 最佳前沿点坐标
    """
    if not frontiers:
        return None

    scores = []
    for frontier in frontiers:
        # 计算距离代价（越小越好）
        distance = np.hypot(frontier[0] - robot_pos[0], frontier[1] - robot_pos[1])
        distance_cost = distance / 10.0  # 归一化到0-1

        # 计算信息增益（越大越好）
        info_gain = calculate_information_gain(frontier, occupancy_grid, map_resolution)
        info_cost = 1.0 - (info_gain / 1000.0)  # 归一化到0-1

        # 综合评分（越小越好）
        score = distance_weight * distance_cost + info_weight * info_cost
        scores.append(score)

    min_index = np.argmin(scores)
    return frontiers[min_index]


def world_to_grid(x, y, map_resolution):
    """
    世界坐标转换为格子坐标
    """
    gx = int(x / map_resolution)
    gy = int(y / map_resolution)
    return gx, gy


def grid_to_world(gx, gy, map_resolution):
    """
    格子坐标转换为世界坐标（格子中心）
    """
    x = gx * map_resolution + map_resolution / 2
    y = gy * map_resolution + map_resolution / 2
    return x, y


def detect_frontiers_optimized(occupancy_grid, unknown_val=-1, free_threshold=0.2, map_resolution=0.01):
    """
    优化的前沿检测函数（使用向量化操作）
    """
    h, w = occupancy_grid.shape

    # 创建掩码
    free_mask = (occupancy_grid >= 0) & (occupancy_grid < free_threshold)
    unknown_mask = (occupancy_grid == unknown_val)

    # 使用卷积检测前沿（更高效）
    # 膨胀未知区域
    dilated_unknown = binary_dilation(unknown_mask, structure=np.ones((3, 3)))

    # 前沿 = 空闲区域 ∩ 膨胀的未知区域
    frontiers_mask = free_mask & dilated_unknown

    # 连通区域标记
    labeled, num_features = label(frontiers_mask)  # type: ignore

    frontiers_world = []

    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)

        if len(xs) > 0:  # 确保有前沿点
            # 计算质心
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))

            # 转换为世界坐标
            x_m, y_m = grid_to_world(cx, cy, map_resolution)
            frontiers_world.append((x_m, y_m))

    return frontiers_world


def is_exploration_complete(occupancy_grid, unknown_val=-1, min_unknown_ratio=0.05):
    """
    检测探索是否完成

    参数：
    - occupancy_grid: 占用栅格地图
    - unknown_val: 未知区域标记值
    - min_unknown_ratio: 最小未知区域比例阈值

    返回：
    - bool: 探索是否完成
    """
    total_cells = occupancy_grid.size
    unknown_cells = np.sum(occupancy_grid == unknown_val)
    unknown_ratio = unknown_cells / total_cells

    return unknown_ratio < min_unknown_ratio


class ExplorationManager:
    def __init__(self, map_resolution=0.01):
        self.map_resolution = map_resolution
        self.explored_frontiers = set()

    def get_next_target(self, occupancy_grid, robot_pos):
        """
        获取下一个探索目标

        参数：
        - occupancy_grid: 当前占用栅格地图
        - robot_pos: 机器人当前位置

        返回：
        - target_pos: 下一个目标位置，None表示探索完成
        """
        # 检测前沿
        frontiers = detect_frontiers_optimized(
            occupancy_grid,
            map_resolution=self.map_resolution
        )

        if not frontiers:
            return None

        # 选择最佳前沿（综合考虑距离和信息增益）
        best_frontier = select_best_frontier(
            frontiers, robot_pos, occupancy_grid, self.map_resolution
        )

        return best_frontier

