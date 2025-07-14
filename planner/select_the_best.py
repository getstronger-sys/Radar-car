import numpy as np
from planner.path_planner import validate_path
from scipy.ndimage import distance_transform_edt
from config.settings import MAP_RESOLUTION

def path_length(path):
    """
    计算路径总长度
    """
    if len(path) < 2:
        return 0.0
    return sum(np.linalg.norm(np.array(path[i]) - np.array(path[i-1])) for i in range(1, len(path)))

def path_smoothness(path):
    """
    计算路径平滑度，值越小越平滑（角度变化总和）
    """
    if len(path) < 3:
        return 0.0
    angles = []
    for i in range(1, len(path)-1):
        v1 = np.array(path[i]) - np.array(path[i-1])
        v2 = np.array(path[i+1]) - np.array(path[i])
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            continue
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)
        angle = np.arccos(cos_angle)
        angles.append(angle)
    return np.sum(np.abs(angles))

def path_safety(path, grid_map):
    """
    计算路径安全性，值越大越安全（路径点到障碍物的最小距离均值）
    """
    obstacle_mask = (grid_map == 1)
    dist_field = distance_transform_edt(~obstacle_mask)
    h, w = dist_field.shape
    dists = []
    for p in path:
        gx = int(np.clip(round(p[0] / MAP_RESOLUTION), 0, w-1))
        gy = int(np.clip(round(p[1] / MAP_RESOLUTION), 0, h-1))
        dists.append(dist_field[gy, gx])
    return np.mean(dists)

def select_best_path(paths, grid_map, criteria='shortest'):
    """
    从多条路径中选择最优路径
    参数：
        paths: List[List[(x, y)]] 多条路径
        grid_map: 2D numpy数组，地图
        criteria: 选择标准，默认'shortest'（最短路径）
    返回：
        best_path: 最优路径
        best_idx: 最优路径索引
    """
    valid_paths = []
    for idx, path in enumerate(paths):
        if validate_path(path, grid_map):
            valid_paths.append((idx, path))
    if not valid_paths:
        raise ValueError('没有可行路径')
    if criteria == 'shortest':
        best = min(valid_paths, key=lambda x: path_length(x[1]))
        return best[1], best[0]
    elif criteria == 'smoothest':
        best = min(valid_paths, key=lambda x: path_smoothness(x[1]))
        return best[1], best[0]
    elif criteria == 'safest':
        best = max(valid_paths, key=lambda x: path_safety(x[1], grid_map))
        return best[1], best[0]
    else:
        raise NotImplementedError(f'未知的路径选择标准: {criteria}')
