import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目根目录路径，使得可以导入 planner 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from planner.path_planner import plan_path, smooth_path

# ========== 1. 构建地图 ==========
map_size = 50            # 地图为 50x50 的栅格地图
map_size_m = 5.0         # 实际物理尺寸为 5.0m x 5.0m
resolution = map_size_m / map_size  # 每个栅格表示的实际长度（米）

# 初始化空地图（0 表示空地）
grid_map = np.zeros((map_size, map_size), dtype=np.uint8)

# 添加障碍物：
# 中心方形障碍物
grid_map[20:30, 20:30] = 1

# 右上角的矩形障碍物
grid_map[10:15, 35:40] = 1


# ========== 2. 可视化函数 ==========
def plot_map(grid_map, start, goal, path=None, smoothed_path=None):
    """可视化地图、起点终点、路径和光滑路径"""
    plt.figure(figsize=(8, 8))

    # 显示背景地图
    plt.imshow(grid_map, cmap='Greys', origin='lower',
               extent=(0, map_size_m, 0, map_size_m), alpha=0.3)

    # 绘制障碍物点（黑色小圆点）
    obs_y, obs_x = np.where(grid_map == 1)
    plt.scatter(obs_x * resolution + resolution / 2,
                obs_y * resolution + resolution / 2,
                c='k', s=10, label='Obstacles')

    # 起点（绿色圆点）与终点（红色星型）
    plt.scatter([start['x']], [start['y']], c='g', s=100, marker='o', label='Start')
    plt.scatter([goal['x']], [goal['y']], c='r', s=100, marker='*', label='Goal')

    # 规划路径（蓝色实线）
    if path:
        px, py = zip(*path)
        plt.plot(px, py, 'b-', linewidth=2, label='Path')

    # 平滑路径（洋红色虚线）
    if smoothed_path:
        spx, spy = zip(*smoothed_path)
        plt.plot(spx, spy, 'm--', linewidth=2, label='Smoothed Path')

    # 设置图像范围与标签
    plt.xlim(0, map_size_m)
    plt.ylim(0, map_size_m)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Path Planning Visualization')
    plt.legend()
    plt.grid(True)
    plt.show()


# ========== 3. 主程序 ==========
if __name__ == "__main__":
    # 设置起点和终点的实际坐标（单位为米）
    start = {'x': 0.5, 'y': 0.5}
    goal = {'x': 4.5, 'y': 4.5}

    # 进行路径规划，plan_path
    path = plan_path(grid_map, start, goal, smooth_path_flag=False)

    # 如果路径存在且点数大于2，则进行路径平滑处理
    smoothed = smooth_path(path, grid_map=grid_map) if path and len(path) > 2 else None

    # 绘制最终结果
    plot_map(grid_map, start, goal, path=path, smoothed_path=smoothed)
