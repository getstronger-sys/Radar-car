"""
路径规划可视化模块

该模块提供路径规划算法的测试和可视化功能，包括：
- 多种A*算法的对比测试
- 路径连通性检测
- 可达区域可视化
- 路径平滑和验证

"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from collections import deque
from typing import List, Tuple, Dict, Optional

# 添加项目根目录路径，使得可以导入 planner 模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.path_planner import plan_path, smooth_path, plan_path_simple, plan_path_pythonrobotics
from config.map import get_global_map, MAP_SIZE_M, MAP_RESOLUTION
from config.settings import START_POSITION, EXIT_POSITION


def align_to_grid_center(pos: Dict[str, float], resolution: float) -> Dict[str, float]:
    """
    将位置坐标对齐到栅格中心
    
    Args:
        pos: 位置字典 {'x': float, 'y': float}
        resolution: 地图分辨率
        
    Returns:
        对齐后的位置字典
    """
    return {
        'x': (int(pos['x'] / resolution) + 0.5) * resolution,
        'y': (int(pos['y'] / resolution) + 0.5) * resolution
    }


# ========== 全局地图配置 ==========
grid_map = get_global_map()
map_size = grid_map.shape[0]
map_size_m = MAP_SIZE_M
resolution = MAP_RESOLUTION


# ========== 可视化函数 ==========
def plot_map(grid_map: np.ndarray, 
            start: Dict[str, float], 
            goal: Dict[str, float], 
            path: Optional[List[Tuple[float, float]]] = None, 
            smoothed_path: Optional[List[Tuple[float, float]]] = None) -> None:
    """
    可视化地图、起点终点、路径和光滑路径
    
    Args:
        grid_map: 栅格地图
        start: 起始位置
        goal: 目标位置
        path: 原始路径
        smoothed_path: 平滑后的路径
    """
    plt.figure(figsize=(10, 8))

    # 显示背景地图
    plt.imshow(grid_map, cmap='Greys', origin='lower',
               extent=(0, map_size_m, 0, map_size_m), alpha=0.3)

    # 绘制障碍物点（黑色小圆点）
    obs_y, obs_x = np.where(grid_map == 1)
    plt.scatter(obs_x * resolution + resolution / 2,
                obs_y * resolution + resolution / 2,
                c='k', s=10, label='Obstacles', alpha=0.7)

    # 起点（绿色圆点）与终点（红色星型）
    plt.scatter([start['x']], [start['y']], c='g', s=100, marker='o', label='Start')
    plt.scatter([goal['x']], [goal['y']], c='r', s=100, marker='*', label='Goal')

    # 规划路径（蓝色实线）
    if path:
        px, py = zip(*path)
        plt.plot(px, py, 'b-', linewidth=2, label='Original Path', alpha=0.8)

    # 平滑路径（洋红色虚线）
    if smoothed_path:
        spx, spy = zip(*smoothed_path)
        plt.plot(spx, spy, 'm--', linewidth=3, label='Smoothed Path', alpha=0.9)

    # 设置图像范围与标签
    plt.xlim(0, map_size_m)
    plt.ylim(0, map_size_m)
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Path Planning Visualization')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_smoothed_path_comparison(grid_map: np.ndarray,
                                 start: Dict[str, float],
                                 goal: Dict[str, float],
                                 original_path: List[Tuple[float, float]],
                                 smoothed_path: List[Tuple[float, float]]) -> None:
    """
    专门可视化原始路径和平滑路径的对比
    
    Args:
        grid_map: 栅格地图
        start: 起始位置
        goal: 目标位置
        original_path: 原始路径
        smoothed_path: 平滑后的路径
    """
    # 创建子图布局
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 计算路径长度
    original_length = calculate_path_length(original_path)
    smoothed_length = calculate_path_length(smoothed_path)
    
    # 左图：原始路径
    ax1.imshow(grid_map, cmap='Greys', origin='lower',
               extent=(0, map_size_m, 0, map_size_m), alpha=0.3)
    
    # 绘制障碍物
    obs_y, obs_x = np.where(grid_map == 1)
    ax1.scatter(obs_x * resolution + resolution / 2,
                obs_y * resolution + resolution / 2,
                c='k', s=10, alpha=0.7)
    
    # 起点和终点
    ax1.scatter([start['x']], [start['y']], c='g', s=100, marker='o', label='Start')
    ax1.scatter([goal['x']], [goal['y']], c='r', s=100, marker='*', label='Goal')
    
    # 原始路径
    px, py = zip(*original_path)
    ax1.plot(px, py, 'b-', linewidth=3, label=f'Original Path ({len(original_path)} points)')
    
    # 标记路径点
    ax1.scatter(px, py, c='blue', s=20, alpha=0.6, zorder=5)
    
    ax1.set_xlim(0, map_size_m)
    ax1.set_ylim(0, map_size_m)
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_title(f'Original Path\nLength: {original_length:.3f}m')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 右图：平滑路径
    ax2.imshow(grid_map, cmap='Greys', origin='lower',
               extent=(0, map_size_m, 0, map_size_m), alpha=0.3)
    
    # 绘制障碍物
    ax2.scatter(obs_x * resolution + resolution / 2,
                obs_y * resolution + resolution / 2,
                c='k', s=10, alpha=0.7)
    
    # 起点和终点
    ax2.scatter([start['x']], [start['y']], c='g', s=100, marker='o', label='Start')
    ax2.scatter([goal['x']], [goal['y']], c='r', s=100, marker='*', label='Goal')
    
    # 平滑路径
    spx, spy = zip(*smoothed_path)
    ax2.plot(spx, spy, 'm-', linewidth=3, label=f'Smoothed Path ({len(smoothed_path)} points)')
    
    # 标记路径点
    ax2.scatter(spx, spy, c='magenta', s=20, alpha=0.6, zorder=5)
    
    ax2.set_xlim(0, map_size_m)
    ax2.set_ylim(0, map_size_m)
    ax2.set_xlabel('X [m]')
    ax2.set_ylabel('Y [m]')
    ax2.set_title(f'Smoothed Path\nLength: {smoothed_length:.3f}m')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印对比信息
    print(f"\n📊 路径对比信息:")
    print(f"   原始路径: {len(original_path)} 个点, 长度: {original_length:.3f}m")
    print(f"   平滑路径: {len(smoothed_path)} 个点, 长度: {smoothed_length:.3f}m")
    print(f"   长度变化: {smoothed_length - original_length:+.3f}m ({((smoothed_length - original_length) / original_length * 100):+.1f}%)")


def plot_smoothed_path_only(grid_map: np.ndarray,
                           start: Dict[str, float],
                           goal: Dict[str, float],
                           smoothed_path: List[Tuple[float, float]]) -> None:
    """
    单独可视化平滑后的路径
    
    Args:
        grid_map: 栅格地图
        start: 起始位置
        goal: 目标位置
        smoothed_path: 平滑后的路径
    """
    plt.figure(figsize=(12, 10))
    
    # 显示背景地图
    plt.imshow(grid_map, cmap='Greys', origin='lower',
               extent=(0, map_size_m, 0, map_size_m), alpha=0.3)
    
    # 绘制障碍物点
    obs_y, obs_x = np.where(grid_map == 1)
    plt.scatter(obs_x * resolution + resolution / 2,
                obs_y * resolution + resolution / 2,
                c='k', s=10, label='Obstacles', alpha=0.7)
    
    # 起点和终点
    plt.scatter([start['x']], [start['y']], c='g', s=120, marker='o', label='Start', zorder=10)
    plt.scatter([goal['x']], [goal['y']], c='r', s=120, marker='*', label='Goal', zorder=10)
    
    # 平滑路径
    spx, spy = zip(*smoothed_path)
    plt.plot(spx, spy, 'm-', linewidth=4, label='Smoothed Path', alpha=0.9, zorder=5)
    
    # 标记路径点
    plt.scatter(spx, spy, c='magenta', s=30, alpha=0.8, zorder=6, label='Path Points')
    
    # 突出显示起点和终点
    plt.scatter([spx[0]], [spy[0]], c='lime', s=150, marker='o', edgecolors='green', linewidth=3, zorder=11, label='Path Start')
    plt.scatter([spx[-1]], [spy[-1]], c='orange', s=150, marker='*', edgecolors='red', linewidth=3, zorder=11, label='Path End')
    
    # 设置图像范围与标签
    plt.xlim(0, map_size_m)
    plt.ylim(0, map_size_m)
    plt.xlabel('X [m]', fontsize=12)
    plt.ylabel('Y [m]', fontsize=12)
    plt.title('Smoothed Path Visualization', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 打印平滑路径信息
    path_length = calculate_path_length(smoothed_path)
    print(f"\n🎯 平滑路径信息:")
    print(f"   路径点数: {len(smoothed_path)}")
    print(f"   路径长度: {path_length:.3f}m")
    print(f"   起点: ({spx[0]:.3f}, {spy[0]:.3f})")
    print(f"   终点: ({spx[-1]:.3f}, {spy[-1]:.3f})")


def is_connected(grid_map: np.ndarray, 
                start: Dict[str, float], 
                goal: Dict[str, float], 
                resolution: float) -> bool:
    """
    使用Flood Fill算法检测起点和终点是否在同一连通区域
    
    Args:
        grid_map: 栅格地图
        start: 起始位置
        goal: 目标位置
        resolution: 地图分辨率
        
    Returns:
        True表示连通，False表示不连通
    """
    h, w = grid_map.shape
    visited = np.zeros_like(grid_map, dtype=bool)
    
    # 转换为格子坐标
    sx = int(start['x'] / resolution)
    sy = int(start['y'] / resolution)
    gx = int(goal['x'] / resolution)
    gy = int(goal['y'] / resolution)
    
    # 检查起点和终点是否在障碍物内
    if grid_map[sy, sx] != 0 or grid_map[gy, gx] != 0:
        return False
    
    # Flood Fill算法
    queue = deque([(sx, sy)])
    visited[sy, sx] = True
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上下左右四个方向
    
    while queue:
        x, y = queue.popleft()
        
        # 找到目标点
        if (x, y) == (gx, gy):
            return True
            
        # 检查四个方向的邻居
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < w and 0 <= ny < h and 
                not visited[ny, nx] and grid_map[ny, nx] == 0):
                visited[ny, nx] = True
                queue.append((nx, ny))
    
    return False


def plot_reachable_area(grid_map: np.ndarray, 
                       start: Dict[str, float], 
                       resolution: float) -> None:
    """
    可视化起点flood fill可达的所有格子
    
    Args:
        grid_map: 栅格地图
        start: 起始位置
        resolution: 地图分辨率
    """
    h, w = grid_map.shape
    visited = np.zeros_like(grid_map, dtype=bool)
    
    # 转换为格子坐标
    sx = int(start['x'] / resolution)
    sy = int(start['y'] / resolution)
    
    # 检查起点是否在障碍物内
    if grid_map[sy, sx] != 0:
        print("⚠️  起点在障碍物内，无法可视化可达区域！")
        return
    
    # Flood Fill算法
    queue = deque([(sx, sy)])
    visited[sy, sx] = True
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        x, y = queue.popleft()
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < w and 0 <= ny < h and 
                not visited[ny, nx] and grid_map[ny, nx] == 0):
                visited[ny, nx] = True
                queue.append((nx, ny))
    
    # 可视化可达区域
    plt.figure(figsize=(10, 8))
    plt.imshow(grid_map, cmap='Greys', origin='lower', 
               extent=(0, map_size_m, 0, map_size_m), alpha=0.3)
    
    # 可达区域用浅蓝色显示
    reachable_y, reachable_x = np.where(visited)
    plt.scatter(reachable_x * resolution + resolution / 2, 
                reachable_y * resolution + resolution / 2, 
                c='cyan', s=15, alpha=0.6, label='Reachable Area')
    
    # 标记起点
    plt.scatter([start['x']], [start['y']], c='g', s=100, marker='o', label='Start')
    
    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Reachable Area from Start Point')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def calculate_path_length(path: List[Tuple[float, float]]) -> float:
    """
    计算路径总长度
    
    Args:
        path: 路径点列表
        
    Returns:
        路径总长度（米）
    """
    if len(path) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        total_length += np.hypot(dx, dy)
    
    return total_length


def run_path_planning_algorithms(grid_map: np.ndarray, 
                                start: Dict[str, float], 
                                goal: Dict[str, float]) -> Tuple[Optional[List], Optional[List], Optional[List]]:
    """
    测试不同的路径规划算法
    
    Args:
        grid_map: 栅格地图
        start: 起始位置
        goal: 目标位置
        
    Returns:
        三种算法的路径结果：(simple_path, pythonrobotics_path, main_path)
    """
    print("=" * 70)
    print("🧪 路径规划算法对比测试")
    print("=" * 70)
    
    simple_path = None
    pythonrobotics_path = None
    main_path = None

    # 测试1: 简单A*算法
    print("\n1️⃣ 测试简单A*算法 (plan_path_simple)")
    try:
        simple_path = plan_path_simple(grid_map, start, goal, resolution)
        if simple_path:
            path_length = calculate_path_length(simple_path)
            print(f"   ✅ 找到路径")
            print(f"   路径点数: {len(simple_path)}")
            print(f"   路径长度: {path_length:.3f}m")
        else:
            print("   ❌ 未找到路径")
    except Exception as e:
        print(f"   ❌ 算法错误: {e}")

    # 测试2: PythonRobotics A*算法
    print("\n2️⃣ 测试PythonRobotics A*算法 (plan_path_pythonrobotics)")
    try:
        pythonrobotics_path = plan_path_pythonrobotics(grid_map, start, goal)
        if pythonrobotics_path:
            path_length = calculate_path_length(pythonrobotics_path)
            print(f"   ✅ 找到路径")
            print(f"   路径点数: {len(pythonrobotics_path)}")
            print(f"   路径长度: {path_length:.3f}m")
        else:
            print("   ❌ 未找到路径")
    except Exception as e:
        print(f"   ❌ 算法错误: {e}")

    # 测试3: 主路径规划函数 (自动选择算法)
    print("\n3️⃣ 测试主路径规划函数 (plan_path)")
    try:
        main_path = plan_path(grid_map, start, goal, smooth_path_flag=True)
        if main_path:
            path_length = calculate_path_length(main_path)
            print(f"   ✅ 找到路径")
            print(f"   路径点数: {len(main_path)}")
            print(f"   路径长度: {path_length:.3f}m")
        else:
            print("   ❌ 未找到路径")
    except Exception as e:
        print(f"   ❌ 算法错误: {e}")

    # 测试4: 路径验证
    print("\n4️⃣ 路径验证测试")
    if main_path:
        try:
            from planner.path_planner import validate_path
            is_valid = validate_path(main_path, grid_map, resolution)
            print(f"   路径验证: {'✅ 通过' if is_valid else '❌ 失败'}")
        except Exception as e:
            print(f"   路径验证: ❌ 验证过程出错 - {e}")
    else:
        print("   路径验证: ⚠️  无有效路径可验证")

    return simple_path, pythonrobotics_path, main_path


def print_environment_info(grid_map: np.ndarray, 
                          start: Dict[str, float], 
                          goal: Dict[str, float], 
                          resolution: float) -> None:
    """
    打印环境信息
    
    Args:
        grid_map: 栅格地图
        start: 起始位置
        goal: 目标位置
        resolution: 地图分辨率
    """
    print(f"\n📊 环境信息")
    print(f"   地图尺寸: {grid_map.shape}")
    print(f"   地图物理尺寸: {map_size_m}m x {map_size_m}m")
    print(f"   分辨率: {resolution}m/格子")
    print(f"   起点: ({start['x']:.2f}, {start['y']:.2f})")
    print(f"   终点: ({goal['x']:.2f}, {goal['y']:.2f})")
    
    # 检查起点和终点的格子坐标
    gx_start = int(start['x'] / resolution)
    gy_start = int(start['y'] / resolution)
    gx_goal = int(goal['x'] / resolution)
    gy_goal = int(goal['y'] / resolution)
    
    print(f"   起点格子坐标: ({gx_start}, {gy_start}), 值: {grid_map[gy_start, gx_start]}")
    print(f"   终点格子坐标: ({gx_goal}, {gy_goal}), 值: {grid_map[gy_goal, gx_goal]}")
    
    # 检查连通性
    connected = is_connected(grid_map, start, goal, resolution)
    print(f"   起点终点连通性: {'✅ 连通' if connected else '❌ 不连通'}")


def print_goal_environment(grid_map: np.ndarray, 
                          goal: Dict[str, float], 
                          resolution: float) -> None:
    """
    打印终点周围环境信息
    
    Args:
        grid_map: 栅格地图
        goal: 目标位置
        resolution: 地图分辨率
    """
    gx_goal = int(goal['x'] / resolution)
    gy_goal = int(goal['y'] / resolution)
    
    print(f"\n🎯 终点周围环境 (3x3):")
    for dy in range(-1, 2):
        row_info = []
        for dx in range(-1, 2):
            x = gx_goal + dx
            y = gy_goal + dy
            if 0 <= x < grid_map.shape[1] and 0 <= y < grid_map.shape[0]:
                value = grid_map[y, x]
                marker = "🎯" if dx == 0 and dy == 0 else " "
                row_info.append(f"{marker}({x},{y}):{value}")
            else:
                row_info.append("  边界外  ")
        print("   " + "  ".join(row_info))


# ========== 主程序 ==========
def main():
    """主程序入口"""
    print("🚀 开始路径规划测试")
    
    # 使用全局设置的起点和终点坐标，并对齐到格子中心
    start = align_to_grid_center(START_POSITION, resolution)
    goal = align_to_grid_center(EXIT_POSITION, resolution)
    
    # 打印环境信息
    print_environment_info(grid_map, start, goal, resolution)
    
    # 检查连通性
    connected = is_connected(grid_map, start, goal, resolution)
    if not connected:
        print("\n⚠️  起点和终点不连通，路径规划可能失败")
    
    # 测试所有路径规划算法
    simple_path, pythonrobotics_path, main_path = run_path_planning_algorithms(
        grid_map, start, goal)
    
    # 可视化结果
    print("\n📊 可视化结果")
    # 判断主路径是否有效（点数大于1）
    valid_main_path = main_path and len(main_path) > 1
    path_to_show = main_path if valid_main_path else (simple_path if simple_path and len(simple_path) > 1 else None)
    if path_to_show:
        print(f"   {'主路径' if valid_main_path else 'simple_path'}长度: {len(path_to_show)} 个点")
        # 1. 使用主路径或simple_path进行可视化
        print("   1️⃣ 显示原始路径图...")
        plot_map(grid_map, start, goal, path=path_to_show)
        # 2. 生成带障碍物约束的平滑路径进行对比
        print("\n🔄 生成带障碍物约束的平滑路径...")
        try:
            from planner.path_planner import smooth_path_with_obstacle_avoidance
            original_path_for_smoothing = path_to_show
            smoothed_path = smooth_path_with_obstacle_avoidance(
                original_path_for_smoothing, grid_map, resolution, initial_smoothing=0.2, min_smoothing=0.01, max_iter=20, verbose=True)
            if smoothed_path and len(smoothed_path) > 2 and smoothed_path != original_path_for_smoothing:
                print(f"   避障平滑路径生成成功: {len(smoothed_path)} 个点")
                print("   2️⃣ 显示路径对比图...")
                plot_smoothed_path_comparison(grid_map, start, goal, original_path_for_smoothing, smoothed_path)
                print("   3️⃣ 单独显示平滑路径...")
                plot_smoothed_path_only(grid_map, start, goal, smoothed_path)
            else:
                print("   ⚠️  避障平滑路径生成失败或与原始路径无差异")
        except Exception as e:
            print(f"   ⚠️  避障平滑路径生成错误: {e}")
            import traceback
            traceback.print_exc()
        print("\n🔍 显示起点可达区域...")
        plot_reachable_area(grid_map, start, resolution)
        print_goal_environment(grid_map, goal, resolution)
    else:
        print("❌ 没有找到有效路径，无法可视化")
        plot_reachable_area(grid_map, start, resolution)
    print("\n✅ 路径规划测试完成")


if __name__ == "__main__":
    main()