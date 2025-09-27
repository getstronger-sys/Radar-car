"""
路径规划可视化模块

该模块提供路径规划算法的测试和可视化功能，包括：
- 多种A*算法的对比测试
- 路径连通性检测
- 可达区域可视化
- 路径平滑和验证
- 硬件对接：路径转折线，打印坐标和转向信息

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


# ========== 路径转折线函数 ==========
def path_to_line_segments(path: List[Tuple[float, float]], 
                         min_angle_threshold: float = 5.0) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    将路径转换为最少数量的折线
    
    Args:
        path: 路径点列表
        min_angle_threshold: 最小转向角度阈值（度）
        
    Returns:
        折线列表，每个元素为(起点, 终点)的元组
    """
    if len(path) < 2:
        return []
    
    line_segments = []
    current_start = path[0]
    current_direction = None
    
    for i in range(1, len(path)):
        # 计算当前段的方向向量
        dx = path[i][0] - path[i-1][0]
        dy = path[i][1] - path[i-1][1]
        current_segment_direction = np.arctan2(dy, dx)
        
        # 如果是第一段，记录方向
        if current_direction is None:
            current_direction = current_segment_direction
        else:
            # 计算方向变化角度（度）
            angle_diff = np.abs(current_segment_direction - current_direction) * 180 / np.pi
            # 处理角度跨越±180度的情况
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # 如果转向角度超过阈值，创建新的折线
            if angle_diff > min_angle_threshold:
                line_segments.append((current_start, path[i-1]))
                current_start = path[i-1]
                current_direction = current_segment_direction
    
    # 添加最后一段
    line_segments.append((current_start, path[-1]))
    
    return line_segments


def calculate_turning_angles(line_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> List[float]:
    """
    计算每条折线之间的转向角度
    
    Args:
        line_segments: 折线列表
        
    Returns:
        转向角度列表（度，正值表示左转，负值表示右转）
    """
    if len(line_segments) < 2:
        return []
    
    turning_angles = []
    
    for i in range(1, len(line_segments)):
        # 前一条折线的方向向量
        prev_end = line_segments[i-1][1]
        prev_start = line_segments[i-1][0]
        prev_dx = prev_end[0] - prev_start[0]
        prev_dy = prev_end[1] - prev_start[1]
        prev_angle = np.arctan2(prev_dy, prev_dx)
        
        # 当前折线的方向向量
        curr_start = line_segments[i][0]
        curr_end = line_segments[i][1]
        curr_dx = curr_end[0] - curr_start[0]
        curr_dy = curr_end[1] - curr_start[1]
        curr_angle = np.arctan2(curr_dy, curr_dx)
        
        # 计算转向角度
        angle_diff = curr_angle - prev_angle
        
        # 将角度限制在-180到180度之间
        while angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        while angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        
        # 转换为度
        turning_angle = angle_diff * 180 / np.pi
        turning_angles.append(turning_angle)
    
    return turning_angles


def print_line_segments_info(line_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]], 
                           turning_angles: List[float]) -> None:
    """
    打印折线信息，包括坐标和转向角度，并输出到文件
    
    Args:
        line_segments: 折线列表
        turning_angles: 转向角度列表
    """
    print("\n" + "=" * 80)
    print("🔧 硬件对接 - 路径转折线信息")
    print("=" * 80)
    
    print(f"\n📏 折线总数: {len(line_segments)}")
    
    # 准备输出到文件的数据
    output_lines = []
    
    for i, segment in enumerate(line_segments):
        start_x, start_y = segment[0]
        end_x, end_y = segment[1]
        segment_length = np.hypot(end_x - start_x, end_y - start_y)
        
        print(f"\n📍 折线 {i+1}:")
        print(f"   起点: ({start_x:.3f}, {start_y:.3f})")
        print(f"   终点: ({end_x:.3f}, {end_y:.3f})")
        print(f"   长度: {segment_length:.3f}m")
        
        # 计算方向角度
        dx = end_x - start_x
        dy = end_y - start_y
        direction_angle = np.arctan2(dy, dx) * 180 / np.pi
        print(f"   方向: {direction_angle:.1f}°")
        
        # 打印转向信息（除了第一条折线）
        if i > 0:
            turning_angle = turning_angles[i-1]
            turning_direction = "左转" if turning_angle > 0 else "右转"
            print(f"   转向: {turning_direction} {abs(turning_angle):.1f}°")
            
            # 添加到输出文件（格式：Px y 转向）
            output_lines.append(f"P{start_x:.3f} {start_y:.3f} {turning_angle:.1f}")
        else:
            # 第一条折线，转向为0
            output_lines.append(f"P{start_x:.3f} {start_y:.3f} 0.0")
    
    # 添加最后一条折线的终点
    if line_segments:
        last_end_x, last_end_y = line_segments[-1][1]
        output_lines.append(f"P{last_end_x:.3f} {last_end_y:.3f} 0.0")
    
    print(f"\n🔄 总转向次数: {len(turning_angles)}")
    if turning_angles:
        max_turn = max(turning_angles, key=abs)
        print(f"   最大转向角度: {max_turn:.1f}°")
        avg_turn = np.mean([abs(angle) for angle in turning_angles])
        print(f"   平均转向角度: {avg_turn:.1f}°")
    
    # 输出到文件
    output_file_path = "output/hardware_path_data.txt"
    try:
        # 确保输出目录存在
        os.makedirs("output", exist_ok=True)
        
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write("# 硬件对接路径数据 - 格式：Px y 转向\n")
            f.write(f"# 生成时间：{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 折线总数：{len(line_segments)}\n")
            f.write(f"# 总转向次数：{len(turning_angles)}\n")
            f.write("# 数据格式说明：Px y 转向 - 其中转向为0表示直行，正值表示左转，负值表示右转\n\n")
            
            for i, line in enumerate(output_lines):
                f.write(f"{line}\n")
        
        print(f"\n💾 硬件路径数据已保存到: {output_file_path}")
        print("📄 文件格式: Px y 转向")
        print("📋 数据内容:")
        for i, line in enumerate(output_lines):
            print(f"   {i+1:2d}: {line}")
            
    except Exception as e:
        print(f"\n❌ 保存文件失败: {e}")


def plot_line_segments_only(grid_map: np.ndarray,
                           start: Dict[str, float],
                           goal: Dict[str, float],
                           line_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> None:
    """
    只可视化折线，用于硬件对接
    
    Args:
        grid_map: 栅格地图
        start: 起始位置
        goal: 目标位置
        line_segments: 折线列表
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
    
    # 绘制折线
    colors = plt.cm.Set3(np.linspace(0, 1, len(line_segments)))
    for i, (segment, color) in enumerate(zip(line_segments, colors)):
        start_point = segment[0]
        end_point = segment[1]
        
        # 绘制折线
        plt.plot([start_point[0], end_point[0]], [start_point[1], end_point[1]], 
                color=color, linewidth=4, alpha=0.8, label=f'Segment {i+1}')
        
        # 标记折线端点
        plt.scatter([start_point[0]], [start_point[1]], c=color, s=60, alpha=0.8, zorder=6)
        plt.scatter([end_point[0]], [end_point[1]], c=color, s=60, alpha=0.8, zorder=6)
    
    # 设置图像范围与标签
    plt.xlim(0, map_size_m)
    plt.ylim(0, map_size_m)
    plt.xlabel('X [m]', fontsize=12)
    plt.ylabel('Y [m]', fontsize=12)
    plt.title('Hardware Integration - Line Segments Visualization', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ========== 注释掉的可视化函数 ==========
def plot_map(grid_map: np.ndarray, 
            start: Dict[str, float], 
            goal: Dict[str, float], 
            path: Optional[List[Tuple[float, float]]] = None, 
            smoothed_path: Optional[List[Tuple[float, float]]] = None) -> None:
    # 注释掉的可视化函数
    pass

def plot_smoothed_path_comparison(grid_map: np.ndarray,
                                 start: Dict[str, float],
                                 goal: Dict[str, float],
                                 original_path: List[Tuple[float, float]],
                                 smoothed_path: List[Tuple[float, float]]) -> None:
    # 注释掉的可视化函数
    pass

def plot_smoothed_path_only(grid_map: np.ndarray,
                           start: Dict[str, float],
                           goal: Dict[str, float],
                           smoothed_path: List[Tuple[float, float]]) -> None:
    # 注释掉的可视化函数
    pass

def plot_reachable_area(grid_map: np.ndarray, 
                       start: Dict[str, float], 
                       resolution: float) -> None:
    # 注释掉的可视化函数
    pass


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
    
    # 硬件对接：路径转折线
    print("\n🔧 硬件对接处理")
    # 选择有效路径进行折线转换
    path_to_convert = main_path if main_path and len(main_path) > 1 else (
        simple_path if simple_path and len(simple_path) > 1 else None)
    
    if path_to_convert:
        print(f"   使用路径长度: {len(path_to_convert)} 个点")
        
        # 转换为折线
        line_segments = path_to_line_segments(path_to_convert, min_angle_threshold=5.0)
        
        # 计算转向角度
        turning_angles = calculate_turning_angles(line_segments)
        
        # 打印折线信息
        print_line_segments_info(line_segments, turning_angles)
        
        # 可视化折线
        print("\n📊 可视化折线...")
        plot_line_segments_only(grid_map, start, goal, line_segments)
        
    else:
        print("❌ 没有找到有效路径，无法转换为折线")
    
    print("\n✅ 硬件对接测试完成")


if __name__ == "__main__":
    main()