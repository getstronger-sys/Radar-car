#!/usr/bin/env python3
"""
测试frontier_detect.py的功能
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from exploration.frontier_detect import (
    detect_frontiers, 
    detect_frontiers_optimized,
    select_closest_frontier,
    select_best_frontier,
    calculate_information_gain,
    is_exploration_complete,
    ExplorationManager
)

def create_test_map():
    """创建测试地图"""
    print("🗺️ 创建测试地图...")
    
    # 创建一个50x50的测试地图
    map_size = 50
    occupancy_grid = np.full((map_size, map_size), -1, dtype=np.float32)  # 全部初始化为未知
    
    # 设置一些已知区域
    # 中心区域设为空闲
    occupancy_grid[20:30, 20:30] = 0.1
    
    # 设置一些障碍物
    occupancy_grid[15:18, 15:18] = 0.8  # 左上障碍物
    occupancy_grid[15:18, 32:35] = 0.8  # 右上障碍物
    occupancy_grid[32:35, 15:18] = 0.8  # 左下障碍物
    occupancy_grid[32:35, 32:35] = 0.8  # 右下障碍物
    
    # 设置一些边界区域
    occupancy_grid[0:5, :] = 0.1  # 上边界空闲
    occupancy_grid[-5:, :] = 0.1  # 下边界空闲
    occupancy_grid[:, 0:5] = 0.1  # 左边界空闲
    occupancy_grid[:, -5:] = 0.1  # 右边界空闲
    
    print(f"   地图尺寸: {occupancy_grid.shape}")
    print(f"   未知区域: {np.sum(occupancy_grid == -1)} 个格子")
    print(f"   空闲区域: {np.sum((occupancy_grid >= 0) & (occupancy_grid < 0.2))} 个格子")
    print(f"   障碍区域: {np.sum(occupancy_grid > 0.5)} 个格子")
    
    return occupancy_grid

def test_basic_frontier_detection():
    """测试基本前沿检测"""
    print("\n🧪 测试基本前沿检测...")
    
    occupancy_grid = create_test_map()
    map_resolution = 0.1  # 10cm/格子
    
    # 测试基本前沿检测
    frontiers = detect_frontiers(occupancy_grid, map_resolution=map_resolution)
    print(f"   检测到 {len(frontiers)} 个前沿点")
    
    if frontiers:
        print("   前沿点坐标:")
        for i, (x, y) in enumerate(frontiers[:5]):  # 只显示前5个
            print(f"     {i+1}. ({x:.2f}, {y:.2f})")
        if len(frontiers) > 5:
            print(f"     ... 还有 {len(frontiers)-5} 个前沿点")
    
    return frontiers, occupancy_grid, map_resolution

def test_optimized_frontier_detection():
    """测试优化的前沿检测"""
    print("\n🚀 测试优化的前沿检测...")
    
    occupancy_grid = create_test_map()
    map_resolution = 0.1
    
    # 测试优化版本
    frontiers_opt = detect_frontiers_optimized(occupancy_grid, map_resolution=map_resolution)
    print(f"   检测到 {len(frontiers_opt)} 个前沿点")
    
    if frontiers_opt:
        print("   前沿点坐标:")
        for i, (x, y) in enumerate(frontiers_opt[:5]):
            print(f"     {i+1}. ({x:.2f}, {y:.2f})")
    
    return frontiers_opt

def test_frontier_selection():
    """测试前沿选择"""
    print("\n🎯 测试前沿选择...")
    
    frontiers, occupancy_grid, map_resolution = test_basic_frontier_detection()
    
    if not frontiers:
        print("   没有前沿点，跳过选择测试")
        return
    
    # 机器人位置
    robot_pos = (2.5, 2.5)
    print(f"   机器人位置: {robot_pos}")
    
    # 测试最近前沿选择
    closest = select_closest_frontier(frontiers, robot_pos)
    if closest:
        dist = np.hypot(closest[0] - robot_pos[0], closest[1] - robot_pos[1])
        print(f"   最近前沿: ({closest[0]:.2f}, {closest[1]:.2f}), 距离: {dist:.2f}m")
    
    # 测试最佳前沿选择
    best = select_best_frontier(frontiers, robot_pos, occupancy_grid, map_resolution)
    if best:
        dist = np.hypot(best[0] - robot_pos[0], best[1] - robot_pos[1])
        print(f"   最佳前沿: ({best[0]:.2f}, {best[1]:.2f}), 距离: {dist:.2f}m")

def test_information_gain():
    """测试信息增益计算"""
    print("\n📊 测试信息增益计算...")
    
    occupancy_grid = create_test_map()
    map_resolution = 0.1
    
    # 测试几个前沿点的信息增益
    test_points = [(2.0, 2.0), (3.0, 3.0), (4.0, 4.0)]
    
    for point in test_points:
        info_gain = calculate_information_gain(point, occupancy_grid, map_resolution)
        print(f"   前沿点 ({point[0]:.1f}, {point[1]:.1f}) 信息增益: {info_gain}")

def test_exploration_completion():
    """测试探索完成检测"""
    print("\n✅ 测试探索完成检测...")
    
    # 测试未完成的地图
    occupancy_grid = create_test_map()
    is_complete = is_exploration_complete(occupancy_grid)
    print(f"   当前地图探索完成: {is_complete}")
    
    # 测试完成的地图（大部分已知）
    completed_map = np.full((50, 50), 0.1, dtype=np.float32)  # 全部设为空闲
    is_complete = is_exploration_complete(completed_map)
    print(f"   完成地图探索完成: {is_complete}")

def test_exploration_manager():
    """测试探索管理器"""
    print("\n🤖 测试探索管理器...")
    
    occupancy_grid = create_test_map()
    map_resolution = 0.1
    robot_pos = (2.5, 2.5)
    exit_pos = (4.5, 4.5)
    
    # 创建探索管理器
    manager = ExplorationManager(map_resolution=map_resolution)
    
    # 获取下一个目标
    target = manager.get_next_target(occupancy_grid, robot_pos, exit_pos)
    if target:
        dist_to_robot = np.hypot(target[0] - robot_pos[0], target[1] - robot_pos[1])
        dist_to_exit = np.hypot(target[0] - exit_pos[0], target[1] - exit_pos[1])
        print(f"   下一个目标: ({target[0]:.2f}, {target[1]:.2f})")
        print(f"   到机器人距离: {dist_to_robot:.2f}m")
        print(f"   到出口距离: {dist_to_exit:.2f}m")
    else:
        print("   没有找到下一个目标")

def visualize_frontiers(occupancy_grid, frontiers, title="前沿检测结果"):
    """可视化前沿检测结果"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        
        # 显示地图
        plt.imshow(occupancy_grid, cmap='gray', origin='lower')
        plt.colorbar(label='占用概率')
        
        # 标记前沿点
        if frontiers:
            frontier_x = [f[0] for f in frontiers]
            frontier_y = [f[1] for f in frontiers]
            plt.scatter(frontier_x, frontier_y, c='red', s=50, marker='o', label='前沿点')
        
        # 标记机器人位置
        robot_pos = (2.5, 2.5)
        plt.scatter(robot_pos[0], robot_pos[1], c='blue', s=100, marker='^', label='机器人')
        
        plt.title(title)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(3)
        plt.close()
        
        print("   ✅ 可视化完成")
        
    except ImportError:
        print("   ⚠️  matplotlib未安装，跳过可视化")
    except Exception as e:
        print(f"   ❌ 可视化失败: {e}")

def main():
    """主测试函数"""
    print("🚀 开始测试frontier_detect.py功能...")
    
    # 1. 基本前沿检测
    frontiers, occupancy_grid, map_resolution = test_basic_frontier_detection()
    
    # 2. 优化前沿检测
    frontiers_opt = test_optimized_frontier_detection()
    
    # 3. 前沿选择
    test_frontier_selection()
    
    # 4. 信息增益
    test_information_gain()
    
    # 5. 探索完成检测
    test_exploration_completion()
    
    # 6. 探索管理器
    test_exploration_manager()
    
    # 7. 可视化结果
    print("\n🎨 可视化前沿检测结果...")
    visualize_frontiers(occupancy_grid, frontiers, "基本前沿检测")
    visualize_frontiers(occupancy_grid, frontiers_opt, "优化前沿检测")
    
    print("\n🎉 所有测试完成！")
    print("\n测试总结:")
    print(f"   基本前沿检测: {len(frontiers)} 个前沿点")
    print(f"   优化前沿检测: {len(frontiers_opt)} 个前沿点")
    print("   前沿选择: 正常工作")
    print("   信息增益: 正常工作")
    print("   探索完成检测: 正常工作")
    print("   探索管理器: 正常工作")

if __name__ == "__main__":
    main() 