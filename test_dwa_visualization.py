#!/usr/bin/env python3
"""
测试DWA可视化功能
"""

import sys
import os

# 添加项目根目录路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dwa_visualization():
    """测试DWA可视化功能"""
    try:
        print("🧪 开始测试DWA可视化功能...")
        
        # 导入可视化模块
        from planner.visualize_dwa_path import (
            run_dwa_simulation, 
            plot_dwa_simulation,
            dwa_config
        )
        
        print("✅ 模块导入成功")
        print(f"📋 DWA配置参数:")
        for key, value in dwa_config.items():
            print(f"   {key}: {value}")
        
        # 创建简单的地图进行测试
        import numpy as np
        
        map_size = 30  # 较小的地图用于快速测试
        map_size_m = 3.0
        resolution = map_size_m / map_size
        
        # 创建简单地图
        grid_map = np.zeros((map_size, map_size), dtype=np.uint8)
        
        # 添加一个中心障碍物
        grid_map[10:20, 10:20] = 1
        
        # 设置起点和终点
        start = {'x': 0.3, 'y': 0.3}
        goal = {'x': 2.7, 'y': 2.7}
        
        print(f"🗺️  地图尺寸: {map_size}x{map_size}")
        print(f"   起点: ({start['x']:.2f}, {start['y']:.2f})")
        print(f"   终点: ({goal['x']:.2f}, {goal['y']:.2f})")
        
        # 运行DWA仿真
        print("\n🚀 运行DWA仿真...")
        robot_states, control_history, path = run_dwa_simulation(
            grid_map, start, goal, max_iterations=500, goal_threshold=0.15
        )
        
        if robot_states:
            print(f"✅ 仿真成功完成")
            print(f"   机器人状态数: {len(robot_states)}")
            print(f"   控制输入数: {len(control_history)}")
            print(f"   全局路径点数: {len(path) if path else 0}")
            
            # 计算最终距离
            final_state = robot_states[-1]
            final_distance = np.hypot(final_state[0] - goal['x'], final_state[1] - goal['y'])
            print(f"   最终距离: {final_distance:.3f}m")
            
            # 可视化结果
            print("\n📊 生成可视化...")
            plot_dwa_simulation(
                grid_map, start, goal, path, robot_states, 
                robot_states, control_history, animation_mode=False
            )
            
            print("✅ 测试完成!")
            return True
        else:
            print("❌ 仿真失败")
            return False
            
    except ImportError as e:
        print(f"❌ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"❌ 测试错误: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dwa_visualization()
    if success:
        print("\n🎉 DWA可视化测试通过!")
    else:
        print("\n💥 DWA可视化测试失败!")
        sys.exit(1) 