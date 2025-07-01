#用于实验的测试脚本
import json
from slam import SLAMWrapper

def test_slam():
    # 读取 JSON 文件
    json_file_path = r"d:\\Codes\\RD\\Radar-car-2\\slam\\test_data\\exploration_data(1).json"
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 初始化 SLAMWrapper
    slam = SLAMWrapper()  # 假设 SLAMWrapper 可直接初始化
    
    # 遍历激光雷达扫描数据
    for scan in data["lidar_scans"]:
        ranges = scan["ranges"]
        # 假设 update 方法接收雷达数据更新地图和位置
        slam.update(ranges)
        
        # 打印进度和可视化地图
        slam.print_progress()
        slam.visualize_map()

if __name__ == "__main__":
    test_slam()