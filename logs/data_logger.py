import json
import time
import numpy as np
from datetime import datetime
import os

class DataLogger:
    def __init__(self, log_dir="logs"):
        """初始化数据记录器"""
        self.log_dir = log_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"navigation_log_{self.session_id}.json")
        
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)
        
        # 初始化日志数据结构
        self.log_data = {
            "session_info": {
                "session_id": self.session_id,
                "start_time": datetime.now().isoformat(),
                "description": "Autonomous Navigation Session"
            },
            "sensor_data": [],
            "robot_trajectory": [],
            "slam_updates": [],
            "path_planning": [],
            "control_commands": [],
            "performance_metrics": []
        }
        
        print(f"📝 数据记录器初始化完成，日志文件: {self.log_file}")

    def log_sensor_data(self, lidar_data, odom_data, timestamp=None):
        """记录传感器数据"""
        if timestamp is None:
            timestamp = time.time()
        
        entry = {
            "timestamp": timestamp,
            "lidar_data": {
                "scan_size": len(lidar_data),
                "max_distance": max(lidar_data) if lidar_data else 0,
                "min_distance": min(lidar_data) if lidar_data else 0,
                "average_distance": np.mean(lidar_data) if lidar_data else 0
            },
            "odometry": {
                "x": odom_data[0],
                "y": odom_data[1],
                "theta": odom_data[2]
            }
        }
        
        self.log_data["sensor_data"].append(entry)

    def log_robot_trajectory(self, pose, timestamp=None):
        """记录机器人轨迹"""
        if timestamp is None:
            timestamp = time.time()
        
        entry = {
            "timestamp": timestamp,
            "x": pose[0],
            "y": pose[1],
            "theta": pose[2]
        }
        
        self.log_data["robot_trajectory"].append(entry)

    def log_slam_update(self, pose, map_info, timestamp=None):
        """记录SLAM更新"""
        if timestamp is None:
            timestamp = time.time()
        
        entry = {
            "timestamp": timestamp,
            "estimated_pose": {
                "x": pose[0],
                "y": pose[1],
                "theta": pose[2]
            },
            "map_info": {
                "resolution": map_info.get("resolution", 0),
                "size": map_info.get("size", 0),
                "unknown_cells": map_info.get("unknown_cells", 0),
                "occupied_cells": map_info.get("occupied_cells", 0),
                "free_cells": map_info.get("free_cells", 0)
            }
        }
        
        self.log_data["slam_updates"].append(entry)

    def log_path_planning(self, start_pose, goal_pose, path, algorithm="A*", timestamp=None):
        """记录路径规划"""
        if timestamp is None:
            timestamp = time.time()
        
        entry = {
            "timestamp": timestamp,
            "algorithm": algorithm,
            "start_pose": {
                "x": start_pose[0],
                "y": start_pose[1]
            },
            "goal_pose": {
                "x": goal_pose[0],
                "y": goal_pose[1]
            },
            "path_length": len(path),
            "path_points": path[:10] if len(path) > 10 else path  # 只记录前10个点
        }
        
        self.log_data["path_planning"].append(entry)

    def log_control_command(self, v, omega, target_pose, timestamp=None):
        """记录控制命令"""
        if timestamp is None:
            timestamp = time.time()
        
        entry = {
            "timestamp": timestamp,
            "linear_velocity": v,
            "angular_velocity": omega,
            "target_pose": {
                "x": target_pose[0],
                "y": target_pose[1]
            }
        }
        
        self.log_data["control_commands"].append(entry)

    def log_performance_metric(self, metric_name, value, timestamp=None):
        """记录性能指标"""
        if timestamp is None:
            timestamp = time.time()
        
        entry = {
            "timestamp": timestamp,
            "metric_name": metric_name,
            "value": value
        }
        
        self.log_data["performance_metrics"].append(entry)

    def log_frontier_detection(self, frontiers, robot_pose, timestamp=None):
        """记录前沿检测结果"""
        if timestamp is None:
            timestamp = time.time()
        
        entry = {
            "timestamp": timestamp,
            "robot_pose": {
                "x": robot_pose[0],
                "y": robot_pose[1],
                "theta": robot_pose[2]
            },
            "frontier_count": len(frontiers),
            "frontiers": frontiers[:5] if len(frontiers) > 5 else frontiers  # 只记录前5个前沿
        }
        
        if "frontier_detection" not in self.log_data:
            self.log_data["frontier_detection"] = []
        
        self.log_data["frontier_detection"].append(entry)

    def save_log(self):
        """保存日志到文件"""
        try:
            # 添加结束时间
            self.log_data["session_info"]["end_time"] = datetime.now().isoformat()
            
            # 计算会话统计信息
            self._calculate_session_stats()
            
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.log_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 日志已保存到: {self.log_file}")
            return True
            
        except Exception as e:
            print(f"❌ 保存日志失败: {e}")
            return False

    def _calculate_session_stats(self):
        """计算会话统计信息"""
        if not self.log_data["robot_trajectory"]:
            return
        
        # 计算总行驶距离
        trajectory = self.log_data["robot_trajectory"]
        total_distance = 0
        for i in range(1, len(trajectory)):
            dx = trajectory[i]["x"] - trajectory[i-1]["x"]
            dy = trajectory[i]["y"] - trajectory[i-1]["y"]
            total_distance += np.sqrt(dx*dx + dy*dy)
        
        # 计算会话时长
        start_time = datetime.fromisoformat(self.log_data["session_info"]["start_time"])
        end_time = datetime.fromisoformat(self.log_data["session_info"]["end_time"])
        session_duration = (end_time - start_time).total_seconds()
        
        # 添加统计信息
        self.log_data["session_stats"] = {
            "total_distance": total_distance,
            "session_duration": session_duration,
            "average_speed": total_distance / session_duration if session_duration > 0 else 0,
            "total_sensor_readings": len(self.log_data["sensor_data"]),
            "total_slam_updates": len(self.log_data["slam_updates"]),
            "total_path_plans": len(self.log_data["path_planning"]),
            "total_control_commands": len(self.log_data["control_commands"])
        }

    def get_recent_trajectory(self, num_points=100):
        """获取最近的轨迹点"""
        trajectory = self.log_data["robot_trajectory"]
        return trajectory[-num_points:] if len(trajectory) > num_points else trajectory

    def get_performance_summary(self):
        """获取性能摘要"""
        if "session_stats" not in self.log_data:
            return None
        
        return self.log_data["session_stats"]

    def export_csv(self, data_type="trajectory"):
        """导出CSV格式数据"""
        csv_file = os.path.join(self.log_dir, f"{data_type}_{self.session_id}.csv")
        
        try:
            if data_type == "trajectory" and self.log_data["robot_trajectory"]:
                import pandas as pd
                df = pd.DataFrame(self.log_data["robot_trajectory"])
                df.to_csv(csv_file, index=False)
                print(f"✅ 轨迹数据已导出到: {csv_file}")
                return csv_file
                
        except Exception as e:
            print(f"❌ 导出CSV失败: {e}")
            return None

    def close(self):
        """关闭记录器并保存日志"""
        self.save_log()
        print("📝 数据记录器已关闭") 