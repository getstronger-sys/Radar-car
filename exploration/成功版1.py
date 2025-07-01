import numpy as np
import json
import math
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrow
from scipy.ndimage import label, binary_dilation
import time
import heapq


class LidarScan:
    def __init__(self, ranges, angles, timestamp):
        self.ranges = ranges
        self.angles = angles
        self.timestamp = timestamp


class Odometry:
    def __init__(self, x, y, theta, timestamp):
        self.x = x
        self.y = y
        self.theta = theta
        self.timestamp = timestamp


class Robot:
    def __init__(self, x, y, theta=0):
        self.x = x
        self.y = y
        self.theta = theta
        self.max_speed = 0.5
        self.max_angular_speed = 1.0

    def move_to_position(self, target_x, target_y):
        """直接移动到目标位置"""
        self.x = target_x
        self.y = target_y
        # 更新朝向
        if hasattr(self, 'last_x') and hasattr(self, 'last_y'):
            dx = target_x - self.last_x
            dy = target_y - self.last_y
            if abs(dx) > 0.001 or abs(dy) > 0.001:
                self.theta = math.atan2(dy, dx)
        self.last_x = self.x
        self.last_y = self.y

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


class MapConverter:
    def __init__(self, resolution=0.05):
        self.resolution = resolution

    def segments_to_occupancy_grid(self, segments, map_size=(16, 16)):
        """Convert line segments to occupancy grid"""
        width = int(map_size[0] / self.resolution)
        height = int(map_size[1] / self.resolution)

        # Initialize as free space (0)
        grid = np.zeros((height, width), dtype=np.float32)

        # Mark walls as occupied (1.0)
        for segment in segments:
            self._draw_line(grid, segment['start'], segment['end'])

        return grid

    def _draw_line(self, grid, start, end):
        """Draw a line on the grid using Bresenham's algorithm"""
        x0 = int(start[0] / self.resolution)
        y0 = int(start[1] / self.resolution)
        x1 = int(end[0] / self.resolution)
        y1 = int(end[1] / self.resolution)

        points = self._bresenham_line(x0, y0, x1, y1)

        for x, y in points:
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                grid[y, x] = 1.0  # Occupied

    def _bresenham_line(self, x0, y0, x1, y1):
        """Bresenham's line algorithm"""
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        n = 1 + dx + dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        error = dx - dy

        dx *= 2
        dy *= 2

        for _ in range(n):
            points.append((x, y))

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        return points


class LidarSimulator:
    def __init__(self, range_max=5.0, num_beams=360, angle_res=1):
        self.range_max = range_max
        self.num_beams = num_beams
        self.angle_res = angle_res
        self.angles = np.linspace(0, 2 * np.pi, num_beams, endpoint=False)

    def scan(self, robot_pos, occupancy_grid, resolution):
        """Simulate lidar scan - optimized version"""
        ranges = []
        robot_x, robot_y, robot_theta = robot_pos

        for angle in self.angles:
            # Global angle
            global_angle = robot_theta + angle

            # Ray casting - optimized
            range_val = self._cast_ray_fast(
                robot_x, robot_y, global_angle,
                occupancy_grid, resolution
            )
            ranges.append(range_val)

        return LidarScan(ranges, self.angles.tolist(), 0)

    def _cast_ray_fast(self, start_x, start_y, angle, grid, resolution):
        """Fast ray casting"""
        step_size = resolution
        max_steps = int(self.range_max / step_size)

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        for step in range(max_steps):
            x = start_x + step * step_size * cos_a
            y = start_y + step * step_size * sin_a

            # Convert to grid coordinates
            gx = int(x / resolution)
            gy = int(y / resolution)

            # Check bounds
            if (gx < 0 or gx >= grid.shape[1] or
                    gy < 0 or gy >= grid.shape[0]):
                return self.range_max

            # Check if obstacle
            if grid[gy, gx] > 0.5:
                return step * step_size

        return self.range_max


# Frontier exploration functions
def detect_frontiers_optimized(occupancy_grid, unknown_val=-1, free_threshold=0.2, map_resolution=0.01):
    """Optimized frontier detection function"""
    h, w = occupancy_grid.shape

    # Create masks
    free_mask = (occupancy_grid >= 0) & (occupancy_grid < free_threshold)
    unknown_mask = (occupancy_grid == unknown_val)

    # Use convolution to detect frontiers
    dilated_unknown = binary_dilation(unknown_mask, structure=np.ones((3, 3)))

    # Frontiers = free regions ∩ dilated unknown regions
    frontiers_mask = free_mask & dilated_unknown

    # Connected component labeling
    labeled, num_features = label(frontiers_mask)

    frontiers_world = []

    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)

        if len(xs) > 3:
            # Calculate centroid
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))

            # Convert to world coordinates
            x_m = cx * map_resolution
            y_m = cy * map_resolution
            frontiers_world.append((x_m, y_m))

    return frontiers_world


def select_best_frontier(frontiers, robot_pos, occupancy_grid, map_resolution):
    """Select best frontier based on distance"""
    if not frontiers:
        return None

    distances = []
    for frontier in frontiers:
        distance = np.hypot(frontier[0] - robot_pos[0], frontier[1] - robot_pos[1])
        distances.append(distance)

    min_index = np.argmin(distances)
    return frontiers[min_index]


def world_to_grid(x, y, resolution):
    """Convert world coordinates to grid coordinates"""
    return int(x / resolution), int(y / resolution)


def grid_to_world(gx, gy, resolution):
    """Convert grid coordinates to world coordinates"""
    return (gx + 0.5) * resolution, (gy + 0.5) * resolution


class AStarPathPlanner:
    """A* path planning algorithm"""

    def __init__(self, resolution):
        self.resolution = resolution

    def plan(self, start, goal, occupancy_grid):
        """Plan path from start to goal using A*"""
        sx, sy = world_to_grid(start[0], start[1], self.resolution)
        gx, gy = world_to_grid(goal[0], goal[1], self.resolution)

        # Check if goal is valid
        if (gx < 0 or gx >= occupancy_grid.shape[1] or
                gy < 0 or gy >= occupancy_grid.shape[0] or
                occupancy_grid[gy, gx] > 0.5):
            return None

        open_set = []
        heapq.heappush(open_set, (0, (sx, sy)))
        came_from = {}
        g_score = {(sx, sy): 0}

        # 8-connected directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == (gx, gy):
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()

                # Convert to world coordinates
                world_path = []
                for x, y in path:
                    wx, wy = grid_to_world(x, y, self.resolution)
                    world_path.append((wx, wy))
                return world_path

            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy

                # Check bounds and obstacles
                if (0 <= nx < occupancy_grid.shape[1] and
                        0 <= ny < occupancy_grid.shape[0] and
                        occupancy_grid[ny, nx] <= 0.5):

                    tentative_g = g_score[current] + np.hypot(dx, dy)

                    if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = tentative_g
                        f_score = tentative_g + np.hypot(gx - nx, gy - ny)
                        heapq.heappush(open_set, (f_score, (nx, ny)))
                        came_from[(nx, ny)] = current

        return None


class OptimizedFrontierExplorer:
    def __init__(self, map_data, resolution=0.1, enable_visualization=True):
        self.resolution = resolution
        self.map_converter = MapConverter(resolution)
        self.lidar_sim = LidarSimulator(range_max=5.0, num_beams=180)
        self.path_planner = AStarPathPlanner(resolution)
        self.enable_visualization = enable_visualization

        # Convert map
        self.true_map = self.map_converter.segments_to_occupancy_grid(
            map_data['segments'], (16, 16)
        )

        # Initialize SLAM map (unknown at start)
        self.slam_map = np.full_like(self.true_map, -1, dtype=np.float32)

        # Initialize robot
        start_pos = map_data['start_point']
        self.robot = Robot(start_pos[0], start_pos[1], 0)

        # Data storage
        self.lidar_data = []
        self.odometry_data = []
        self.robot_path = []
        self.timestamp = 0

        # Exploration parameters
        self.exploration_complete = False
        self.current_target = None
        self.current_path = None
        self.path_index = 0
        self.frontiers = []
        self.current_scan_points = []

        # Visualization setup
        if self.enable_visualization:
            self.setup_visualization()

    def setup_visualization(self):
        """Setup visualization if enabled"""
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.suptitle('动态前沿探索可视化', fontsize=16)

        # Initialize images
        self.true_map_img = self.ax1.imshow(self.true_map, cmap='gray', origin='lower', alpha=0.7)
        self.ax1.set_title('真实地图 (带机器人)')

        self.slam_map_img = self.ax2.imshow(self.slam_map, cmap='gray', origin='lower', vmin=-1, vmax=1)
        self.ax2.set_title('SLAM地图')

        # Robot visualization elements
        self.robot_true = Circle((0, 0), 2, color='red', alpha=0.8)
        self.robot_slam = Circle((0, 0), 2, color='red', alpha=0.8)
        self.robot_arrow_true = None
        self.robot_arrow_slam = None

        self.ax1.add_patch(self.robot_true)
        self.ax2.add_patch(self.robot_slam)

        # Path lines
        self.path_line_true, = self.ax1.plot([], [], 'b-', linewidth=2, alpha=0.7, label='路径')
        self.path_line_slam, = self.ax2.plot([], [], 'b-', linewidth=2, alpha=0.7, label='路径')

        # Frontier points
        self.frontier_points, = self.ax2.plot([], [], 'go', markersize=8, alpha=0.8, label='前沿点')

        # Target point
        self.target_point, = self.ax2.plot([], [], 'r*', markersize=15, label='目标')

        # Planned path
        self.planned_path, = self.ax2.plot([], [], 'r--', linewidth=2, alpha=0.6, label='规划路径')

        # Lidar scan points
        self.scan_points, = self.ax1.plot([], [], 'y.', markersize=1, alpha=0.6, label='激光扫描')

        # Add legends
        self.ax1.legend()
        self.ax2.legend()

        # Set axis limits
        self.ax1.set_xlim(0, self.true_map.shape[1])
        self.ax1.set_ylim(0, self.true_map.shape[0])
        self.ax2.set_xlim(0, self.slam_map.shape[1])
        self.ax2.set_ylim(0, self.slam_map.shape[0])

        # Status text
        self.status_text = self.fig.text(0.5, 0.02, '', ha='center', fontsize=12)

    def update_slam_map(self, lidar_scan):
        """Update SLAM map with lidar data - optimized version"""
        robot_pos = (self.robot.x, self.robot.y, self.robot.theta)

        # 清除之前的扫描点
        self.current_scan_points = []

        for i, (angle, range_val) in enumerate(zip(lidar_scan.angles, lidar_scan.ranges)):
            if range_val >= self.lidar_sim.range_max - 0.1:
                continue

            # Calculate end point of beam
            global_angle = self.robot.theta + angle
            end_x = self.robot.x + range_val * math.cos(global_angle)
            end_y = self.robot.y + range_val * math.sin(global_angle)

            # 保存扫描点用于可视化
            self.current_scan_points.extend([end_x / self.resolution, end_y / self.resolution])

            # Mark cells along the ray as free (simplified)
            self._update_ray_fast(self.robot.x, self.robot.y, end_x, end_y)

            # Mark end point as obstacle
            gx = int(end_x / self.resolution)
            gy = int(end_y / self.resolution)
            if (0 <= gx < self.slam_map.shape[1] and
                    0 <= gy < self.slam_map.shape[0]):
                self.slam_map[gy, gx] = 1.0

    def _update_ray_fast(self, start_x, start_y, end_x, end_y):
        """Fast ray update using simplified line drawing"""
        # Simple line sampling
        distance = np.hypot(end_x - start_x, end_y - start_y)
        num_samples = int(distance / self.resolution) + 1

        for i in range(num_samples):
            ratio = i / max(num_samples - 1, 1)
            x = start_x + ratio * (end_x - start_x)
            y = start_y + ratio * (end_y - start_y)

            gx = int(x / self.resolution)
            gy = int(y / self.resolution)

            if (0 <= gx < self.slam_map.shape[1] and
                    0 <= gy < self.slam_map.shape[0] and
                    self.slam_map[gy, gx] == -1):
                self.slam_map[gy, gx] = 0.0

    def get_next_frontier(self):
        """Get next frontier target"""
        self.frontiers = detect_frontiers_optimized(
            self.slam_map,
            map_resolution=self.resolution
        )

        if not self.frontiers:
            return None

        robot_pos = (self.robot.x, self.robot.y)
        return select_best_frontier(self.frontiers, robot_pos, self.slam_map, self.resolution)

    def navigate_to_target(self, target):
        """Navigate robot to target using A* path planning"""
        if target is None:
            return False

        # Plan path if needed
        if self.current_path is None or self.path_index >= len(self.current_path):
            self.current_path = self.path_planner.plan(
                (self.robot.x, self.robot.y), target, self.slam_map
            )
            self.path_index = 0

            if self.current_path is None:
                return False

        # Follow path
        if self.path_index < len(self.current_path):
            next_waypoint = self.current_path[self.path_index]

            # Move robot to next waypoint
            self.robot.move_to_position(next_waypoint[0], next_waypoint[1])
            self.path_index += 1

            # Check if reached target
            distance_to_target = np.hypot(target[0] - self.robot.x, target[1] - self.robot.y)
            if distance_to_target < self.resolution * 2:
                return True

        return False

    def update_visualization(self):
        """Update all visualization elements"""
        if not self.enable_visualization:
            return

        # Update robot position
        robot_x_grid = self.robot.x / self.resolution
        robot_y_grid = self.robot.y / self.resolution

        self.robot_true.center = (robot_x_grid, robot_y_grid)
        self.robot_slam.center = (robot_x_grid, robot_y_grid)

        # Remove old arrows
        if self.robot_arrow_true:
            self.robot_arrow_true.remove()
        if self.robot_arrow_slam:
            self.robot_arrow_slam.remove()

        # Add new direction arrows
        arrow_length = 5
        dx = arrow_length * math.cos(self.robot.theta)
        dy = arrow_length * math.sin(self.robot.theta)

        self.robot_arrow_true = FancyArrow(robot_x_grid, robot_y_grid, dx, dy,
                                           head_width=2, head_length=2, fc='red', ec='red')
        self.robot_arrow_slam = FancyArrow(robot_x_grid, robot_y_grid, dx, dy,
                                           head_width=2, head_length=2, fc='red', ec='red')

        self.ax1.add_patch(self.robot_arrow_true)
        self.ax2.add_patch(self.robot_arrow_slam)

        # Update path
        if len(self.robot_path) > 1:
            path_x = [p[0] / self.resolution for p in self.robot_path]
            path_y = [p[1] / self.resolution for p in self.robot_path]
            self.path_line_true.set_data(path_x, path_y)
            self.path_line_slam.set_data(path_x, path_y)

        # Update SLAM map
        slam_display = self.slam_map.copy()
        slam_display[slam_display == -1] = 0.5  # Unknown as gray
        self.slam_map_img.set_array(slam_display)

        # Update frontier points
        if self.frontiers:
            frontier_x = [f[0] / self.resolution for f in self.frontiers]
            frontier_y = [f[1] / self.resolution for f in self.frontiers]
            self.frontier_points.set_data(frontier_x, frontier_y)
        else:
            self.frontier_points.set_data([], [])

        # Update target point
        if self.current_target:
            target_x = self.current_target[0] / self.resolution
            target_y = self.current_target[1] / self.resolution
            self.target_point.set_data([target_x], [target_y])
        else:
            self.target_point.set_data([], [])

        # Update planned path
        if self.current_path and len(self.current_path) > 1:
            path_x = [p[0] / self.resolution for p in self.current_path]
            path_y = [p[1] / self.resolution for p in self.current_path]
            self.planned_path.set_data(path_x, path_y)
        else:
            self.planned_path.set_data([], [])

        # Update lidar scan points
        if len(self.current_scan_points) > 1:
            scan_x = self.current_scan_points[::2]
            scan_y = self.current_scan_points[1::2]
            self.scan_points.set_data(scan_x, scan_y)
        else:
            self.scan_points.set_data([], [])

        # Update status text
        num_frontiers = len(self.frontiers) if self.frontiers else 0
        unknown_ratio = np.sum(self.slam_map == -1) / self.slam_map.size
        status = f"步骤: {len(self.robot_path)} | 机器人位置: ({self.robot.x:.2f}, {self.robot.y:.2f}) | "
        status += f"前沿点数: {num_frontiers} | 未知区域: {unknown_ratio:.1%} | "
        status += f"目标: {'有' if self.current_target else '无'} | "
        status += f"时间: {self.timestamp:.1f}s"
        self.status_text.set_text(status)

        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def explore_step(self, dt=0.1):
        """Single exploration step - optimized"""
        # Record current position
        self.robot_path.append([self.robot.x, self.robot.y])

        # Simulate lidar scan
        robot_pos = (self.robot.x, self.robot.y, self.robot.theta)
        lidar_scan = self.lidar_sim.scan(robot_pos, self.true_map, self.resolution)
        lidar_scan.timestamp = self.timestamp
        self.lidar_data.append(lidar_scan)

        # Record odometry
        odom = Odometry(self.robot.x, self.robot.y, self.robot.theta, self.timestamp)
        self.odometry_data.append(odom)

        # Update SLAM map
        self.update_slam_map(lidar_scan)

        # Check if we need a new target
        if self.current_target is None:
            self.current_target = self.get_next_frontier()
            self.current_path = None

        # Navigate to target
        if self.current_target:
            reached = self.navigate_to_target(self.current_target)
            if reached:
                self.current_target = None
                self.current_path = None
        else:
            # Check if exploration is complete
            unknown_ratio = np.sum(self.slam_map == -1) / self.slam_map.size
            if unknown_ratio < 0.05:
                self.exploration_complete = True

        self.timestamp += dt

        # Update visualization每一步都更新
        if self.enable_visualization:
            self.update_visualization()

        return not self.exploration_complete

    def run_exploration(self, max_steps=1000, delay=0.05):
        """Run exploration with visualization"""
        print("开始优化的前沿探索...")
        print("按 Ctrl+C 可以提前停止")

        step = 0
        start_time = time.time()

        try:
            while step < max_steps and self.explore_step():
                step += 1

                # Add delay for visualization
                if self.enable_visualization and delay > 0:
                    time.sleep(delay)

                # Print progress every 50 steps
                if step % 50 == 0:
                    unknown_ratio = np.sum(self.slam_map == -1) / self.slam_map.size
                    elapsed_time = time.time() - start_time
                    print(f"步骤 {step}: 机器人位置 ({self.robot.x:.2f}, {self.robot.y:.2f}), "
                          f"未知区域: {unknown_ratio:.1%}, 用时: {elapsed_time:.1f}秒")

        except KeyboardInterrupt:
            print("用户停止了探索")

        elapsed_time = time.time() - start_time
        unknown_ratio = np.sum(self.slam_map == -1) / self.slam_map.size

        print(f"探索完成！用了 {step} 步, {elapsed_time:.1f} 秒")
        print(f"最终未知区域比例: {unknown_ratio:.1%}")

        if self.enable_visualization:
            plt.ioff()
            self.update_visualization()
            plt.show()

        return self.get_exploration_data()

    def run_fast_exploration(self, max_steps=1000):
        """Run exploration without visualization for maximum speed"""
        print("开始快速探索（无可视化）...")

        step = 0
        start_time = time.time()

        try:
            while step < max_steps and self.explore_step():
                step += 1

                # Print progress every 100 steps
                if step % 100 == 0:
                    unknown_ratio = np.sum(self.slam_map == -1) / self.slam_map.size
                    elapsed_time = time.time() - start_time
                    print(f"步骤 {step}: 机器人位置 ({self.robot.x:.2f}, {self.robot.y:.2f}), "
                          f"未知区域: {unknown_ratio:.1%}, 用时: {elapsed_time:.1f}秒")

        except KeyboardInterrupt:
            print("用户停止了探索")

        elapsed_time = time.time() - start_time
        unknown_ratio = np.sum(self.slam_map == -1) / self.slam_map.size

        print(f"探索完成！用了 {step} 步, {elapsed_time:.1f} 秒")
        print(f"最终未知区域比例: {unknown_ratio:.1%}")

        return self.get_exploration_data()

    def get_exploration_data(self):
        """Get exploration data in JSON format"""
        exploration_data = {
            "metadata": {
                "total_steps": len(self.lidar_data),
                "total_time": self.timestamp,
                "map_resolution": self.resolution,
                "final_unknown_ratio": float(np.sum(self.slam_map == -1) / self.slam_map.size)
            },
            "lidar_scans": [],
            "odometry": [],
            "final_map": self.slam_map.tolist(),
            "robot_path": self.robot_path
        }

        for scan in self.lidar_data:
            exploration_data["lidar_scans"].append({
                "ranges": scan.ranges,
                "angles": scan.angles,
                "timestamp": scan.timestamp
            })

        for odo in self.odometry_data:
            exploration_data["odometry"].append({
                "x": odo.x,
                "y": odo.y,
                "theta": odo.theta,
                "timestamp": odo.timestamp
            })

        return exploration_data


# 使用示例
if __name__ == "__main__":
    # 地图数据
    map_data = {
        "segments": [
            {"start": [0, 0], "end": [2, 0]},
            {"start": [2, 0], "end": [2, 2]},
            {"start": [0, 0], "end": [0, 15]},
            {"start": [0, 11], "end": [2, 11]},
            {"start": [2, 11], "end": [2, 6]},
            {"start": [2, 6], "end": [4, 6]},
            {"start": [0, 15], "end": [11, 15]},
            {"start": [2, 15], "end": [2, 13]},
            {"start": [2, 13], "end": [9, 13]},
            {"start": [4, 13], "end": [4, 8]},
            {"start": [6, 13], "end": [6, 10]},
            {"start": [6, 10], "end": [9, 10]},
            {"start": [9, 10], "end": [9, 13]},
            {"start": [11, 15], "end": [11, 10]},
            {"start": [4, 0], "end": [4, 2]},
            {"start": [4, 0], "end": [15, 0]},
            {"start": [11, 0], "end": [11, 2]},
            {"start": [11, 2], "end": [6, 2]},
            {"start": [6, 2], "end": [6, 6]},
            {"start": [6, 4], "end": [2, 4]},
            {"start": [9, 2], "end": [9, 4]},
            {"start": [15, 0], "end": [15, 15]},
            {"start": [15, 6], "end": [13, 6]},
            {"start": [13, 6], "end": [13, 2]},
            {"start": [15, 15], "end": [13, 15]},
            {"start": [13, 15], "end": [13, 8]},
            {"start": [13, 8], "end": [11, 8]}
        ],
        "start_point": [3, 0]
    }

    # 可视化模式 - 可以观察探索过程
    print("=== 可视化模式 ===")
    explorer = OptimizedFrontierExplorer(map_data, resolution=0.1, enable_visualization=True)

    # 运行探索 (delay=0.05 控制速度，0为最快，0.1为较慢)
    exploration_data = explorer.run_exploration(max_steps=800, delay=0.05)

    # 保存数据
    with open('优化探索数据_可视化.json', 'w', encoding='utf-8') as f:
        json.dump(exploration_data, f, indent=2, ensure_ascii=False)

    print("探索数据已保存到 优化探索数据_可视化.json")
    print(f"收集了 {len(exploration_data['lidar_scans'])} 个激光扫描数据")
    print(f"收集了 {len(exploration_data['odometry'])} 个里程计数据")

    # 如果想要快速运行模式，取消下面注释：
    # print("\n=== 快速模式 ===")
    # explorer_fast = OptimizedFrontierExplorer(map_data, resolution=0.1, enable_visualization=False)
    # exploration_data_fast = explorer_fast.run_fast_exploration(max_steps=800)
    # with open('优化探索数据_快速.json', 'w', encoding='utf-8') as f:
    #     json.dump(exploration_data_fast, f, indent=2, ensure_ascii=False)