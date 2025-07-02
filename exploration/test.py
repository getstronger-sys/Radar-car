import numpy as np
import json
import math
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrow, Rectangle
from scipy.ndimage import label, binary_dilation
import time
import heapq
from config.map import get_global_map, MAP_RESOLUTION
from config.settings import START_POSITION


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
        self.radius = 0.15

    def is_position_safe(self, x, y, occupancy_grid, resolution, safety_margin=0.15):
        """Check if position is safe (no collision with obstacles)"""
        margin_cells = int(safety_margin / resolution)
        gx = int(x / resolution)
        gy = int(y / resolution)

        for dx in range(-margin_cells, margin_cells + 1):
            for dy in range(-margin_cells, margin_cells + 1):
                check_x = gx + dx
                check_y = gy + dy

                if (check_x < 0 or check_x >= occupancy_grid.shape[1] or
                        check_y < 0 or check_y >= occupancy_grid.shape[0]):
                    return False

                if occupancy_grid[check_y, check_x] > 0.5:
                    return False

        return True

    def move_to_position(self, target_x, target_y, occupancy_grid=None, resolution=None):
        """Safely move to target position"""
        if occupancy_grid is not None and resolution is not None:
            if not self.is_position_safe(target_x, target_y, occupancy_grid, resolution):
                return False

        if hasattr(self, 'last_x') and hasattr(self, 'last_y'):
            dx = target_x - self.last_x
            dy = target_y - self.last_y
            if abs(dx) > 0.001 or abs(dy) > 0.001:
                self.theta = math.atan2(dy, dx)

        self.last_x = self.x
        self.last_y = self.y
        self.x = target_x
        self.y = target_y
        return True

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
        """Convert line segments to occupancy grid with boundaries"""
        width = int(map_size[0] / self.resolution)
        height = int(map_size[1] / self.resolution)

        grid = np.zeros((height, width), dtype=np.float32)
        self._add_boundary_walls(grid, map_size)

        for segment in segments:
            self._draw_line(grid, segment['start'], segment['end'])

        return grid

    def _add_boundary_walls(self, grid, map_size):
        """Add boundary walls around the map"""
        height, width = grid.shape
        grid[0, :] = 1.0  # Top boundary
        grid[height - 1, :] = 1.0  # Bottom boundary
        grid[:, 0] = 1.0  # Left boundary
        grid[:, width - 1] = 1.0  # Right boundary

    def _draw_line(self, grid, start, end):
        """Draw a line on the grid using Bresenham's algorithm"""
        x0 = int(start[0] / self.resolution)
        y0 = int(start[1] / self.resolution)
        x1 = int(end[0] / self.resolution)
        y1 = int(end[1] / self.resolution)

        points = self._bresenham_line(x0, y0, x1, y1)

        for x, y in points:
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                grid[y, x] = 1.0

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
        """Simulate lidar scan"""
        ranges = []
        robot_x, robot_y, robot_theta = robot_pos

        for angle in self.angles:
            global_angle = robot_theta + angle
            range_val = self._cast_ray_fast(
                robot_x, robot_y, global_angle,
                occupancy_grid, resolution
            )
            ranges.append(range_val)

        return LidarScan(ranges, self.angles.tolist(), 0)

    def _cast_ray_fast(self, start_x, start_y, angle, grid, resolution):
        """Fast ray casting"""
        step_size = resolution * 0.5
        max_steps = int(self.range_max / step_size)

        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        for step in range(max_steps):
            x = start_x + step * step_size * cos_a
            y = start_y + step * step_size * sin_a

            gx = int(x / resolution)
            gy = int(y / resolution)

            if (gx < 0 or gx >= grid.shape[1] or
                    gy < 0 or gy >= grid.shape[0]):
                return step * step_size

            if grid[gy, gx] > 0.5:
                return step * step_size

        return self.range_max


def detect_frontiers_optimized(occupancy_grid, unknown_val=-1, free_threshold=0.2, map_resolution=0.01):
    """Optimized frontier detection function"""
    h, w = occupancy_grid.shape

    free_mask = (occupancy_grid >= 0) & (occupancy_grid < free_threshold)
    unknown_mask = (occupancy_grid == unknown_val)

    dilated_unknown = binary_dilation(unknown_mask, structure=np.ones((3, 3)))
    frontiers_mask = free_mask & dilated_unknown
    labeled, num_features = label(frontiers_mask)

    frontiers_world = []

    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)

        if len(xs) > 3:
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))
            x_m = cx * map_resolution
            y_m = cy * map_resolution
            frontiers_world.append((x_m, y_m))

    return frontiers_world


def select_best_frontier(frontiers, robot_pos, occupancy_grid, map_resolution, path_planner=None):
    """Select best frontier based on path distance"""
    if not frontiers:
        return None

    distances = []
    valid_frontiers = []

    for frontier in frontiers:
        if path_planner:
            path = path_planner.plan(robot_pos, frontier, occupancy_grid)
            if path:
                path_length = 0
                for i in range(1, len(path)):
                    dx = path[i][0] - path[i - 1][0]
                    dy = path[i][1] - path[i - 1][1]
                    path_length += np.hypot(dx, dy)
                distances.append(path_length)
                valid_frontiers.append(frontier)
            else:
                distances.append(float('inf'))
                valid_frontiers.append(frontier)
        else:
            distance = np.hypot(frontier[0] - robot_pos[0], frontier[1] - robot_pos[1])
            distances.append(distance)
            valid_frontiers.append(frontier)

    if not valid_frontiers:
        return None

    min_index = np.argmin(distances)
    return valid_frontiers[min_index]


def world_to_grid(x, y, resolution):
    """Convert world coordinates to grid coordinates"""
    return int(x / resolution), int(y / resolution)


def grid_to_world(gx, gy, resolution):
    """Convert grid coordinates to world coordinates"""
    return (gx + 0.5) * resolution, (gy + 0.5) * resolution


class AStarPathPlanner:
    """A* path planning algorithm with safety margins"""

    def __init__(self, resolution, safety_margin=0.15):
        self.resolution = resolution
        self.safety_margin = safety_margin

    def is_valid_position(self, gx, gy, occupancy_grid):
        """Check if position is valid with safety margin"""
        margin_cells = int(self.safety_margin / self.resolution)

        for dx in range(-margin_cells, margin_cells + 1):
            for dy in range(-margin_cells, margin_cells + 1):
                check_x = gx + dx
                check_y = gy + dy

                if (check_x < 0 or check_x >= occupancy_grid.shape[1] or
                        check_y < 0 or check_y >= occupancy_grid.shape[0]):
                    return False

                if occupancy_grid[check_y, check_x] > 0.5:
                    return False

        return True

    def plan(self, start, goal, occupancy_grid):
        """Plan path from start to goal using A*"""
        sx, sy = world_to_grid(start[0], start[1], self.resolution)
        gx, gy = world_to_grid(goal[0], goal[1], self.resolution)

        if not self.is_valid_position(sx, sy, occupancy_grid):
            return None
        if not self.is_valid_position(gx, gy, occupancy_grid):
            return None

        open_set = []
        heapq.heappush(open_set, (0, (sx, sy)))
        came_from = {}
        g_score = {(sx, sy): 0}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == (gx, gy):
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()

                world_path = []
                for x, y in path:
                    wx, wy = grid_to_world(x, y, self.resolution)
                    world_path.append((wx, wy))
                return world_path

            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy

                if self.is_valid_position(nx, ny, occupancy_grid):
                    tentative_g = g_score[current] + np.hypot(dx, dy)

                    if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = tentative_g
                        f_score = tentative_g + np.hypot(gx - nx, gy - ny)
                        heapq.heappush(open_set, (f_score, (nx, ny)))
                        came_from[(nx, ny)] = current

        return None


class SimplifiedFrontierExplorer:
    def __init__(self, enable_visualization=True):
        # 直接用全局地图和分辨率
        self.resolution = MAP_RESOLUTION
        self.true_map = get_global_map()

        # SLAM map 初始化为未知
        self.slam_map = np.full_like(self.true_map, -1, dtype=np.float32)

        # 机器人起点
        start_pos = [START_POSITION['x'], START_POSITION['y']]
        self.robot = Robot(start_pos[0], start_pos[1], 0)

        # 其它初始化保持不变
        self.lidar_data = []
        self.odometry_data = []
        self.robot_path = []
        self.timestamp = 0
        self.exploration_complete = False
        self.current_target = None
        self.current_path = None
        self.path_index = 0
        self.frontiers = []
        self.current_scan_points = []
        self.enable_visualization = enable_visualization
        self.map_converter = MapConverter(self.resolution)
        self.lidar_sim = LidarSimulator(range_max=5.0, num_beams=180)
        self.path_planner = AStarPathPlanner(self.resolution, safety_margin=0.15)
        if self.enable_visualization:
            self.setup_visualization()

    def setup_visualization(self):
        """Setup simplified visualization with two windows"""
        plt.style.use('dark_background')
        plt.ion()

        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor('#1e1e1e')

        # Create custom grid layout - 2x2 instead of 2x3
        gs = self.fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[1, 1],
                                   hspace=0.3, wspace=0.3)

        # Main map displays - only two windows
        self.ax1 = self.fig.add_subplot(gs[0, 0])  # Ground Truth
        self.ax2 = self.fig.add_subplot(gs[0, 1])  # SLAM Mapping

        # Status panel spans both columns
        self.ax_status = self.fig.add_subplot(gs[1, :])
        self.ax_status.axis('off')

        # Set titles with modern styling
        title_style = {'fontsize': 16, 'fontweight': 'bold', 'color': '#00ff88', 'pad': 20}
        self.ax1.set_title('🌍 Ground Truth Environment', **title_style)
        self.ax2.set_title('🗺️ SLAM Mapping & Exploration', **title_style)

        # Set background colors
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor('#2a2a2a')
            ax.grid(True, alpha=0.3, color='#666666', linewidth=0.5)
            ax.set_xlabel('X Position (m)', color='#cccccc', fontsize=12)
            ax.set_ylabel('Y Position (m)', color='#cccccc', fontsize=12)

        # Initialize map images with custom colormaps
        self.true_map_img = self.ax1.imshow(self.true_map, cmap='binary', origin='lower', alpha=0.8)

        # Custom colormap for SLAM map
        slam_cmap = plt.get_cmap('RdYlGn_r')
        self.slam_map_img = self.ax2.imshow(self.slam_map, cmap=slam_cmap, origin='lower',
                                            vmin=-1, vmax=1, alpha=0.9)

        # Robot visualization with enhanced styling
        robot_color = '#ff4444'
        robot_radius_grid = self.robot.radius / self.resolution
        self.robot_true = Circle((0, 0), robot_radius_grid, color=robot_color, alpha=0.9, linewidth=2,
                                 edgecolor='white', zorder=10)
        self.robot_slam = Circle((0, 0), robot_radius_grid, color=robot_color, alpha=0.9, linewidth=2,
                                 edgecolor='white', zorder=10)

        self.ax1.add_patch(self.robot_true)
        self.ax2.add_patch(self.robot_slam)

        # Enhanced path visualization
        self.path_line_true, = self.ax1.plot([], [], color='#00aaff', linewidth=3,
                                             alpha=0.8, label='Robot Path', zorder=5)
        self.path_line_slam, = self.ax2.plot([], [], color='#00aaff', linewidth=3,
                                             alpha=0.8, label='Robot Path', zorder=5)

        # Frontier and target visualization (only on SLAM map)
        self.frontier_points, = self.ax2.plot([], [], 'o', color='#00ff88', markersize=10,
                                              alpha=0.9, label='Frontier Points', zorder=8)

        self.target_point, = self.ax2.plot([], [], '*', color='#ffff00', markersize=8,
                                           label='Current Target', zorder=9)

        # Planned path visualization (only on SLAM map)
        self.planned_path, = self.ax2.plot([], [], '--', color='#ff8800', linewidth=4,
                                           alpha=0.8, label='Planned Path', zorder=7)

        # LiDAR scan visualization (only on ground truth)
        self.scan_points, = self.ax1.plot([], [], '.', color='#ffff44', markersize=2,
                                          alpha=0.8, label='LiDAR Scan', zorder=3)

        # Enhanced legends
        legend_style = {'fancybox': True, 'framealpha': 0.9, 'facecolor': '#2a2a2a',
                        'edgecolor': '#666666', 'fontsize': 10}
        self.ax1.legend(loc='upper right', **legend_style)
        self.ax2.legend(loc='upper right', **legend_style)

        # Set axis limits and styling
        for ax in [self.ax1, self.ax2]:
            ax.set_xlim(0, self.true_map.shape[1])
            ax.set_ylim(0, self.true_map.shape[0])
            ax.tick_params(colors='#cccccc', labelsize=10)

        # Initialize arrow placeholders
        self.robot_arrow_true = None
        self.robot_arrow_slam = None

        # Status display setup
        self.setup_status_display()

    def setup_status_display(self):
        """Setup enhanced status display panel"""
        # Create status text areas
        self.status_texts = {}

        # Main status line
        self.main_status = self.ax_status.text(0.5, 0.8, '', ha='center', va='center',
                                               fontsize=18, fontweight='bold', color='#00ff88',
                                               transform=self.ax_status.transAxes)

        # Statistics grid
        stats_x_positions = [0.1, 0.3, 0.5, 0.7, 0.9]
        stats_labels = ['Steps', 'Position', 'Frontiers', 'Unknown', 'Time']
        self.stats_labels = []
        self.stats_values = []

        for i, (x_pos, label) in enumerate(zip(stats_x_positions, stats_labels)):
            # Label
            label_text = self.ax_status.text(x_pos, 0.5, label, ha='center', va='top',
                                             fontsize=12, fontweight='bold', color='#cccccc',
                                             transform=self.ax_status.transAxes)
            self.stats_labels.append(label_text)

            # Value
            value_text = self.ax_status.text(x_pos, 0.2, '---', ha='center', va='top',
                                             fontsize=14, fontweight='bold', color='#ffffff',
                                             transform=self.ax_status.transAxes)
            self.stats_values.append(value_text)

        # Progress bar background
        self.progress_bg = Rectangle((0.05, 0.05), 0.9, 0.1, transform=self.ax_status.transAxes,
                                     facecolor='#444444', edgecolor='#666666', linewidth=1)
        self.ax_status.add_patch(self.progress_bg)

        # Progress bar
        self.progress_bar = Rectangle((0.05, 0.05), 0.0, 0.1, transform=self.ax_status.transAxes,
                                      facecolor='#00ff88', alpha=0.8)
        self.ax_status.add_patch(self.progress_bar)

    def update_slam_map(self, lidar_scan):
        """Update SLAM map with lidar data"""
        self.current_scan_points = []

        for i, (angle, range_val) in enumerate(zip(lidar_scan.angles, lidar_scan.ranges)):
            global_angle = self.robot.theta + angle
            end_x = self.robot.x + range_val * math.cos(global_angle)
            end_y = self.robot.y + range_val * math.sin(global_angle)

            self.current_scan_points.extend([end_x / self.resolution, end_y / self.resolution])
            self._update_ray_fast(self.robot.x, self.robot.y, end_x, end_y)

            if range_val < self.lidar_sim.range_max - 0.1:
                gx = int(end_x / self.resolution)
                gy = int(end_y / self.resolution)
                if (0 <= gx < self.slam_map.shape[1] and
                        0 <= gy < self.slam_map.shape[0]):
                    self.slam_map[gy, gx] = 1.0

    def _update_ray_fast(self, start_x, start_y, end_x, end_y):
        """Fast ray update using simplified line drawing"""
        distance = np.hypot(end_x - start_x, end_y - start_y)
        num_samples = int(distance / self.resolution) + 1

        # 确保 num_samples 至少为1
        if num_samples < 1:
            num_samples = 1

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
        return select_best_frontier(self.frontiers, robot_pos, self.slam_map,
                                    self.resolution, self.path_planner)

    def find_nearest_unknown_area(self):
        """Find the nearest unknown area when no frontiers are available"""
        unknown_y, unknown_x = np.where(self.slam_map == -1)

        if len(unknown_x) == 0:
            return None

        robot_x_grid = self.robot.x / self.resolution
        robot_y_grid = self.robot.y / self.resolution

        distances = np.sqrt((unknown_x - robot_x_grid) ** 2 + (unknown_y - robot_y_grid) ** 2)

        nearest_idx = np.argmin(distances)
        nearest_x = unknown_x[nearest_idx]
        nearest_y = unknown_y[nearest_idx]

        world_x = nearest_x * self.resolution
        world_y = nearest_y * self.resolution

        return (world_x, world_y)

    def navigate_to_target(self, target):
        """Navigate robot to target using A* path planning"""
        if target is None:
            return False

        if self.current_path is None or self.path_index >= len(self.current_path):
            self.current_path = self.path_planner.plan(
                (self.robot.x, self.robot.y), target, self.slam_map
            )
            self.path_index = 0

            if self.current_path is None:
                print(f"⚠️  Path planning failed to target {target}, abandoning target")
                return True

        if self.path_index < len(self.current_path):
            next_waypoint = self.current_path[self.path_index]

            success = self.robot.move_to_position(
                next_waypoint[0], next_waypoint[1],
                self.slam_map, self.resolution
            )

            if success:
                self.path_index += 1
            else:
                print(f"⚠️  Movement failed to {next_waypoint}, replanning path")
                self.current_path = None
                return False

            distance_to_target = np.hypot(target[0] - self.robot.x, target[1] - self.robot.y)
            if distance_to_target < self.resolution * 2:
                return True

        return False

    def update_visualization(self):
        """Update visualization elements"""
        if not self.enable_visualization:
            return

        # Update robot positions
        robot_x_grid = self.robot.x / self.resolution
        robot_y_grid = self.robot.y / self.resolution

        self.robot_true.center = (robot_x_grid, robot_y_grid)
        self.robot_slam.center = (robot_x_grid, robot_y_grid)

        # Update robot direction arrows
        self._update_robot_arrows(robot_x_grid, robot_y_grid)

        # Update paths
        if len(self.robot_path) > 1:
            path_x = [p[0] / self.resolution for p in self.robot_path]
            path_y = [p[1] / self.resolution for p in self.robot_path]

            self.path_line_true.set_data(path_x, path_y)
            self.path_line_slam.set_data(path_x, path_y)

        # Update SLAM map
        slam_display = self.slam_map.copy()
        slam_display[slam_display == -1] = 0.5  # Unknown as neutral
        self.slam_map_img.set_array(slam_display)

        # Update frontiers
        self._update_frontiers()

        # Update target
        self._update_target()

        # Update planned path
        self._update_planned_path()

        # Update LiDAR scan
        self._update_lidar_scan()

        # Update status display
        self._update_status_display()

        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _update_robot_arrows(self, robot_x_grid, robot_y_grid):
        """Update robot direction arrows"""
        # Remove old arrows
        if self.robot_arrow_true:
            self.robot_arrow_true.remove()
        if self.robot_arrow_slam:
            self.robot_arrow_slam.remove()

        # Add new arrows
        arrow_length = self.robot.radius / self.resolution * 2  # 箭头长度为小车半径的2倍（单位：格）
        dx = arrow_length * math.cos(self.robot.theta)
        dy = arrow_length * math.sin(self.robot.theta)

        # 让箭头头部宽度和长度也随箭头长度缩放
        head_width = arrow_length * 0.5   # 你可以调节这个系数
        head_length = arrow_length * 0.4  # 你可以调节这个系数

        arrow_style = dict(
            head_width=head_width,
            head_length=head_length,
            fc='#ffffff',
            ec='#ffffff',
            linewidth=2.0,
            alpha=0.9,
            zorder=15
        )

        self.robot_arrow_true = FancyArrow(robot_x_grid, robot_y_grid, dx, dy, **arrow_style)
        self.robot_arrow_slam = FancyArrow(robot_x_grid, robot_y_grid, dx, dy, **arrow_style)

        self.ax1.add_patch(self.robot_arrow_true)
        self.ax2.add_patch(self.robot_arrow_slam)

    def _update_frontiers(self):
        """Update frontier visualization"""
        if self.frontiers:
            frontier_x = [f[0] / self.resolution for f in self.frontiers]
            frontier_y = [f[1] / self.resolution for f in self.frontiers]
            self.frontier_points.set_data(frontier_x, frontier_y)
        else:
            self.frontier_points.set_data([], [])

    def _update_target(self):
        """Update target visualization"""
        if self.current_target:
            target_x = self.current_target[0] / self.resolution
            target_y = self.current_target[1] / self.resolution
            self.target_point.set_data([target_x], [target_y])
        else:
            self.target_point.set_data([], [])

    def _update_planned_path(self):
        """Update planned path visualization"""
        if self.current_path and len(self.current_path) > 1:
            path_x = [p[0] / self.resolution for p in self.current_path]
            path_y = [p[1] / self.resolution for p in self.current_path]
            self.planned_path.set_data(path_x, path_y)
        else:
            self.planned_path.set_data([], [])

    def _update_lidar_scan(self):
        """Update LiDAR scan visualization"""
        if len(self.current_scan_points) > 1:
            scan_x = self.current_scan_points[::2]
            scan_y = self.current_scan_points[1::2]
            self.scan_points.set_data(scan_x, scan_y)
        else:
            self.scan_points.set_data([], [])

    def _update_status_display(self):
        """Update status display with current information"""
        # Main status
        num_frontiers = len(self.frontiers) if self.frontiers else 0
        unknown_ratio = np.sum(self.slam_map == -1) / self.slam_map.size

        if self.exploration_complete:
            status = "🎉 EXPLORATION COMPLETE!"
            self.main_status.set_color('#00ff88')
        elif self.current_target:
            status = "🎯 ACTIVELY EXPLORING"
            self.main_status.set_color('#ffff00')
        else:
            status = "🔍 SEARCHING FOR TARGETS"
            self.main_status.set_color('#ff8800')

        self.main_status.set_text(status)

        # Update statistics
        stats_data = [
            len(self.robot_path),
            f"({self.robot.x:.1f}, {self.robot.y:.1f})",
            num_frontiers,
            f"{unknown_ratio:.1%}",
            f"{self.timestamp:.1f}s"
        ]

        for i, (value_text, data) in enumerate(zip(self.stats_values, stats_data)):
            value_text.set_text(str(data))

        # Update progress bar
        progress = 1.0 - unknown_ratio
        self.progress_bar.set_width(0.9 * progress)

        # Color code progress
        if progress > 0.8:
            self.progress_bar.set_facecolor('#00ff88')
        elif progress > 0.5:
            self.progress_bar.set_facecolor('#ffff00')
        else:
            self.progress_bar.set_facecolor('#ff8800')

    def explore_step(self, dt=0.1):
        """Single exploration step"""
        self.robot_path.append([self.robot.x, self.robot.y])

        robot_pos = (self.robot.x, self.robot.y, self.robot.theta)
        lidar_scan = self.lidar_sim.scan(robot_pos, self.true_map, self.resolution)
        lidar_scan.timestamp = self.timestamp
        self.lidar_data.append(lidar_scan)

        odom = Odometry(self.robot.x, self.robot.y, self.robot.theta, self.timestamp)
        self.odometry_data.append(odom)

        self.update_slam_map(lidar_scan)

        if self.current_target is None:
            self.current_target = self.get_next_frontier()
            self.current_path = None

        if self.current_target:
            reached = self.navigate_to_target(self.current_target)
            if reached:
                self.current_target = None
                self.current_path = None
        else:
            self.current_target = self.get_next_frontier()
            if self.current_target is None:
                self.current_target = self.find_nearest_unknown_area()
                if self.current_target is None:
                    self.exploration_complete = True
                else:
                    unknown_ratio = np.sum(self.slam_map == -1) / self.slam_map.size
                    print(f"🔍 No frontiers found, exploring nearest unknown area: {self.current_target}, "
                          f"Unknown: {unknown_ratio:.1%}")

        self.timestamp += dt

        if self.enable_visualization:
            self.update_visualization()

        return not self.exploration_complete

    def run_exploration(self, max_steps=1000, delay=0.05):
        """Run exploration with enhanced visualization"""
        print("🚀 Starting Simplified Frontier Exploration...")
        print("💡 Press Ctrl+C to stop early")

        step = 0
        start_time = time.time()

        try:
            while step < max_steps and self.explore_step():
                step += 1

                if self.enable_visualization and delay > 0:
                    time.sleep(delay)

                if step % 50 == 0:
                    unknown_ratio = np.sum(self.slam_map == -1) / self.slam_map.size
                    elapsed_time = time.time() - start_time
                    print(f"📊 Step {step}: Position ({self.robot.x:.2f}, {self.robot.y:.2f}), "
                          f"Unknown: {unknown_ratio:.1%}, Time: {elapsed_time:.1f}s")

        except KeyboardInterrupt:
            print("⏹️  Exploration stopped by user")

        elapsed_time = time.time() - start_time
        unknown_ratio = np.sum(self.slam_map == -1) / self.slam_map.size

        print(f"✅ Exploration completed in {step} steps, {elapsed_time:.1f} seconds")
        print(f"📈 Final unknown area ratio: {unknown_ratio:.1%}")

        if self.enable_visualization:
            plt.ioff()
            self.update_visualization()
            print("🖼️  Visualization window will remain open")
            plt.show()

        return self.get_exploration_data()

    def run_fast_exploration(self, max_steps=1000):
        """Run exploration without visualization for maximum speed"""
        print("⚡ Starting Fast Exploration (No Visualization)...")

        step = 0
        start_time = time.time()

        try:
            while step < max_steps and self.explore_step():
                step += 1

                if step % 100 == 0:
                    unknown_ratio = np.sum(self.slam_map == -1) / self.slam_map.size
                    elapsed_time = time.time() - start_time
                    print(f"📊 Step {step}: Position ({self.robot.x:.2f}, {self.robot.y:.2f}), "
                          f"Unknown: {unknown_ratio:.1%}, Time: {elapsed_time:.1f}s")

        except KeyboardInterrupt:
            print("⏹️  Exploration stopped by user")

        elapsed_time = time.time() - start_time
        unknown_ratio = np.sum(self.slam_map == -1) / self.slam_map.size

        print(f"✅ Fast exploration completed in {step} steps, {elapsed_time:.1f} seconds")
        print(f"📈 Final unknown area ratio: {unknown_ratio:.1%}")

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


# Example usage
if __name__ == "__main__":
    print("🎨 === Simplified Frontier Exploration with Dual Display ===")
    explorer = SimplifiedFrontierExplorer(enable_visualization=True)
    exploration_data = explorer.run_exploration(max_steps=500, delay=0.02)
    with open('simplified_exploration_data.json', 'w', encoding='utf-8') as f:
        json.dump(exploration_data, f, indent=2, ensure_ascii=False)
    print("💾 Exploration data saved to 'simplified_exploration_data.json'")
    print(f"📊 Collected {len(exploration_data['lidar_scans'])} LiDAR scans")
    print(f"📊 Collected {len(exploration_data['odometry'])} odometry readings")