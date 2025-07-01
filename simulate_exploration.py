import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from config.map import get_global_map, MAP_SIZE, MAP_RESOLUTION
from config.settings import START_POSITION, EXIT_POSITION, ROBOT_RADIUS
from exploration.frontier_detect import ExplorationManager, world_to_grid, grid_to_world, is_exploration_complete
import heapq
import json

# 激光参数
LIDAR_ANGLE_RES = 2  # 每2度一束
LIDAR_NUM = 360 // LIDAR_ANGLE_RES
LIDAR_MAX_DIST = 10.0  # 米

# 地图初始化
true_map = get_global_map()  # 0空地 1障碍
known_map = np.full_like(true_map, -1, dtype=float)  # -1未知 0空地 1障碍

# 小车初始化
robot_pos = np.array([START_POSITION['x'], START_POSITION['y']])
robot_theta = START_POSITION['theta']
trajectory = [robot_pos.copy()]

# 探索管理器
explorer = ExplorationManager(map_resolution=MAP_RESOLUTION)

# 激光扫描模拟

def simulate_lidar(pos, theta, true_map):
    scan = np.zeros(LIDAR_NUM)
    for i in range(LIDAR_NUM):
        angle = np.deg2rad(i * LIDAR_ANGLE_RES)
        a = theta + angle
        for r in np.arange(0, LIDAR_MAX_DIST, MAP_RESOLUTION/2):
            x = pos[0] + r * np.cos(a)
            y = pos[1] + r * np.sin(a)
            gx, gy = world_to_grid(x, y, MAP_RESOLUTION)
            if gx < 0 or gx >= MAP_SIZE or gy < 0 or gy >= MAP_SIZE:
                scan[i] = r
                break
            if true_map[gy, gx] == 1:
                scan[i] = r
                break
        else:
            scan[i] = LIDAR_MAX_DIST
    return scan

# 用激光更新已知地图

def update_known_map(pos, scan, known_map):
    for i, dist in enumerate(scan):
        angle = np.deg2rad(i * LIDAR_ANGLE_RES)
        a = robot_theta + angle
        for r in np.arange(0, min(dist, LIDAR_MAX_DIST), MAP_RESOLUTION/2):
            x = pos[0] + r * np.cos(a)
            y = pos[1] + r * np.sin(a)
            gx, gy = world_to_grid(x, y, MAP_RESOLUTION)
            if gx < 0 or gx >= MAP_SIZE or gy < 0 or gy >= MAP_SIZE:
                break
            if known_map[gy, gx] == -1:
                known_map[gy, gx] = 0  # 空地
        # 障碍物格
        if dist < LIDAR_MAX_DIST:
            x = pos[0] + dist * np.cos(a)
            y = pos[1] + dist * np.sin(a)
            gx, gy = world_to_grid(x, y, MAP_RESOLUTION)
            if 0 <= gx < MAP_SIZE and 0 <= gy < MAP_SIZE:
                known_map[gy, gx] = 1

# 简单A*寻路

def astar(start, goal, occ_map):
    sx, sy = world_to_grid(*start, MAP_RESOLUTION)
    gx, gy = world_to_grid(*goal, MAP_RESOLUTION)
    open_set = []
    heapq.heappush(open_set, (0, (sx, sy)))
    came_from = {}
    g_score = { (sx, sy): 0 }
    dirs = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == (gx, gy):
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return [grid_to_world(x, y, MAP_RESOLUTION) for x, y in path]
        for dx, dy in dirs:
            nx, ny = current[0]+dx, current[1]+dy
            if 0<=nx<MAP_SIZE and 0<=ny<MAP_SIZE and occ_map[ny, nx]!=1:
                tentative_g = g_score[current] + np.hypot(dx, dy)
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + np.hypot(gx-nx, gy-ny)
                    heapq.heappush(open_set, (f, (nx, ny)))
                    came_from[(nx, ny)] = current
    return None

# 可视化

def plot_map(true_map, known_map, robot_pos, trajectory, target=None):
    plt.clf()
    plt.imshow(true_map, cmap='gray_r', alpha=0.3, origin='lower')
    show_map = known_map.copy()
    show_map[show_map==-1] = 0.5
    plt.imshow(show_map, cmap='Blues', alpha=0.5, origin='lower')
    plt.plot([p[0]/MAP_RESOLUTION for p in trajectory], [p[1]/MAP_RESOLUTION for p in trajectory], 'g.-', label='Trajectory')
    plt.plot(robot_pos[0]/MAP_RESOLUTION, robot_pos[1]/MAP_RESOLUTION, 'ro', label='Robot')
    if target is not None:
        plt.plot(target[0]/MAP_RESOLUTION, target[1]/MAP_RESOLUTION, 'yx', markersize=12, label='Target')
        plt.plot(target[0]/MAP_RESOLUTION, target[1]/MAP_RESOLUTION, marker='*', color='y', markersize=18, label='Best Frontier')
    plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.xlim(0, MAP_SIZE)
    plt.ylim(0, MAP_SIZE)
    circle = patches.Circle((robot_pos[0]/MAP_RESOLUTION, robot_pos[1]/MAP_RESOLUTION), radius=ROBOT_RADIUS/MAP_RESOLUTION, fill=False, color='r', linestyle='--')
    plt.gca().add_patch(circle)
    plt.tight_layout(rect=(0, 0, 0.93, 1))
    plt.pause(0.01)

# 主循环
plt.ion()
step = 0
lidar_log = []
while True:
    scan = simulate_lidar(robot_pos, robot_theta, true_map)
    update_known_map(robot_pos, scan, known_map)
    lidar_log.append({
        'step': step,
        'timestamp': round(step * 0.1, 3),
        'robot_pos': [float(robot_pos[0]), float(robot_pos[1]), float(robot_theta)],
        'scan': scan.tolist()
    })
    target = explorer.get_next_target(known_map, robot_pos, exit_pos=(EXIT_POSITION['x'], EXIT_POSITION['y']))
    if target is None:
        print('所有前沿已探索，地图构建完成！')
        break
    path = astar(robot_pos, target, known_map)
    if path is None or len(path)<2:
        print('无法到达目标，跳过')
        continue
    # 只走一步
    next_pos = np.array(path[1])
    # 碰撞检测
    gx, gy = world_to_grid(next_pos[0], next_pos[1], MAP_RESOLUTION)
    if true_map[gy, gx]==1:
        print('碰到障碍，跳过')
        continue
    robot_pos = next_pos
    trajectory.append(robot_pos.copy())
    step += 1
    if step%2==0:
        plot_map(true_map, known_map, robot_pos, trajectory, target)

# 保存结果
np.save('exploration_trajectory.npy', np.array(trajectory))
np.save('exploration_final_map.npy', known_map)
np.savetxt('exploration_trajectory.txt', np.array(trajectory), fmt='%.4f', delimiter=',')
np.savetxt('exploration_final_map.txt', known_map, fmt='%.2f', delimiter=',')
with open('exploration_lidar.json', 'w') as f:
    json.dump(lidar_log, f, indent=2, ensure_ascii=False)
print('轨迹、已知地图和激光数据已保存。')
plt.ioff()
plot_map(true_map, known_map, robot_pos, trajectory, target=None)
plt.show()

# 保存最终地图图片
try:
    map_data = np.loadtxt('exploration_final_map.txt', delimiter=',')
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(['white', 'black', 'gray'])  # 0:空地, 1:障碍, -1:未知
    display_map = map_data.copy()
    display_map[display_map == -1] = 2
    plt.figure(figsize=(6,6))
    plt.imshow(display_map, cmap=cmap, origin='lower')
    plt.title('Exploration Final Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(ticks=[0,1,2], label='0:Free 1:Obstacle 2:Unknown')
    plt.tight_layout()
    plt.savefig('exploration_final_map.png', dpi=150)
    plt.close()
    print('最终地图图片 exploration_final_map.png 已保存。')
except Exception as e:
    print('保存地图图片失败:', e) 