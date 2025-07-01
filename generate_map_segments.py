import numpy as np
import json

# 读取轨迹点
trajectory = np.loadtxt('exploration_trajectory.txt', delimiter=',')

# 生成线段列表
segments = []
for i in range(len(trajectory) - 1):
    seg = {
        "start": [float(trajectory[i][0]), float(trajectory[i][1])],
        "end": [float(trajectory[i+1][0]), float(trajectory[i+1][1])]
    }
    segments.append(seg)

# 保存为json，格式与SEGMENTS一致
with open('trajectory_segments.json', 'w', encoding='utf-8') as f:
    json.dump(segments, f, indent=2, ensure_ascii=False)

print("已生成 trajectory_segments.json，格式与SEGMENTS一致")