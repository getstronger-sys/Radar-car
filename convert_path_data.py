import math
import os

# 文件路径
input_file = r"D:\PycharmProjects\PythonProject1\planner\output\hardware_path_data.txt"
output_file = r"D:\PycharmProjects\PythonProject1\planner\output\hardware_motion_cmds.txt"

# 读取路径数据
points = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        x = float(parts[0][1:])  # 去掉P
        y = float(parts[1])
        angle = float(parts[2])
        points.append((x, y, angle))

# 生成小车指令
instructions = []
for i in range(len(points) - 1):
    x1, y1, _ = points[i]
    x2, y2, angle_next = points[i + 1]
    # 计算直行距离
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)*100
    instructions.append(f"F {dist:.2f}")
    # 添加转向指令
    if angle_next > 0:
        instructions.append(f"L {abs(angle_next):.1f}")
    elif angle_next < 0:
        instructions.append(f"R {abs(angle_next):.1f}")

# 保存到文件
with open(output_file, 'w', encoding='utf-8') as f:
    for ins in instructions:
        f.write(ins + "\n")

print(f"已生成小车行进指令，保存至: {output_file}")