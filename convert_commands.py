import json

# 读取JSON文件
with open('hardware_commands_75_segments.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

commands = data['commands']
instructions = []

# 转换每个命令为指定格式
for cmd in commands:
    rotation = cmd['rotation_degrees']
    distance = cmd['distance']
    
    # 处理旋转命令
    if rotation > 0:
        instructions.append(f"R {rotation}")
    elif rotation < 0:
        instructions.append(f"L {-rotation}")
    
    # 处理移动命令（将米转换为厘米，乘以100）
    if distance > 0:
        instructions.append(f"F {distance * 100}")

# 将指令写入新文件
with open('hardware_instructions.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(instructions))

print(f"已成功生成指令文件: hardware_instructions.txt")