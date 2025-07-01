import subprocess

def main():
    print("==== [1] 路径规划与可视化 ====")
    subprocess.run(["python", "planner/visualize_path.py"], check=True)
    print("==== [2] DWA仿真与可视化 ====")
    subprocess.run(["python", "planner/visualize_dwa_path.py"], check=True)
    print("==== 全流程完成 ====")

if __name__ == "__main__":
    main() 