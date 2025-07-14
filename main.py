import subprocess

def main():
    print("==== [0] 地图探索仿真 ====")
    subprocess.run(["python", "simulate_exploration_new.py"], check=True)
    print("==== [1] 路径规划与可视化 ====")
    subprocess.run(["python", "main_pipeline.py"], check=True)
    print("==== 全流程完成 ====")

if __name__ == "__main__":
    main()
