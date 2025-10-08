import serial.tools.list_ports
import psutil
import winreg
import os

class SerialPortChecker:
    def __init__(self):
        pass
    
    def list_all_ports(self):
        """列出所有可用的串口及其信息"""
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("没有找到可用的串口设备。")
            return []
        
        print("所有可用的串口设备:")
        print("="*60)
        print("序号 | 端口名 | 描述 | 硬件ID")
        print("="*60)
        
        for i, port in enumerate(ports):
            print(f"{i:4d} | {port.device} | {port.description} | {port.hwid}")
        print("="*60)
        
        return ports
    
    def check_port_status(self, port_name):
        """检查指定端口的状态"""
        try:
            # 尝试打开端口
            ser = serial.Serial(port_name, 9600, timeout=1)
            ser.close()
            print(f"端口 {port_name} 可以正常打开。")
            return True
        except Exception as e:
            print(f"无法打开端口 {port_name}: {e}")
            return False
    
    def find_process_using_port(self, port_name):
        """查找占用指定串口的进程"""
        try:
            # 使用wmic命令查找占用串口的进程
            import subprocess
            cmd = f'wmic path Win32_PnPEntity where "Name like \'%{port_name}%\'" get DeviceID,PNPDeviceID'
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout.strip():
                print(f"端口 {port_name} 相关信息:")
                print(result.stdout)
            else:
                print(f"未找到端口 {port_name} 的进程信息。")
        except Exception as e:
            print(f"查找进程信息时出错: {e}")
    
    def get_com_port_registry_info(self):
        """从注册表获取COM端口信息"""
        try:
            key_path = r'SYSTEM\CurrentControlSet\Enum\USB'  
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as key:
                print("USB设备注册表信息:")
                i = 0
                while True:
                    try:
                        subkey_name = winreg.EnumKey(key, i)
                        # 只打印包含COM端口的设备信息
                        if 'COM' in subkey_name.upper():
                            print(f"- {subkey_name}")
                        i += 1
                    except OSError:
                        break
        except Exception as e:
            print(f"读取注册表时出错: {e}")
    
    def check_system_resources(self):
        """检查系统资源使用情况"""
        memory = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=1)
        
        print("系统资源使用情况:")
        print(f"- CPU使用率: {cpu}%")
        print(f"- 内存使用: {memory.percent}%")
        print(f"- 可用内存: {memory.available // (1024*1024)} MB")
        
        if memory.percent > 90:
            print("警告: 系统内存使用率过高，可能导致资源不足错误。")
        if cpu > 90:
            print("警告: CPU使用率过高，可能导致系统响应缓慢。")
    
    def provide_troubleshooting_advice(self):
        """提供串口连接问题的故障排除建议"""
        print("\n故障排除建议:")
        print("1. 确保蓝牙设备已正确连接并驱动正常安装")
        print("2. 检查是否有其他程序正在使用COM3端口")
        print("3. 尝试关闭其他占用系统资源的程序")
        print("4. 重启计算机后再试")
        print("5. 尝试将蓝牙设备连接到其他USB端口")
        print("6. 更新蓝牙设备驱动程序")
        print("7. 检查设备管理器中是否有冲突的设备")
    
    def run_full_check(self):
        """运行完整的串口检查"""
        print("=== 串口连接问题排查工具 ===")
        
        # 列出所有可用串口
        self.list_all_ports()
        
        # 检查COM3端口状态
        print("\n=== 检查COM3端口状态 ===")
        self.check_port_status('COM3')
        
        # 查找占用COM3的进程
        print("\n=== 查找占用COM3的进程 ===")
        self.find_process_using_port('COM3')
        
        # 检查系统资源
        print("\n=== 检查系统资源 ===")
        self.check_system_resources()
        
        # 提供故障排除建议
        self.provide_troubleshooting_advice()

if __name__ == "__main__":
    checker = SerialPortChecker()
    checker.run_full_check()