import serial
import serial.tools.list_ports
import threading
import time

class BluetoothController:
    def __init__(self):
        self.ser = None
        self.connected = False
        self.current_instruction_index = 0
        self.instructions = []
        self.waiting_for_q = False
        self.stop_event = threading.Event()

    def list_ports(self):
        """列出所有可用串口"""
        ports = list(serial.tools.list_ports.comports())
        for i, port in enumerate(ports):
            print(f"{i}: {port.device} - {port.description}")
        return ports

    def connect(self, port_name, baud_rate=9600):
        """连接到指定串口"""
        try:
            self.ser = serial.Serial(port_name, baud_rate, timeout=1)
            self.connected = True
            print(f"已连接 {port_name}，波特率 {baud_rate}")
            return True
        except Exception as e:
            print(f"连接失败: {e}")
            return False

    def read_from_port(self):
        """不断读取串口数据，当收到'q'时发送下一条指令"""
        while not self.stop_event.is_set():
            try:
                if self.ser and self.ser.in_waiting > 0:
                    data = self.ser.read().decode('utf-8', errors='ignore')
                    if data:
                        print(f"[小车]: {data}")
                        # 检查是否收到'q'
                        if data.strip() == 'q' and self.waiting_for_q:
                            self.waiting_for_q = False
                            self.send_next_instruction()
            except Exception as e:
                print(f"读取错误: {e}")
                time.sleep(1)

    def load_instructions(self, file_path):
        """从文件中加载指令"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.instructions = [line.strip() for line in f if line.strip()]
            print(f"已加载 {len(self.instructions)} 条指令")
            return True
        except Exception as e:
            print(f"加载指令失败: {e}")
            return False

    def send_next_instruction(self):
        """发送下一条指令"""
        if self.current_instruction_index < len(self.instructions):
            instruction = self.instructions[self.current_instruction_index]
            self.send_instruction(instruction)
            self.current_instruction_index += 1
        else:
            print("所有指令已发送完成")

    def send_instruction(self, instruction):
        """发送单条指令"""
        if self.ser and self.connected:
            try:
                print(f"[发送]: {instruction}")
                self.ser.write((instruction + '\n').encode('utf-8'))
                self.waiting_for_q = True
            except Exception as e:
                print(f"发送失败: {e}")

    def start(self):
        """启动控制流程"""
        # 列出可用串口
        ports = self.list_ports()
        if not ports:
            print("没有找到串口设备！")
            return

        # 用户选择串口
        index = int(input("选择要连接的串口序号: "))
        port_name = ports[index].device
        baud_rate = int(input("输入波特率 (默认9600): ") or 9600)

        # 连接串口
        if not self.connect(port_name, baud_rate):
            return

        # 加载指令文件
        if not self.load_instructions('hardware_instructions.txt'):
            self.close()
            return

        # 开启线程读取数据
        t = threading.Thread(target=self.read_from_port)
        t.daemon = True
        t.start()

        try:
            # 发送第一条指令
            self.send_next_instruction()
            
            # 保持主程序运行
            while self.connected and (self.current_instruction_index < len(self.instructions) or self.waiting_for_q):
                time.sleep(0.1)
                # 检查是否需要手动退出
                if not self.stop_event.is_set():
                    # 这里可以添加其他需要的逻辑
                    pass

        except KeyboardInterrupt:
            print("程序被用户中断")
        finally:
            self.close()

    def close(self):
        """关闭串口连接"""
        self.stop_event.set()
        if self.ser:
            try:
                self.ser.close()
                print("串口已关闭")
            except:
                pass
        self.connected = False

if __name__ == "__main__":
    controller = BluetoothController()
    controller.start()