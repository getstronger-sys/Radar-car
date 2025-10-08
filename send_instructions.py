import serial
import serial.tools.list_ports
import threading
import time
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('BluetoothController')

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

    def connect(self, port_name, baud_rate=9600, max_retries=3):
        """连接到指定串口，支持重试"""
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # 确保之前的连接已关闭
                if self.ser and self.ser.is_open:
                    self.ser.close()
                    time.sleep(0.5)  # 给系统时间释放端口
                
                # 尝试打开端口
                logger.info(f"尝试连接 {port_name}，波特率 {baud_rate}，重试次数 {retry_count+1}/{max_retries}")
                
                # 使用更多参数配置串口
                self.ser = serial.Serial(
                    port=port_name,
                    baudrate=baud_rate,
                    timeout=1,
                    write_timeout=2,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    bytesize=serial.EIGHTBITS,
                    xonxoff=False,
                    rtscts=False,
                    dsrdtr=False
                )
                
                # 确认端口已打开
                if self.ser.is_open:
                    self.connected = True
                    logger.info(f"已成功连接 {port_name}，波特率 {baud_rate}")
                    return True
                else:
                    logger.error(f"端口已创建但未打开: {port_name}")
                    retry_count += 1
                    time.sleep(1)  # 延迟后重试
                    continue
                    
            except serial.SerialException as e:
                logger.error(f"串口异常: {e}")
                retry_count += 1
                time.sleep(1)  # 延迟后重试
            except OSError as e:
                logger.error(f"操作系统错误: {e}")
                # 资源不足错误的特殊处理
                if "系统资源不足" in str(e):
                    logger.error("系统资源不足，请关闭其他程序后重试。")
                    # 尝试释放更多资源
                    import gc
                    gc.collect()
                retry_count += 1
                time.sleep(1.5)  # 资源不足时延迟更长时间
            except Exception as e:
                logger.error(f"连接失败: {type(e).__name__}: {e}")
                retry_count += 1
                time.sleep(1)
        
        logger.error(f"在 {max_retries} 次重试后仍无法连接到 {port_name}")
        self.connected = False
        return False

    def read_from_port(self):
        """不断读取串口数据，当收到'q'时发送下一条指令"""
        buffer = ""
        while not self.stop_event.is_set():
            try:
                if self.ser and self.ser.in_waiting > 0:
                    # 读取所有可用数据
                    data = self.ser.read(self.ser.in_waiting).decode('utf-8', errors='ignore')
                    if data:
                        buffer += data
                        # 处理缓冲区中的完整行
                        if '\n' in buffer or '\r' in buffer:
                            lines = buffer.splitlines()
                            # 保存未完成的行
                            buffer = lines[-1] if lines else ""
                            for line in lines[:-1] if lines else []:
                                line = line.strip()
                                logger.info(f"[小车]: {line}")
                                # 检查是否收到'q'或包含'q'的响应
                                if 'q' in line.lower() and self.waiting_for_q:
                                    logger.info("收到完成信号，准备发送下一条指令")
                                    self.waiting_for_q = False
                                    self.send_next_instruction()
                    time.sleep(0.1)  # 短暂延迟，减少CPU使用率
            except serial.SerialException as e:
                logger.error(f"串口读取异常: {e}")
                # 尝试重新连接
                if self.ser and not self.ser.is_open:
                    logger.info("尝试重新连接串口...")
                    port_name = self.ser.port
                    baud_rate = self.ser.baudrate
                    self.connect(port_name, baud_rate, max_retries=1)
                time.sleep(1)
            except Exception as e:
                logger.error(f"读取错误: {type(e).__name__}: {e}")
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
        """发送单条指令，支持重试"""
        if self.ser and self.connected:
            max_retries = 2
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    logger.info(f"[发送指令 {self.current_instruction_index+1}/{len(self.instructions)}]: {instruction}")
                    
                    # 确保串口已打开
                    if not self.ser.is_open:
                        logger.warning("串口已关闭，尝试重新打开...")
                        self.ser.open()
                        time.sleep(0.5)
                    
                    # 发送指令，添加\r\n以确保正确的行结束符
                    self.ser.write((instruction + '\r\n').encode('utf-8'))
                    self.ser.flush()  # 确保数据发送出去
                    self.waiting_for_q = True
                    logger.info(f"指令 '{instruction}' 已成功发送")
                    return True
                    
                except serial.SerialTimeoutException:
                    logger.error(f"发送超时: {instruction}")
                    retry_count += 1
                    time.sleep(0.5)
                except serial.SerialException as e:
                    logger.error(f"串口发送异常: {e}")
                    # 尝试重新连接
                    if not self.ser.is_open:
                        logger.info("尝试重新连接串口...")
                        port_name = self.ser.port
                        baud_rate = self.ser.baudrate
                        self.connect(port_name, baud_rate, max_retries=1)
                    retry_count += 1
                    time.sleep(0.5)
                except Exception as e:
                    logger.error(f"发送失败: {type(e).__name__}: {e}")
                    retry_count += 1
                    time.sleep(0.5)
            
            logger.error(f"在 {max_retries} 次重试后仍无法发送指令: {instruction}")
            return False
        else:
            logger.error("无法发送指令: 串口未连接")
            return False

    def start(self):
        """启动控制流程"""
        logger.info("==== 蓝牙指令发送器启动 ====")
        
        # 列出可用串口
        ports = self.list_ports()
        if not ports:
            logger.error("没有找到串口设备！")
            print("\n请确认蓝牙设备已正确连接，然后按Enter键退出...")
            input()
            return

        try:
            # 用户选择串口
            print("\n请选择要连接的串口序号: ")
            index_input = input().strip()
            index = int(index_input) if index_input else 0  # 默认选择第一个端口
            
            if 0 <= index < len(ports):
                port_name = ports[index].device
            else:
                logger.error(f"无效的端口序号: {index}")
                port_name = ports[0].device
                logger.info(f"自动选择第一个端口: {port_name}")
            
            # 波特率设置
            print("输入波特率 (默认9600): ")
            baud_input = input().strip()
            baud_rate = int(baud_input) if baud_input else 9600  # 默认9600
            
            logger.info(f"准备连接端口: {port_name}，波特率: {baud_rate}")
        except ValueError as e:
            logger.error(f"输入错误: {e}")
            print("\n输入无效，请重新运行程序。按Enter键退出...")
            input()
            return
        except Exception as e:
            logger.error(f"用户输入处理错误: {e}")
            print("\n发生错误，请重新运行程序。按Enter键退出...")
            input()
            return

        # 连接串口
        if not self.connect(port_name, baud_rate):
            logger.error("无法连接到串口，程序即将退出")
            print("\n请检查设备连接和端口占用情况，然后按Enter键退出...")
            input()
            return

        # 加载指令文件
        if not self.load_instructions('hardware_instructions.txt'):
            self.close()
            print("\n按Enter键退出...")
            input()
            return

        # 开启线程读取数据
        t = threading.Thread(target=self.read_from_port)
        t.daemon = True
        t.start()

        try:
            # 发送第一条指令
            self.send_next_instruction()
            
            logger.info("开始监听小车响应，按Ctrl+C可中断程序...")
            
            # 主循环 - 可以添加手动控制选项
            while self.connected and (self.current_instruction_index < len(self.instructions) or self.waiting_for_q):
                time.sleep(0.1)
                
                # 检查是否所有指令都已发送且等待完成
                if self.current_instruction_index >= len(self.instructions) and not self.waiting_for_q:
                    logger.info("所有指令已发送完成，等待最终确认...")
                    time.sleep(2)  # 等待最后响应
                    break

        except KeyboardInterrupt:
            logger.info("程序被用户中断")
            print("\n程序已停止发送指令")
        finally:
            self.close()
            print("\n蓝牙通信已关闭，按Enter键退出...")
            input()

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