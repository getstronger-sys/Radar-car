import serial
import serial.tools.list_ports
import threading
import time
import logging

# 配置日志 - 设置为DEBUG级别以获取更多诊断信息
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
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
        read_count = 0
        while not self.stop_event.is_set():
            try:
                if self.ser and self.ser.is_open:
                    # 无论是否有数据等待，都尝试读取，防止某些情况下数据被遗漏
                    data = self.ser.read(1024).decode('utf-8', errors='ignore')
                    read_count += 1
                    
                    # 每50次读取记录一次状态，帮助诊断
                    if read_count % 50 == 0:
                        logger.debug(f"读取状态: 缓冲区长度={len(buffer)}, 上次读取数据长度={len(data)}")
                    
                    if data:
                        print(f"[原始数据]: {repr(data)}")  # 打印原始数据，包括不可见字符
                        buffer += data
                        
                        # 处理缓冲区中的数据，不仅限于完整行
                        if len(buffer) > 0:
                            # 首先尝试按行处理
                            if '\n' in buffer or '\r' in buffer:
                                lines = buffer.splitlines()
                                # 保存未完成的行
                                buffer = lines[-1] if lines else ""
                                for line in lines[:-1] if lines else []:
                                    line = line.strip()
                                    # 同时记录日志和打印到控制台，实现实时显示
                                    print(f"[小车]: {line}")
                                    logger.info(f"[小车]: {line}")
                                    # 检查是否收到'q'或包含'q'的响应
                                    if 'q' in line.lower() and self.waiting_for_q:
                                        logger.info("收到完成信号，准备发送下一条指令")
                                        self.waiting_for_q = False
                                        self.send_next_instruction()
                            
                            # 即使没有换行符，也检查缓冲区中是否有'q'字符
                            # 这确保即使数据不完整也能检测到'q'
                            if 'q' in buffer.lower() and self.waiting_for_q:
                                logger.info("在非完整行中收到完成信号，准备发送下一条指令")
                                self.waiting_for_q = False
                                self.send_next_instruction()
                                # 清空缓冲区，避免重复检测
                                buffer = ""
                    
                    # 如果缓冲区过大，清空部分数据以防止内存问题
                    if len(buffer) > 4096:
                        logger.warning(f"缓冲区过大({len(buffer)}字节)，清空部分数据")
                        buffer = buffer[-2048:]  # 只保留最后2048字节
                    
                    time.sleep(0.05)  # 减少延迟，提高响应速度
                else:
                    logger.warning("串口未打开，无法读取数据")
                    time.sleep(1)
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
            logger.info("调试模式: 您可以输入 't' 进行手动测试发送指令，输入其他键将被忽略")
            
            # 主循环 - 添加手动测试功能
            import sys
            import select
            
            while self.connected and (self.current_instruction_index < len(self.instructions) or self.waiting_for_q):
                # 检查是否有键盘输入（非阻塞方式）
                if sys.platform == 'win32':
                    # Windows平台的非阻塞输入检测
                    import msvcrt
                    if msvcrt.kbhit():
                        key = msvcrt.getch().decode('utf-8')
                        if key == 't':
                            logger.info("进入手动测试模式")
                            test_instruction = input("请输入要发送的测试指令: ").strip()
                            if test_instruction:
                                logger.info(f"发送手动测试指令: {test_instruction}")
                                # 保存当前状态
                                original_waiting = self.waiting_for_q
                                original_index = self.current_instruction_index
                                
                                # 发送测试指令
                                self.waiting_for_q = True
                                success = self.send_instruction(test_instruction)
                                
                                if success:
                                    logger.info("等待小车响应...")
                                    # 等待响应或超时
                                    start_time = time.time()
                                    while self.waiting_for_q and time.time() - start_time < 10:
                                        time.sleep(0.1)
                                    
                                    if self.waiting_for_q:
                                        logger.warning("手动测试超时，未收到响应")
                                        self.waiting_for_q = original_waiting
                                    else:
                                        logger.info("手动测试完成，已收到响应")
                                
                                # 恢复原始状态
                                self.current_instruction_index = original_index
                else:
                    # Unix/Mac平台的非阻塞输入检测
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.readline().strip()
                        if key == 't':
                            logger.info("进入手动测试模式")
                            test_instruction = input("请输入要发送的测试指令: ").strip()
                            if test_instruction:
                                logger.info(f"发送手动测试指令: {test_instruction}")
                                # 保存当前状态
                                original_waiting = self.waiting_for_q
                                original_index = self.current_instruction_index
                                
                                # 发送测试指令
                                self.waiting_for_q = True
                                success = self.send_instruction(test_instruction)
                                
                                if success:
                                    logger.info("等待小车响应...")
                                    # 等待响应或超时
                                    start_time = time.time()
                                    while self.waiting_for_q and time.time() - start_time < 10:
                                        time.sleep(0.1)
                                    
                                    if self.waiting_for_q:
                                        logger.warning("手动测试超时，未收到响应")
                                        self.waiting_for_q = original_waiting
                                    else:
                                        logger.info("手动测试完成，已收到响应")
                                
                                # 恢复原始状态
                                self.current_instruction_index = original_index
                
                # 检查是否所有指令都已发送且等待完成
                if self.current_instruction_index >= len(self.instructions) and not self.waiting_for_q:
                    logger.info("所有指令已发送完成，等待最终确认...")
                    time.sleep(2)  # 等待最后响应
                    break
                
                time.sleep(0.1)

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