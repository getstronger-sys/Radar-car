import serial
import serial.tools.list_ports
import threading

def list_ports():
    """列出所有可用串口"""
    ports = list(serial.tools.list_ports.comports())
    for i, port in enumerate(ports):
        print(f"{i}: {port.device} - {port.description}")
    return ports

def read_from_port(ser):
    """不断读取串口数据"""
    while True:
        try:
            data = ser.readline()
            if data:
                print(f"[HC-04]: {data.decode('utf-8', errors='ignore').strip()}")
        except Exception as e:
            print(f"读取错误: {e}")
            break

def main():
    ports = list_ports()
    if not ports:
        print("没有找到串口设备！")
        return

    index = int(input("选择要连接的串口序号: "))
    port_name = ports[index].device
    baud_rate = int(input("输入波特率 (默认9600): ") or 9600)

    try:
        ser = serial.Serial(port_name, baud_rate, timeout=1)
        print(f"已连接 {port_name}，波特率 {baud_rate}")

        # 开启线程读取数据
        t = threading.Thread(target=read_from_port, args=(ser,))
        t.daemon = True
        t.start()

        # 主循环发送数据
        while True:
            msg = input()
            if msg.lower() == 'exit':
                break
            ser.write((msg + '\n').encode('utf-8'))

    except Exception as e:
        print(f"连接失败: {e}")
    finally:
        ser.close()
        print("串口已关闭")

if __name__ == "__main__":
    main()