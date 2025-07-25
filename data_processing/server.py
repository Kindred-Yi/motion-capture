import time
import pybullet as p
import pybullet_data
from natnet_client import DataDescriptions, DataFrame, NatNetClient
import threading

# --- 用于在线程间安全共享数据的全局变量 ---
# 使用线程锁来防止数据竞争
data_lock = threading.Lock()
# 初始化一个空的marker列表
latest_marker_data = []
# 帧计数器
num_frames = 0

def receive_new_frame(data_frame: DataFrame):
    """当新数据帧到达时，由NatNet后台线程自动调用此函数"""
    global latest_marker_data, num_frames
    
    # 增加帧计数
    num_frames += 1
    
    # 使用锁来安全地更新全局变量
    with data_lock:
        if data_frame.markers:
            # 清空旧数据并存储新数据
            latest_marker_data = data_frame.markers
            # 为了调试，可以保留打印语句，但在正式运行时可以注释掉以提高性能
            # print(f"✅ Frame #{num_frames}: Received {len(latest_marker_data)} unlabeled markers")
        else:
            # 如果没有接收到markers，也清空列表
            latest_marker_data = []

def receive_new_desc(desc: DataDescriptions):
    print("✅ Received data descriptions from Motive.")

def main():
    global latest_marker_data

    client = NatNetClient(
        server_ip_address="10.134.71.231",
        local_ip_address="10.134.71.40",
        use_multicast=False
    )

    client.on_data_description_received_event.handlers.append(receive_new_desc)
    client.on_data_frame_received_event.handlers.append(receive_new_frame)
    
    # 启动PyBullet
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    plane = p.loadURDF("plane.urdf")
    
    # 启动NatNet客户端的后台线程
    print("🚀 Starting NatNet client in background...")
    client.run_async()
    
    try:
        # 主模拟循环
        while True:
            local_markers = []
            # 使用锁来安全地读取共享数据
            with data_lock:
                if latest_marker_data:
                    # 创建数据的本地副本以进行处理
                    local_markers = list(latest_marker_data)

            if local_markers:
                print(f"Main loop processing {len(local_markers)} markers...")
                # 在这里，你可以使用 local_markers 数据来做IK解算等
                for i, marker in enumerate(local_markers):
                     print(f"  Marker {i}: x={marker[0]:.3f}, y={marker[1]:.3f}, z={marker[2]:.3f}")
                # joint_targets = solve_ik_from_markers(local_markers)
                # p.setJointMotorControlArray(...)
            else:
                print("Main loop: Waiting for marker data...")

            p.stepSimulation()
            time.sleep(1.0 / 1.0) # 按240Hz的频率运行模拟

    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped by user.")
    finally:
        # 确保在退出时关闭客户端
        print("🔌 Shutting down NatNet client...")
        client.shutdown()
        p.disconnect()
        print("✅ Done.")

if __name__ == "__main__":
    main()