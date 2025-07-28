import time
import pybullet as p
import pybullet_data
from natnet_client import DataDescriptions, DataFrame, NatNetClient

num_frames = 0

def receive_new_frame(data_frame: DataFrame):
    global num_frames
    num_frames += 1
    print(f"\n✅ Frame #{num_frames}")

    #for rb in data_frame.rigid_bodies:
    #    if rb.tracking_valid:
    #        print(f"Rigid Body {rb.id}: {rb.name if hasattr(rb, 'name') else ''}")
    #        print(f"  Position:     x={rb.position[0]:.3f}, y={rb.position[1]:.3f}, z={rb.position[2]:.3f}")
    #        print(f"  Orientation:  x={rb.orientation[0]:.3f}, y={rb.orientation[1]:.3f}, "
    #              f"z={rb.orientation[2]:.3f}, w={rb.orientation[3]:.3f}")
    #    else:
    #        print(f"❌ Rigid Body {rb.id} tracking lost in this frame.")




def receive_new_desc(desc: DataDescriptions):
    print("Received data descriptions.")
    print(desc)

def main():

    client = NatNetClient(
        server_ip_address="10.134.71.231",
        local_ip_address="10.134.71.40",
        use_multicast=False
    )

    client.on_data_description_received_event.handlers.append(receive_new_desc)
    client.on_data_frame_received_event.handlers.append(receive_new_frame)


    with client:

        client.request_modeldef()

        sim_dt = 1.0 / 1.0
        last_frame = 0
        for i in range(10):
            time.sleep(1)
            client.update_sync()
            print(f"Received {num_frames} frames in {i + 1}s")



if __name__ == "__main__":
    main()
