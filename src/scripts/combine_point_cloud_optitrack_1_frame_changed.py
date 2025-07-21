import open3d as o3d
import numpy as np
import os
import sys
import argparse
import cv2

from parser import RigidBody, extract_rigid_body_data
from mocap_classes import Marker, extract_marker_data

ARUCO_1x1_size = 0.0353  # meters, size of the ArUco marker in meters
OPTI_SCALE = 1  # OptiTrack scale factor when messed up calibration, usually should be 1
OPTI_FRAME_NUM = 140

def load_camera_extrinsics(extrinsic_file="azure_kinect_extrinsics.yml"):
    """
    Loads a 4x4 transformation matrix from a YAML file containing rvec and tvec.
    """
    if not os.path.exists(extrinsic_file):
        raise FileNotFoundError(f"Extrinsics file not found: {extrinsic_file}")

    fs = cv2.FileStorage(extrinsic_file, cv2.FILE_STORAGE_READ)
    rvec = fs.getNode("rvec").mat()
    tvec = fs.getNode("tvec").mat()
    fs.release()

    if rvec is None or tvec is None:
        raise ValueError(f"Failed to read 'rvec' or 'tvec' from extrinsics file: {extrinsic_file}")

    R, _ = cv2.Rodrigues(rvec)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = tvec.ravel()

    return transformation_matrix

def quaternion_to_rotation_matrix(q):
    """
    Converts a quaternion into a 3x3 rotation matrix.
    """
    w, x, y, z = q['w'], q['x'], q['y'], q['z']
    x2, y2, z2 = x*x, y*y, z*z
    
    R = np.array([
        [1 - 2*y2 - 2*z2,   2*x*y - 2*z*w,   2*x*z + 2*y*w],
        [2*x*y + 2*z*w,   1 - 2*x2 - 2*z2,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,   2*y*z + 2*x*w,   1 - 2*x2 - 2*y2]
    ])
    return R

def get_all_rigid_body_poses(filepath, frame_number=OPTI_FRAME_NUM):
    """
    Loads motion capture data from a CSV file and returns the world poses of all 
    rigid bodies at a specific frame.
    """
    all_poses = {}
    mocap_data = extract_rigid_body_data(filepath)

    if not mocap_data:
        print(f"⚠️ Warning: Could not extract any rigid body data from '{filepath}'.")
        return all_poses

    print(f"Found rigid bodies: {list(mocap_data.keys())}")

    for name, body in mocap_data.items():
        position = body.get_position_at_frame(frame_number)
        rotation_q = body.get_rotation_at_frame(frame_number)

        if position is None or rotation_q is None:
            print(f"⚠️ Warning: No data found for rigid body '{name}' at frame {frame_number}. Skipping.")
            continue
            
        T_world_rigid = np.eye(4)
        T_world_rigid[:3, :3] = quaternion_to_rotation_matrix(rotation_q)
        T_world_rigid[:3, 3] = [position['x'] / OPTI_SCALE, position['y'] / OPTI_SCALE, position['z'] / OPTI_SCALE]
        
        all_poses[name] = T_world_rigid

    print(f"✅ Loaded poses for {len(all_poses)} rigid bodies at frame {frame_number}.")
    return all_poses

def get_all_marker_positions(filepath, frame_number=1):
    """
    Loads motion capture data from a CSV file and returns the world positions of all
    markers at a specific frame.
    """
    all_positions = {}
    marker_data = extract_marker_data(filepath)

    if not marker_data:
        print(f"⚠️ Warning: Could not extract any marker data from '{filepath}'.")
        return all_positions

    print(f"Found markers: {list(marker_data.keys())}")

    for name, marker in marker_data.items():
        position = marker.get_position_at_frame(frame_number)

        if position is None or any(v is None for v in position.values()):
            print(f"⚠️ Warning: No data found for marker '{name}' at frame {frame_number}. Skipping.")
            continue
            
        all_positions[name] = np.array([position['x'] / OPTI_SCALE, position['y'] / OPTI_SCALE, position['z'] / OPTI_SCALE])
        
    print(f"✅ Loaded positions for {len(all_positions)} markers at frame {frame_number}.")
    return all_positions

def create_pcd_from_mkv(mkv_path, frame_number):
    """
    Creates a registered point cloud by reading a specific frame from an Azure Kinect MKV file.
    This replaces the separate image/intrinsic loading and undistortion steps.
    """
    print(f"Reading data from {mkv_path}...")
    reader = o3d.read_azure_kinect_mkv_reader(mkv_path)
    if not reader.is_opened():
        raise RuntimeError(f"Failed to open MKV file: {mkv_path}")

    # Get the camera intrinsics directly from the file's metadata
    camera_intrinsic = reader.get_intrinsic()

    # Seek to the desired frame by reading frames one by one
    print(f"Seeking to frame {frame_number}...")
    rgbd_image = None
    for i in range(frame_number):
        rgbd_image = reader.next_frame()
        if rgbd_image is None:
            reader.close()
            raise RuntimeError(f"Failed to read frame {i}. The recording may be shorter than {frame_number} frames.")
    
    # The last frame read is the one we want
    print("✅ Captured and registered RGBD frame from MKV.")
    
    # Create the point cloud from the registered RGBD image.
    # The SDK handles the alignment of depth to color.
    pcd_camera_frame = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        camera_intrinsic
    )
    
    reader.close()
    
    print(f"✅ Created point cloud with {len(pcd_camera_frame.points)} points.")
    return pcd_camera_frame, camera_intrinsic

def main():
    parser = argparse.ArgumentParser(description='Convert RGB/D images to a point cloud and transform it into the OptiTrack world frame.')
    # MODIFIED: Replaced --rgb and --depth with --mkv
    parser.add_argument('--mkv', required=True, help='Path to the Azure Kinect .mkv recording file')
    parser.add_argument('-o','--optitrack', dest='optitrack', required=True, help='Path to the CSV file that has the OptiTrack data')
    parser.add_argument('-crc', '--camRigidcalib', dest='camRigidcalib', default = 'Kinect_cam.yml', help='Transformation file from Camera to its Rigid Body (T_rigid_cam)')
    parser.add_argument('-c','--cam_body_name', dest='camBodyName', required=True, help='The name of the rigid body attached to the camera in the OptiTrack data')
    parser.add_argument('--output', default='output_world.pcd', help='Output world-frame point cloud file path')

    args = parser.parse_args()

    try:
        # 1. & 2. Create point cloud directly from the MKV file
        # This new function handles loading the data and creating a registered point cloud.
        pcd_camera_frame, camera_intrinsic = create_pcd_from_mkv(args.mkv, frame_number=OPTI_FRAME_NUM)
        
        # 3. Load all rigid body poses from OptiTrack (T_world_rigid)
        all_poses_world = get_all_rigid_body_poses(args.optitrack, frame_number=OPTI_FRAME_NUM)
        if not all_poses_world:
            print("Could not load any rigid body poses from the OptiTrack file. Continuing without them.")

        # Load all marker positions from OptiTrack
        all_marker_positions_world = get_all_marker_positions(args.optitrack, frame_number=OPTI_FRAME_NUM)

        # 4. Get the specific pose of the camera's rigid body in the world
        if args.camBodyName not in all_poses_world:
            raise ValueError(f"Camera rigid body '{args.camBodyName}' not found in OptiTrack data. Found: {list(all_poses_world.keys())}")
        T_world_camRigid = all_poses_world[args.camBodyName]

        # 5. Load the static transformation from the camera's optical frame to its rigid body frame (T_rigid_cam)
        T_rigid_cam = load_camera_extrinsics(args.camRigidcalib)

        # 6. Calculate the full transformation from the camera frame to the world frame
        T_world_cam = T_world_camRigid @ T_rigid_cam

        # 7. Transform the point cloud into the world frame
        pcd_world_frame = pcd_camera_frame.transform(T_world_cam)
        
        # 8. Save the transformed point cloud
        o3d.io.write_point_cloud(args.output, pcd_world_frame)
        print(f"✅ Point cloud transformed to world frame and saved to {args.output}")
        print(f"Point cloud contains {len(pcd_world_frame.points)} points")

        # 9. Prepare for visualization
        geometries_to_draw = [pcd_world_frame]

        print("Visualizing world origin (large frame at [0,0,0])")
        world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4, origin=[0, 0, 0])
        geometries_to_draw.append(world_frame)

        for name, T_world_rigid in all_poses_world.items():
            print(f"Visualizing rigid body: {name}")
            rigid_body_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            rigid_body_frame.transform(T_world_rigid)
            geometries_to_draw.append(rigid_body_frame)
        
        camera_pose_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
        camera_pose_frame.transform(T_world_cam)
        geometries_to_draw.append(camera_pose_frame)

        # Add a sphere for each marker to visualize its position
        for name, position in all_marker_positions_world.items():
            print(f"Visualizing marker: {name}")
            marker_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01) # 1 cm radius
            marker_sphere.paint_uniform_color([1.0, 0.0, 0.0]) # Red color
            marker_sphere.translate(position)
            geometries_to_draw.append(marker_sphere)

        # 10. Visualize everything together
        o3d.visualization.draw_geometries(
            geometries_to_draw,
            window_name="Point Cloud, Markers, and All Rigid Bodies in World Frame"
        )

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()