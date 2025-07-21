import open3d as o3d
import numpy as np
import os
import sys
import argparse
import cv2
import cv2.aruco as aruco   

from parser import RigidBody, extract_rigid_body_data
# MODIFIED: Import Marker and extract_marker_data to handle marker data
from mocap_classes import Marker, extract_marker_data

ARUCO_1x1_size = 0.0353  # meters, size of the ArUco marker in meters
OPTI_SCALE = 1  # OptiTrack scale factor when messed up calibration, usually should be 1
OPTI_FRAME_NUM = 140

def load_azure_kinect_intrinsics(color_raw, intrinsic_file="azure_kinect_intrinsics.yml"):
    """Load Azure Kinect intrinsic parameters"""
    try:
        fs = cv2.FileStorage(intrinsic_file, cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode("cameraMatrix").mat()
        dist_coeffs = fs.getNode("distCoeffs").mat()
        fs.release()

        color_np = np.asarray(color_raw)
        print(f"Image resolution: {color_np.shape[1]}x{color_np.shape[0]}")

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width=color_np.shape[1],
            height=color_np.shape[0],
            fx=camera_matrix[0, 0],
            fy=camera_matrix[1, 1],
            cx=camera_matrix[0, 2],
            cy=camera_matrix[1, 2]
        )
        return intrinsic, dist_coeffs
    except Exception as e:
        print(f"Warning: Unable to load Azure Kinect intrinsics from '{intrinsic_file}' or '{color_raw}'. Using default. Error: {e}")
        return None

def read_rgbd_images(rgb_path, depth_path):
    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        raise FileNotFoundError(f"RGB or depth image not found at {rgb_path} or {depth_path}")

    color_raw = o3d.io.read_image(rgb_path)
    depth_raw = o3d.io.read_image(depth_path)

    return color_raw, depth_raw

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

# NEW: Function to get all marker positions using mocap_classes.py
def get_all_marker_positions(filepath, frame_number=1):
    """
    Loads motion capture data from a CSV file and returns the world positions of all
    markers at a specific frame.

    Args:
        filepath (str): Path to the OptiTrack CSV file.
        frame_number (int): The frame number to extract data from.

    Returns:
        dict: A dictionary where keys are marker names (str) and values
              are their 3D position vectors (np.ndarray).
    """
    all_positions = {}
    # Use the extract_marker_data function from mocap_classes.py
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
            
        # Store the position as a NumPy array for easy use with Open3D
        all_positions[name] = np.array([position['x'] / OPTI_SCALE, position['y'] / OPTI_SCALE, position['z'] / OPTI_SCALE])
        
    print(f"✅ Loaded positions for {len(all_positions)} markers at frame {frame_number}.")
    return all_positions


# def detect_and_estimate_aruco_poses(image, camera_matrix, dist_coeffs, marker_size=0.05):
#     """
#     Detects ArUco markers in an image and estimates their poses.
#     Returns rvecs and tvecs for each detected marker.
#     """
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
#     parameters = aruco.DetectorParameters()
    
#     corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

#     if ids is not None and len(ids) > 0:
#         print(f"✅ Detected {len(ids)} ArUco marker(s): {ids.flatten().tolist()}")
#         rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
#         return rvecs, tvecs, ids
#     else:
#         print("⚠️ No ArUco markers detected in the image.")
#         return None, None, None

# def get_aruco_geometries_in_world_frame(color_raw, camera_matrix, dist_coeffs, T_world_cam, marker_size):
#     """
#     Detects ArUco tags and returns a list of Open3D geometries for visualization.

#     Args:
#         color_raw (o3d.geometry.Image): The raw color image.
#         camera_matrix (np.ndarray): The camera intrinsic matrix.
#         dist_coeffs (np.ndarray): The camera distortion coefficients.
#         T_world_cam (np.ndarray): The 4x4 transformation matrix from the camera to the world frame.
#         marker_size (float): The real-world size of the ArUco markers in meters.

#     Returns:
#         list: A list of o3d.geometry.TriangleMesh objects, one for each detected tag.
#     """
#     aruco_geometries = []
    
#     # Convert Open3D image to NumPy array for OpenCV (RGB to BGR)
#     color_cv_image = cv2.cvtColor(np.asarray(color_raw), cv2.COLOR_RGB2BGR)
#     rvecs, tvecs, ids = detect_and_estimate_aruco_poses(color_cv_image, camera_matrix, dist_coeffs, marker_size)

#     if ids is not None:
#         for i, aruco_id in enumerate(ids):
#             print(f"Visualizing ArUco tag ID: {aruco_id[0]}")
#             # Create the transformation matrix from the ArUco tag to the camera (T_cam_aruco)
#             T_cam_aruco = np.eye(4)
#             R_cam_aruco, _ = cv2.Rodrigues(rvecs[i])
#             T_cam_aruco[:3, :3] = R_cam_aruco
#             T_cam_aruco[:3, 3] = tvecs[i].flatten()

#             # Transform the ArUco pose into the world frame (T_world_aruco)
#             T_world_aruco = T_world_cam @ T_cam_aruco
            
#             # Create a coordinate frame to represent the ArUco tag's pose
#             aruco_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=marker_size, origin=[0, 0, 0])
#             aruco_frame.transform(T_world_aruco)
#             aruco_geometries.append(aruco_frame)
            
#     return aruco_geometries

def create_point_cloud(color_raw, depth_raw, camera_intrinsic, dist_coeff): # currently hand-eye calculates from rigid body to rgb camera frame. However, point cloud uses the depth camera frame.
    """
    Creates a point cloud from RGB and depth images after correcting for lens distortion.
    """

    # --- Create the RGBD image with the undistorted color image ---
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw,  # Use the corrected color image
        depth_raw,
        depth_scale=1000.0,
        depth_trunc=5.0,
        convert_rgb_to_intensity=False)

    # --- Create the final point cloud (without the incorrect parameter) ---
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic=camera_intrinsic
    )
    return pcd

def main():
    parser = argparse.ArgumentParser(description='Convert RGB/D images to a point cloud and transform it into the OptiTrack world frame.')
    parser.add_argument('--rgb', required=True, help='Path to RGB image')
    parser.add_argument('--depth', required=True, help='Path to depth image')
    parser.add_argument('-o','--optitrack', dest='optitrack', required=True, help='Path to the CSV file that has the OptiTrack data')
    parser.add_argument('-crc', '--camRigidcalib', dest='camRigidcalib', default = 'Kinect_cam', help='Transformation file from Camera to its Rigid Body (T_rigid_cam)')
    parser.add_argument('-c','--cam_body_name', dest='camBodyName', required=True, help='The name of the rigid body attached to the camera in the OptiTrack data')
    parser.add_argument('--output', default='output_world.pcd', help='Output world-frame point cloud file path')
    parser.add_argument('--intrinsic', default='azure_kinect_intrinsics.yml', help='Camera intrinsic parameters file')

    args = parser.parse_args()

    try:
        # 1. Load images and intrinsics
        color_raw, depth_raw = read_rgbd_images(args.rgb, args.depth)
        camera_intrinsic, dist_coeff = load_azure_kinect_intrinsics(color_raw=color_raw, intrinsic_file=args.intrinsic)

        # 2. Create point cloud in the Camera's local coordinate frame
        pcd_camera_frame = create_point_cloud(color_raw, depth_raw, camera_intrinsic, dist_coeff=dist_coeff)

        # 3. Load all rigid body poses from OptiTrack (T_world_rigid)
        all_poses_world = get_all_rigid_body_poses(args.optitrack, frame_number=OPTI_FRAME_NUM)
        if not all_poses_world:
            print("Could not load any rigid body poses from the OptiTrack file. Continuing without them.")

        # NEW: Load all marker positions from OptiTrack
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

        # 9. Visualize ArUco markers in the world frame
        camera_matrix = camera_intrinsic.intrinsic_matrix
        # aruco_frames = get_aruco_geometries_in_world_frame(
        #     color_raw, 
        #     camera_matrix = camera_matrix, 
        #     dist_coeffs= dist_coeff, 
        #     T_world_cam=T_world_cam, 
        #     marker_size=ARUCO_1x1_size
        # )

        # 9. Prepare for visualization
        geometries_to_draw = [pcd_world_frame]

        # geometries_to_draw.extend(aruco_frames)

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

        # NEW: Add a sphere for each marker to visualize its position
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