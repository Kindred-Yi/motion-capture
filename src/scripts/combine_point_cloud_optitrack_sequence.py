import open3d as o3d
import numpy as np
import os
import sys
import argparse
import cv2
import cv2.aruco as aruco
import time

from parser import RigidBody, extract_rigid_body_data
from mocap_classes import Marker, extract_marker_data

ARUCO_1x1_size = 0.0353  # meters, size of the ArUco marker in meters
OPTI_SCALE = 1  # OptiTrack scale factor when messed up calibration, usually should be 1


def load_azure_kinect_intrinsics(intrinsic_file="azure_kinect_intrinsics.yml"):
    """Load Azure Kinect intrinsic parameters and distortion coefficients."""
    try:
        fs = cv2.FileStorage(intrinsic_file, cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode("cameraMatrix").mat()
        dist_coeffs = fs.getNode("distCoeffs").mat()
        fs.release()

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width=1280, height=720,
            fx=camera_matrix[0, 0], fy=camera_matrix[1, 1],
            cx=camera_matrix[0, 2], cy=camera_matrix[1, 2]
        )
        return intrinsic, camera_matrix, dist_coeffs
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"⚠️ Warning in {fname} at line {exc_tb.tb_lineno}: Could not load intrinsics. Error: {e}")
        return None, None, None

def read_rgbd_images(rgb_path, depth_path):
    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        print(f"Warning: RGB or depth image not found for frame. Skipping. Searched for:")
        print(f"  - {rgb_path}")
        print(f"  - {depth_path}")
        return None, None

    color_raw = o3d.io.read_image(rgb_path)
    depth_raw = o3d.io.read_image(depth_path)
    return color_raw, depth_raw

def load_camera_extrinsics(extrinsic_file="azure_kinect_extrinsics.yml"):
    """Loads a 4x4 transformation matrix from a YAML file."""
    if not os.path.exists(extrinsic_file):
        raise FileNotFoundError(f"Extrinsics file not found: {extrinsic_file}")
    fs = cv2.FileStorage(extrinsic_file, cv2.FILE_STORAGE_READ)
    rvec = fs.getNode("rvec").mat()
    tvec = fs.getNode("tvec").mat()
    fs.release()
    if rvec is None or tvec is None:
        raise ValueError(f"Failed to read rvec/tvec from extrinsics file: {extrinsic_file}")
    R, _ = cv2.Rodrigues(rvec)
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = tvec.ravel()
    return transformation_matrix

def quaternion_to_rotation_matrix(q):
    """Converts a quaternion into a 3x3 rotation matrix."""
    w, x, y, z = q['w'], q['x'], q['y'], q['z']
    x2, y2, z2 = x * x, y * y, z * z
    R = np.array([
        [1 - 2*y2 - 2*z2,   2*x*y - 2*z*w,   2*x*z + 2*y*w],
        [2*x*y + 2*z*w,   1 - 2*x2 - 2*z2,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,   2*y*z + 2*x*w,   1 - 2*x2 - 2*y2]
    ])
    return R

def get_all_rigid_body_poses(mocap_data, frame_number):
    """Gets world poses of all rigid bodies for a specific frame from pre-loaded data."""
    all_poses = {}
    for name, body in mocap_data.items():
        position = body.get_position_at_frame(frame_number)
        rotation_q = body.get_rotation_at_frame(frame_number)
        if position is None or rotation_q is None:
            continue
        T_world_rigid = np.eye(4)
        T_world_rigid[:3, :3] = quaternion_to_rotation_matrix(rotation_q)
        T_world_rigid[:3, 3] = [position['x'] / OPTI_SCALE, position['y'] / OPTI_SCALE, position['z'] / OPTI_SCALE]
        all_poses[name] = T_world_rigid
    return all_poses

def get_all_marker_positions(marker_data, frame_number):
    """Gets world positions of all markers for a specific frame from pre-loaded data."""
    all_positions = {}
    for name, marker in marker_data.items():
        position = marker.get_position_at_frame(frame_number)
        if position is None or any(v is None for v in position.values()):
            continue
        all_positions[name] = np.array([position['x'] / OPTI_SCALE, position['y'] / OPTI_SCALE, position['z'] / OPTI_SCALE])
    return all_positions

# def detect_and_estimate_aruco_poses(image, camera_matrix, dist_coeffs, marker_size=0.05):
#     """Detects ArUco markers and estimates their poses."""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     aruco_dict = aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
#     parameters = aruco.DetectorParameters()
#     corners, ids, _ = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
#     if ids is not None and len(ids) > 0:
#         rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_size, camera_matrix, dist_coeffs)
#         return rvecs, tvecs, ids
#     return None, None, None

# def get_aruco_geometries_in_world_frame(color_raw, camera_matrix, dist_coeffs, T_world_cam, marker_size):
#     """Returns a list of Open3D geometries for detected ArUco tags."""
#     aruco_geometries = []
#     color_cv_image = cv2.cvtColor(np.asarray(color_raw), cv2.COLOR_RGB2BGR)
#     rvecs, tvecs, ids = detect_and_estimate_aruco_poses(color_cv_image, camera_matrix, dist_coeffs, marker_size)
#     if ids is not None:
#         for i, _ in enumerate(ids):
#             T_cam_aruco = np.eye(4)
#             R_cam_aruco, _ = cv2.Rodrigues(rvecs[i])
#             T_cam_aruco[:3, :3] = R_cam_aruco
#             T_cam_aruco[:3, 3] = tvecs[i].flatten()
#             T_world_aruco = T_world_cam @ T_cam_aruco
#             aruco_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=marker_size)
#             aruco_frame.transform(T_world_aruco)
#             aruco_geometries.append(aruco_frame)
#     return aruco_geometries

def create_point_cloud(color_raw, depth_raw, camera_intrinsic, dist_coeffs):
    """Creates a point cloud from RGB-D images after correcting for lens distortion."""
    color_cv = np.asarray(color_raw)
    camera_matrix = camera_intrinsic.intrinsic_matrix
    undistorted_color_cv = cv2.undistort(color_cv, camera_matrix, dist_coeffs)
    color_raw_undistorted = o3d.geometry.Image(undistorted_color_cv)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw_undistorted, depth_raw,
        depth_scale=1000.0, depth_trunc=5.0, convert_rgb_to_intensity=False)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsic)
    return pcd

def main():
    parser = argparse.ArgumentParser(description='Animate point clouds and OptiTrack data frame by frame.')
    # MODIFIED: Changed from single files to folders
    parser.add_argument('-c','--color_folder', dest='color_folder', required=True, help='Path to the folder containing color images.')
    parser.add_argument('-d', '--depth_folder', dest='depth_folder', required=True, help='Path to the folder containing depth images.')
    parser.add_argument('-o','--optitrack',dest='optitrack', required=True, help='Path to the OptiTrack CSV file.')
    parser.add_argument('-crc', '--camRigidcalib', dest='camRigidcalib', required=True, help='Transformation file from Camera to its Rigid Body.')
    parser.add_argument('--cam_body_name', default='Kinect_cam', help='Name of the rigid body attached to the camera.')
    parser.add_argument('--intrinsic', default='azure_kinect_intrinsics.yml', help='Camera intrinsic parameters file.')
    # MODIFIED: Start frame is now an offset for OptiTrack data, end_frame is removed.
    parser.add_argument('--start_frame', type=int, default=1, help='The frame number in the OptiTrack data that corresponds to the FIRST image file.')

    args = parser.parse_args()

    # --- Pre-load data that doesn't change per frame ---
    camera_intrinsic, camera_matrix_cv, dist_coeffs_cv = load_azure_kinect_intrinsics(args.intrinsic)
    if camera_intrinsic is None:
        sys.exit("Error: Could not load camera intrinsics. Exiting.")

    T_rigid_cam = load_camera_extrinsics(args.camRigidcalib)
    mocap_data = extract_rigid_body_data(args.optitrack)
    marker_data = extract_marker_data(args.optitrack)
    
    # --- NEW: Read and sort all image files from folders ---
    try:
        color_files = sorted([f for f in os.listdir(args.color_folder) if f.endswith(('.jpg', '.png'))])
        depth_files = sorted([f for f in os.listdir(args.depth_folder) if f.endswith(('.jpg', '.png'))])
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find folder: {e.filename}")
        sys.exit(1)

    print(f"✅ Found {len(color_files)} color images and {len(depth_files)} depth images in the specified folders.")
    num_frames = min(len(color_files), len(depth_files))
    if num_frames == 0:
        sys.exit("Error: No image files found in the specified color or depth folders.")
        
    print(f"✅ Found {num_frames} image pairs to process.")

    # --- Setup Visualizer ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Frame-by-Frame Visualization")
    
    is_first_frame = True
    try:
        # --- Main Processing Loop ---
        for i in range(num_frames):
            # Calculate the corresponding frame number for OptiTrack data
            optitrack_frame_num = args.start_frame + i
            print(f"\n--- Processing Image Index: {i} (OptiTrack Frame: {optitrack_frame_num}) ---")
            
            # 1. Construct file paths for the current frame using sorted lists
            color_path = os.path.join(args.color_folder, color_files[i])
            depth_path = os.path.join(args.depth_folder, depth_files[i])

            # 2. Load frame-specific data
            color_raw, depth_raw = read_rgbd_images(color_path, depth_path)
            if color_raw is None: continue

            all_poses_world = get_all_rigid_body_poses(mocap_data, optitrack_frame_num)
            all_marker_positions_world = get_all_marker_positions(marker_data, optitrack_frame_num)

            if args.cam_body_name not in all_poses_world:
                print(f"Warning: Camera rigid body '{args.cam_body_name}' not in data for frame {optitrack_frame_num}. Skipping.")
                continue

            # 3. Perform calculations for the current frame
            T_world_camRigid = all_poses_world[args.cam_body_name]
            T_world_cam = T_world_camRigid @ T_rigid_cam
            
            pcd_camera_frame = create_point_cloud(color_raw, depth_raw, camera_intrinsic, dist_coeffs_cv)
            pcd_world_frame = pcd_camera_frame.transform(T_world_cam)
            
            # aruco_frames = get_aruco_geometries_in_world_frame(
            #     color_raw, camera_matrix_cv, dist_coeffs_cv, T_world_cam, ARUCO_1x1_size)

            # 4. Assemble all geometries for visualization
            geometries_to_draw = [pcd_world_frame]
            # geometries_to_draw.extend(aruco_frames)
            geometries_to_draw.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4))
            
            for name, T_world_rigid in all_poses_world.items():
                rb_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                rb_frame.transform(T_world_rigid)
                geometries_to_draw.append(rb_frame)

            for name, position in all_marker_positions_world.items():
                marker_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                marker_sphere.paint_uniform_color([1.0, 0.0, 0.0])
                marker_sphere.translate(position)
                geometries_to_draw.append(marker_sphere)

            # 5. Update the visualizer window
            vis.clear_geometries()
            for geom in geometries_to_draw:
                vis.add_geometry(geom, reset_bounding_box=is_first_frame)
            
            is_first_frame = False
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.05) 

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        print(f"❌ An error occurred in file '{fname}' at line {exc_tb.tb_lineno}:")
        print(f"   Error Type: {exc_type.__name__}")
        print(f"   Error Details: {e}")
    finally:
        print("--- Visualization finished. Closing window. ---")
        vis.destroy_window()

if __name__ == "__main__":
    main()