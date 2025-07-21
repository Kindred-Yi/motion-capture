import open3d as o3d
import numpy as np
import os
import sys
import argparse
import cv2

def load_azure_kinect_intrinsics(intrinsic_file="azure_kinect_intrinsics.yml"):
    """Load Azure Kinect intrinsic parameters"""
    try:
        fs = cv2.FileStorage(intrinsic_file, cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode("cameraMatrix").mat()
        dist_coeffs = fs.getNode("distCoeffs").mat()
        fs.release()

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width=1280,
            height=720,
            fx=camera_matrix[0, 0],
            fy=camera_matrix[1, 1],
            cx=camera_matrix[0, 2],
            cy=camera_matrix[1, 2]
        )
        return intrinsic
    except:
        print("Warning: Unable to load Azure Kinect intrinsics, using default parameters")
        return None

def read_rgbd_images(rgb_path, depth_path):
    if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
        raise FileNotFoundError(f"RGB or depth image not found at {rgb_path} or {depth_path}")

    color_raw = o3d.io.read_image(rgb_path)
    depth_raw = o3d.io.read_image(depth_path)

    return color_raw, depth_raw

def load_camera_extrinsics(extrinsic_file="azure_kinect_extrinsics.yml"):
    if not os.path.exists(extrinsic_file):
        raise FileNotFoundError(f"Extrinsics file not found: {extrinsic_file}")

    fs = cv2.FileStorage(extrinsic_file, cv2.FILE_STORAGE_READ)
    rvec = fs.getNode("rvec").mat()
    tvec = fs.getNode("tvec").mat()
    fs.release()

    if rvec is None or tvec is None:
        raise ValueError("Failed to read 'rvec' or 'tvec' from extrinsics file")

    R, _ = cv2.Rodrigues(rvec)
    camera_extrinsics = np.eye(4)
    camera_extrinsics[:3, :3] = R
    camera_extrinsics[:3, 3] = tvec.ravel()

    return camera_extrinsics

def create_point_cloud(color_raw, depth_raw, camera_intrinsic, camera_extrinsic):
    if camera_intrinsic is None:
        camera_intrinsic = load_azure_kinect_intrinsics()

    if camera_intrinsic is None:
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic()
        camera_intrinsic.set_intrinsics(
            width=1920, height=1080,
            fx=1000.0, fy=1000.0,
            cx=960.0, cy=540.0
        )
        print("Using estimated Azure Kinect intrinsics, accurate calibration is recommended")

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw,
        depth_raw,
        depth_scale=1000.0,
        depth_trunc=5.0,
        convert_rgb_to_intensity=False)

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic=camera_intrinsic
        )

    return pcd

def main():
    parser = argparse.ArgumentParser(description='Convert RGB and depth images to point cloud')
    parser.add_argument('--rgb', required=True, help='Path to RGB image')
    parser.add_argument('--depth', required=True, help='Path to depth image')
    parser.add_argument('--output', default='output.pcd', help='Output point cloud file path')
    parser.add_argument('--intrinsic', default='azure_kinect_intrinsics.yml', help='Camera intrinsic parameters file')
    parser.add_argument('--extrinsic', default='azure_kinect_extrinsics.yml', help='Camera extrinsic parameters file')

    args = parser.parse_args()

    try:
        color_raw, depth_raw = read_rgbd_images(args.rgb, args.depth)

        camera_intrinsic = None
        if args.intrinsic:
            camera_intrinsic = load_azure_kinect_intrinsics(args.intrinsic)

        
        camera_extrinsics = load_camera_extrinsics(args.extrinsic)

        # Create point cloud using extrinsics, makes it in coordinate system of the Apriltag board
        pcd = create_point_cloud(color_raw, depth_raw, camera_intrinsic, camera_extrinsics)

        # Save point cloud
        o3d.io.write_point_cloud(args.output, pcd)
        print(f"Point cloud saved to {args.output}")
        print(f"Point cloud contains {len(pcd.points)} points")

        # Draw camera_frame
        aruco_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        aruco_sphere.paint_uniform_color([1.0, 0.0, 0.0])  # red
        aruco_sphere.translate([0, 0, 0])  # original origin

        print(camera_extrinsics.shape)
        camera_inv_extrinsics = np.linalg.inv(camera_extrinsics)
        # Draw transformed rigid body as green spehere
        camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        camera_sphere.paint_uniform_color([0.0, 1.0, 0.0])  # green
        camera_sphere.transform(camera_inv_extrinsics)


        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        camera_frame.transform(camera_inv_extrinsics)

        # Coordinate frame for reference
        aruco_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

        # Visualize everything
        o3d.visualization.draw_geometries(
            [pcd, aruco_frame, aruco_sphere, camera_sphere, camera_frame],
            window_name="Transformed Point Cloud with Origins"
        )

    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
