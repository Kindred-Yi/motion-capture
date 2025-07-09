import open3d as o3d
import numpy as np
import os
import sys
import argparse
import cv2
import glob

def load_azure_kinect_intrinsics(intrinsic_file="azure_kinect_intrinsics.yml"):
    """Load Azure Kinect intrinsic parameters from a .yml file."""
    if not os.path.exists(intrinsic_file):
        print(f"Warning: Intrinsics file not found at {intrinsic_file}, using default parameters.")
        return None
    try:
        fs = cv2.FileStorage(intrinsic_file, cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode("cameraMatrix").mat()
        fs.release()

        if camera_matrix is None:
            print(f"Warning: 'cameraMatrix' not found in {intrinsic_file}, using default parameters.")
            return None

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
    except Exception as e:
        print(f"Error loading intrinsics from {intrinsic_file}: {e}")
        print("Warning: Using default parameters.")
        return None

def read_rgbd_images(rgb_path, depth_path):
    """Read RGB and depth images from specified paths."""
    if not os.path.exists(rgb_path):
        raise FileNotFoundError(f"RGB image not found at {rgb_path}")
    if not os.path.exists(depth_path):
        raise FileNotFoundError(f"Depth image not found at {depth_path}")

    color_raw = o3d.io.read_image(rgb_path)
    depth_raw = o3d.io.read_image(depth_path)

    return color_raw, depth_raw

def load_camera_extrinsics(extrinsic_file="azure_kinect_extrinsics.yml"):
    """Load camera extrinsic parameters (pose) from a .yml file."""
    if not os.path.exists(extrinsic_file):
        raise FileNotFoundError(f"Extrinsics file not found: {extrinsic_file}")

    fs = cv2.FileStorage(extrinsic_file, cv2.FILE_STORAGE_READ)
    rvec = fs.getNode("rvec").mat()
    tvec = fs.getNode("tvec").mat()
    fs.release()

    if rvec is None or tvec is None:
        raise ValueError(f"Failed to read 'rvec' or 'tvec' from {extrinsic_file}")

    R, _ = cv2.Rodrigues(rvec)
    camera_extrinsics = np.eye(4)
    camera_extrinsics[:3, :3] = R
    camera_extrinsics[:3, 3] = tvec.ravel()

    return camera_extrinsics

def create_point_cloud(color_raw, depth_raw, camera_intrinsic, camera_extrinsic):
    """Create a point cloud from RGB-D images, intrinsics, and extrinsics."""
    if camera_intrinsic is None:
        print("Using estimated Azure Kinect intrinsics, accurate calibration is recommended")
        camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
            o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault
        )

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw,
        depth_raw,
        depth_scale=1000.0,
        depth_trunc=5.0,
        convert_rgb_to_intensity=False
    )
    
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic=camera_intrinsic,
        extrinsic=np.linalg.inv(camera_extrinsic)
    )

    return pcd

def main():
    """Main function to process directories of images into point clouds."""
    parser = argparse.ArgumentParser(description='Convert RGB and depth images from a folder to point clouds.')
    parser.add_argument('--input', required=True, help='Path to the input directory containing images and extrinsic files.')
    parser.add_argument('--output', required=True, help='Path to the output directory to save .pcd files.')
    parser.add_argument('--intrinsic', default='azure_kinect_intrinsics.yml', help='Path to the single camera intrinsic parameters file.')
    
    parser.add_argument('--rgb_suffix', default='.png', help='Suffix for RGB images (e.g., .jpg, .png).')
    parser.add_argument('--depth_suffix', default='.depth.png', help='Suffix for depth images.')
    parser.add_argument('--extrinsic_suffix', default='.extrinsics.yml', help='Suffix for extrinsic files.')

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    camera_intrinsic = load_azure_kinect_intrinsics(args.intrinsic)
    rgb_files = sorted(glob.glob(os.path.join(args.input, f'*{args.rgb_suffix}')))
    rgb_files = [f for f in rgb_files if args.depth_suffix not in f]

    if not rgb_files:
        print(f"Error: No RGB images with suffix '{args.rgb_suffix}' found in '{args.input}'.")
        sys.exit(1)
        
    print(f"Found {len(rgb_files)} RGB images to process.")

    # This variable will hold the last successfully created point cloud
    last_successful_pcd = None

    for rgb_path in rgb_files:
        base_name = os.path.basename(rgb_path).replace(args.rgb_suffix, '')
        output_pcd_path = os.path.join(args.output, base_name + '.pcd')
        print(f"--- Processing {base_name} ---")

        try:
            # Construct paths for corresponding files
            depth_path = os.path.join(args.input, base_name + args.depth_suffix)
            extrinsic_path = os.path.join(args.input, base_name + args.extrinsic_suffix)

            # --- Success Path ---
            # Try to load all files and create a new point cloud
            color_raw, depth_raw = read_rgbd_images(rgb_path, depth_path)
            camera_extrinsics = load_camera_extrinsics(extrinsic_path)
            pcd = create_point_cloud(color_raw, depth_raw, camera_intrinsic, camera_extrinsics)
            
            o3d.io.write_point_cloud(output_pcd_path, pcd)
            print(f"✅ Successfully saved new point cloud to {output_pcd_path}")
            
            # Update the last successful pcd
            last_successful_pcd = pcd

        except (FileNotFoundError, ValueError) as e:
            # --- Failure Path ---
            # An expected file was missing or invalid
            print(f"⚠️ Warning: {e}")
            if last_successful_pcd is not None:
                # If we have a previous PCD, use it for the current frame
                o3d.io.write_point_cloud(output_pcd_path, last_successful_pcd)
                print(f"↳ Filling gap by duplicating previous frame to {output_pcd_path}")
            else:
                # If this is the first frame and it fails, we can't do anything
                print("↳ Cannot create PCD for the first frame as no previous data exists. Skipping.")

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()