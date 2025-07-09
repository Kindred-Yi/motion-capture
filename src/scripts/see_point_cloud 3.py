import h5py
import numpy as np
import open3d as o3d
import argparse

def visualize_point_cloud(file_path, demo_idx=91, frame_idx=0, save_path=None):
    """
    Visualize and optionally save a point cloud from the HDF5 file.
    
    Args:
        file_path (str): Path to the HDF5 file
        demo_idx (int): Index of the demonstration to visualize
        frame_idx (int): Frame index to visualize
        save_path (str, optional): Path to save the point cloud as PCD file
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # Construct the dataset path
            dataset_path = f'/data/demo_{demo_idx}/obs/pointcloud'
            if dataset_path not in f:
                print(f"Error: Dataset {dataset_path} not found in the HDF5 file")
                return
            
            # Get total frames available
            total_frames = f[dataset_path].shape[0]
            if frame_idx >= total_frames:
                print(f"Error: Frame index {frame_idx} exceeds total frames {total_frames}")
                return
            
            # Read the point cloud data
            pointcloud = f[dataset_path][frame_idx]  # shape: (10000, 6)
            xyz = pointcloud[:, :3]
            rgb = pointcloud[:, 3:]  # Original RGB values

            # Print debug information
            print(f"RGB value range before normalization: min={rgb.min()}, max={rgb.max()}")
            
            # Check if RGB values need normalization
            if rgb.max() > 1:
                rgb = rgb / 255.0  # Normalize only if values are in 0-255 range
            
            print(f"RGB value range after normalization: min={rgb.min()}, max={rgb.max()}")
            print(f"Sample of RGB values (first 5 points):")
            print(rgb[:5])

        # Create Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(rgb)

        # Add coordinate frame for reference
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        # Visualize the point cloud with coordinate frame
        print("\nVisualizing point cloud. Press 'Q' or 'Esc' to close the window.")
        print("Red axis: X, Green axis: Y, Blue axis: Z")
        o3d.visualization.draw_geometries([pcd, coordinate_frame])

        if save_path:
            o3d.io.write_point_cloud(save_path, pcd)
            print(f"Saved point cloud to {save_path}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize point cloud from HDF5 file")
    parser.add_argument("--file", type=str, default="/home/hci-lab/Downloads/hand_wiping_1-14_5actiongap_10000points.hdf5",
                      help="Path to the HDF5 file")
    parser.add_argument("--demo", type=int, default=91,
                      help="Demo index to visualize")
    parser.add_argument("--frame", type=int, default=0,
                      help="Frame index to visualize")
    parser.add_argument("--save", type=str, default=None,
                      help="Path to save the point cloud (optional)")
    
    args = parser.parse_args()
    visualize_point_cloud(args.file, args.demo, args.frame, args.save)
