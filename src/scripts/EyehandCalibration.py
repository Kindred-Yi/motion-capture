#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import os
import re
import sys

# Assuming 'parser.py' with RigidBody and extract_rigid_body_data is in the same directory
from parser import RigidBody, extract_rigid_body_data

def load_target_to_camera_poses(input_dir):
    """
    Loads calibration target poses from all YML files in a directory.

    This function looks for a sub-directory named 'calibration_results', finds
    all .yml files, sorts them numerically, and extracts the rotation and
    translation vectors from each file.

    Args:
        input_dir (str): The path to the PARENT directory of the 'calibration_results' folder.

    Returns:
        A tuple containing two lists:
        - R_target2cam (list): A list of 3x3 rotation matrices.
        - t_target2cam (list): A list of 3x1 translation vectors.
    """
    R_target2cam_list, t_target2cam_list = [], []
    calib_dir = os.path.join(input_dir, 'calibration_results')

    if not os.path.isdir(calib_dir):
        raise FileNotFoundError(f"❌ Sub-directory 'calibration_results' not found at: {calib_dir}")

    # Find and sort all .yml files numerically
    try:
        yml_files = [f for f in os.listdir(calib_dir) if f.lower().endswith('.yml')]
        yml_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))
    except (TypeError, AttributeError):
        print(f"Warning: Could not sort YML files numerically in '{calib_dir}'.")
        sys.exit(1)

    if not yml_files:
        raise FileNotFoundError(f"❌ No .yml files found in '{calib_dir}'")

    print(f"✅ Found {len(yml_files)} YML pose files to process in '{calib_dir}': {yml_files}")

    # Loop through each file and extract its pose
    for filename in yml_files:
        filepath = os.path.join(calib_dir, filename)
        fs = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
        
        rvec_node = fs.getNode("rvec")
        tvec_node = fs.getNode("tvec")

        if rvec_node.empty() or tvec_node.empty():
            print(f"⚠️ Warning: Could not find pose nodes in '{filename}'. Skipping.")
            fs.release()
            sys.exit(1)

        rvec = rvec_node.mat()
        tvec = tvec_node.mat()
        
        # Convert rotation vector to a 3x3 rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        R_target2cam_list.append(R)
        t_target2cam_list.append(tvec)
        
        fs.release()

    print(f"✅ Loaded {len(R_target2cam_list)} target-to-camera poses.")
    return R_target2cam_list, t_target2cam_list

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

def get_cam_rigid_to_world(input_dir, kinect_rigid_body):
    """
    Loads motion capture data from all CSV files in the 'optitrack' sub-directory.
    """
    R_rigid2world, t_rigid2world = [], []
    optitrack_dir = os.path.join(input_dir, 'optitrack')

    if not os.path.isdir(optitrack_dir):
        raise FileNotFoundError(f"❌ Sub-directory 'optitrack' not found at: {optitrack_dir}")

    try:
        csv_files = [f for f in os.listdir(optitrack_dir) if f.lower().endswith('.csv')]
        csv_files.sort(key=lambda f: int(re.search(r'\d+', f).group()))
    except (TypeError, AttributeError):
        print(f"Warning: Could not sort CSV files numerically in '{optitrack_dir}'")
        sys.exit(1)

    if not csv_files:
        raise FileNotFoundError(f"❌ No CSV files found in '{optitrack_dir}'")
        
    print(f"✅ Found {len(csv_files)} CSV files to process in '{optitrack_dir}': {csv_files}")

    for filename in csv_files:
        filepath = os.path.join(optitrack_dir, filename)
        mocap_data = extract_rigid_body_data(filepath)

        if not mocap_data or kinect_rigid_body not in mocap_data:
            print(f"⚠️ Warning: Could not find rigid body '{kinect_rigid_body}' in file '{filename}'. Skipping.")
            sys.exit(1)

        body = mocap_data[kinect_rigid_body]
        # Frame numbers are 1-based in your parser, but arrays are 0-based.
        # It's safer to get the first valid frame.
        position = body.get_position_at_frame(1) 
        rotation_q = body.get_rotation_at_frame(1)

        if position is None or rotation_q is None:
            print(f"⚠️ Warning: No data found in the first frame of '{filename}'. Skipping.")
            sys.exit(1)
            
        R = quaternion_to_rotation_matrix(rotation_q)
        t = np.array([[position['x']], [position['y']], [position['z']]])
        R_rigid2world.append(R)
        t_rigid2world.append(t)

    print(f"✅ Loaded {len(R_rigid2world)} gripper-to-base poses.")
    return R_rigid2world, t_rigid2world

def perform_eye_in_hand_calibration(input_dir, R_rigid2world, t_rigid2world, R_target2cam, t_target2cam):
    """
    Performs the eye-in-hand calibration and saves the result in a format
    readable by the 'load_camera_extrinsics' function.
    """
    print("\nPerforming Eye-in-Hand calibration...")
    
    # This calculates the transformation from the camera to the rigid body (the "gripper")
    R_cam2rigid, t_cam2rigid = cv2.calibrateHandEye(
        R_gripper2base=R_rigid2world,
        t_gripper2base=t_rigid2world,
        R_target2cam=R_target2cam,
        t_target2cam=t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    print("✅ Calibration successful!")
    print("\n--- Result: Camera to Rigid Body Transformation ---")
    print("Rotation Matrix (R_cam2rigid):\n", R_cam2rigid)
    print("\nTranslation Vector (t_cam2rigid) [meters]:\n", t_cam2rigid)
    
    # --- CHANGED SECTION: Save in the required format ---

    # Convert the resulting rotation matrix back to a rotation vector for saving
    rvec_cam2rigid, _ = cv2.Rodrigues(R_cam2rigid)
    
    # Save the result to the standard extrinsics file
    output_filename = "hand_eye_calibration_result.yml"
    full_path = os.path.join(input_dir, output_filename)
    print(f"\nSaving result to '{full_path}'...")
    fs_out = cv2.FileStorage(full_path, cv2.FILE_STORAGE_WRITE)
    fs_out.write("rvec", rvec_cam2rigid) # Save as 'rvec'
    fs_out.write("tvec", t_cam2rigid)   # Save as 'tvec'
    fs_out.release()
    print(f"\n✅ Result saved to {output_filename}")
    
    # Return the filename for verification
    return output_filename
def main():
    parser = argparse.ArgumentParser(description="Perform eye-in-hand calibration.")
    parser.add_argument(
        "--inputs",
        required=True,
        help="Path to the folder containing the 'calibration_results' and 'optitrack' sub-folders."
    )
    parser.add_argument(
        "--kinect_rigid_body",
        default="Kinect_cam",
        help="The name of the rigid body in OptiTrack corresponding to the Kinect camera."
    )
    args = parser.parse_args()

    # 1. Load the poses of the target relative to the camera
    R_target2cam, t_target2cam = load_target_to_camera_poses(args.inputs)

    # 2. Get the poses of the gripper relative to the robot base
    R_rigid2world, t_rigid2world = get_cam_rigid_to_world(args.inputs, args.kinect_rigid_body)

    # 3. Check for matching number of poses
    if len(R_target2cam) != len(R_rigid2world):
        print("\n\n❌ FATAL ERROR: The number of camera poses and gripper poses must be equal.")
        print(f"  Found {len(R_target2cam)} camera poses and {len(R_rigid2world)} gripper poses.")
        exit()
    
    # 4. Perform the calibration
    perform_eye_in_hand_calibration(args.inputs, R_rigid2world, t_rigid2world, R_target2cam, t_target2cam)

if __name__ == "__main__":
    main()