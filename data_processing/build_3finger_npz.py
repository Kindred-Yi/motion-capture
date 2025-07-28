#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert an OptiTrack CSV containing six hand markers
(back/left/right wrist + thumb/index/middle tips) into the NPZ
format expected by solve_ik_3finger_gripper.py.

This script now saves the fingertip positions in the world coordinate
frame after applying the initial global transforms.

Usage
-----
python build_3finger_npz.py \
        --csv "pick_peanut_butter_jar.csv" \
        --out eef_629.npz
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.linalg import norm
from scipy.spatial.transform import Rotation as R


def _parse_motive_header(csv_path: Path, header_rows: int = 7):
    """
    Read the first *header_rows* lines that Motive writes and construct
    proper column names:   Frame, Time,  <marker>_X, <marker>_Y, <marker>_Z…
    """
    with csv_path.open(newline="") as f:
        raw = [next(f) for _ in range(header_rows)]

    # rows 3 and 6 hold the marker names & coordinate labels respectively
    name_row = next(csv.reader([raw[3]]))
    coord_row = next(csv.reader([raw[6]]))

    names = ["Frame", "Time"]
    for marker_name, coord in zip(name_row[2:], coord_row[2:]):
        names.append(f"{marker_name}_{coord}")

    return names


def _load_csv(csv_path: Path) -> pd.DataFrame:
    names = _parse_motive_header(csv_path)
    # data start immediately after the 7-line header
    return pd.read_csv(csv_path, skiprows=7, names=names)


# POSE COMPUTATION
def get_wrist_pose_from_points(left: np.ndarray, right: np.ndarray, back: np.ndarray):
    """
    Given N×3 arrays of left/right/back wrist positions, return
    (N×3) positions & (N×4) xyzw quaternions. This version builds an
    orthonormal basis from the three points.
    """
    # Initialize arrays to store results for each frame
    positions = np.empty((left.shape[0], 3))
    quats = np.empty((left.shape[0], 4))

    for i in range(left.shape[0]):
        # Get points for the current frame
        p_left = left[i]
        p_right = right[i]
        p_back = back[i]

        # 1. Calculate position as the midpoint of left and right markers
        wrist_position = (p_left + p_right) / 2

        # 2. Calculate orientation by building a local coordinate frame
        # Y-axis: From right to left wrist marker
        vec_y = p_left - p_right
        if norm(vec_y) < 1e-9: # Fallback for degenerate cases
            vec_y = np.array([0, 1, 0])
        y_axis = vec_y / norm(vec_y)

        # Temporary X-axis: From wrist center to back marker
        vec_x_temp = p_back - wrist_position
        if norm(vec_x_temp) < 1e-9: # Fallback
            vec_x_temp = np.array([1, 0, 0])

        # Z-axis: Perpendicular to the plane defined by temp X and Y
        vec_z = np.cross(vec_x_temp, y_axis)
        if norm(vec_z) < 1e-9: # Fallback for collinear points
            vec_z = np.array([0, 0, 1])
        z_axis = vec_z / norm(vec_z)

        # Final X-axis: Orthonormal to Y and Z axes
        vec_x = np.cross(y_axis, z_axis)
        x_axis = vec_x / norm(vec_x)

        # 3. Build rotation matrix and convert to quaternion
        # Columns of the matrix are the axes of the local frame
        rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
        
        # Create a Rotation object from the matrix
        rot_obj = R.from_matrix(rotation_matrix)
        
        # Convert to quaternion [x, y, z, w]
        wrist_quaternion = rot_obj.as_quat()

        # Store results for the current frame
        positions[i] = wrist_position
        quats[i] = wrist_quaternion

    return positions, quats


# MAIN
def build_npz(csv_path: Path, out_path: Path):

    df = _load_csv(csv_path)

    # Apply global coordinate transformations 
    x_cols = [c for c in df.columns if c.endswith("_X")]
    df.loc[:, x_cols] *= -1
    y_cols = [c for c in df.columns if c.endswith("_Y")]
    df.loc[:, y_cols] *= -1
    z_cols = [c for c in df.columns if c.endswith("_Z")]
    df.loc[:, z_cols] *= -1

    df.loc[:, y_cols] += 0.5

    # marker column names 
    prefix = "tesollo3f:right_hand_"
    cols = {
        "left_wrist":  [f"{prefix}left_wrist_{c}"   for c in "XYZ"],
        "right_wrist": [f"{prefix}right_wrist_{c}"  for c in "XYZ"],
        "back_wrist":  [f"{prefix}back_wrist_{c}"   for c in "XYZ"],
        "index_tip":   [f"{prefix}index_{c}"        for c in "XYZ"],
        "middle_tip":  [f"{prefix}middle_{c}"       for c in "XYZ"],
        "thumb_tip":   [f"{prefix}thumb_{c}"        for c in "XYZ"],
    }

    # sanity-check that every column exists (raise helpful message otherwise)
    missing = [c for trio in cols.values() for c in trio if c not in df.columns]
    if missing:
        raise ValueError(
            "CSV is missing expected columns:\n" + "\n".join(missing)
        )

    # compute wrist pose 
    left, right, back = (df[cols[k]].to_numpy() for k in
                         ("left_wrist", "right_wrist", "back_wrist"))
    # Use the new function to compute wrist pose
    pos, quat = get_wrist_pose_from_points(left, right, back)
    wrist_tfs = np.hstack([pos, quat])      # (N,7)

    # fingertips
    # Get fingertip positions in the (globally transformed) world frame
    tips_world = np.stack(
        [df[cols[k]].to_numpy() for k in ("index_tip",
                                          "middle_tip",
                                          "thumb_tip")],
        axis=1,                                    # (N,3,3)
    )
    
    # Per user request, we no longer convert to a local frame.
    # The output will contain the world-frame fingertip positions.
    print("Fingertip coordinates will be saved in the world frame.")

    # save npz
    np.savez(out_path,
             right_wrist_tfs=wrist_tfs.astype(np.float32),
             right_tip_poses=tips_world.astype(np.float32)) # Saving tips_world

    # console summary
    print(f"✓   Saved   {out_path.name}")
    print(f"   Frames           : {len(wrist_tfs)}")
    print(f"   Wrist pose shape   : {wrist_tfs.shape}")
    print(f"   Fingertip shape    : {tips_world.shape}")
    print(f"   Pos range (m)      : "
          f"X[{pos[:,0].min():.3f},{pos[:,0].max():.3f}] "
          f"Y[{pos[:,1].min():.3f},{pos[:,1].max():.3f}] "
          f"Z[{pos[:,2].min():.3f},{pos[:,2].max():.3f}]")
    print("   Quat L2-norm mean :", np.mean(np.linalg.norm(quat, axis=1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Motive CSV to 3-finger gripper NPZ")
    parser.add_argument("--csv", required=True, type=Path,
                        help="input CSV exported by Motive")
    parser.add_argument("--out", default="eef_traj_3finger.npz", type=Path,
                        help="output NPZ path")

    args = parser.parse_args()
    build_npz(args.csv, args.out)
