#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import os
from pyk4a import PyK4A, Config, CalibrationType

# ---------- 0) Argument parser ----------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Estimate ArUco GridBoard pose from saved RGB image using Azure Kinect intrinsics."
    )
    parser.add_argument(
        "--rgb", required=True,
        help="Path to the RGB image (e.g. color.png)"
    )
    # parser.add_argument(
    #     "--depth", required=True,
    #     help="(Optional) Path to the depth image (e.g. depth.png or depth.npy)"
    # )
    parser.add_argument(
        '--skip-update-intrinsics', dest='update_intrinsics', action='store_false',
        help="Use this flag to skip connecting to the camera and read intrinsics from the file instead."
    )
    parser.add_argument(
        '-f', '--folder',
        dest='folder_path',
        default='.',
        help='The destination folder path (e.g., /home/user/documents or C:\\Users\\user\\Documents)'
    )

    parser.add_argument(
        '-n', '--name',
        dest='file_name',
        default='azure_kinect_extrinsics.yml',
        help='The name of the file (e.g., report.txt)'
    )

    return parser.parse_args()

# ---------- 0) GridBoard size ----------

# NOTE: Just from observation and trial and error, seems like changing MARKER_LENGTH changes how far the marker is from the camera (like forward and backward),
#       while changing the MARKER_SEPARATION kind of shifts it left, right, up, or down to make, which can be used to move the marker to the correct position if it is offset (like nudging it to the side for better placement)

# GOOD VALUES HERE
# ARUCO_DICT_NAME   = cv2.aruco.DICT_6X6_250
# MARKER_LENGTH     = 0.0355   # meters
# MARKER_SEPARATION = 0.0093  # meters
# BOARD_ROWS        = 4
# BOARD_COLS        = 6

# # Trying out differing values
# ARUCO_DICT_NAME   = cv2.aruco.DICT_6X6_250
# MARKER_LENGTH     = 0.0358   # meters
# MARKER_SEPARATION = 0.0092  # meters
# BOARD_ROWS        = 4
# BOARD_COLS        = 6

# Trying out differing values
# ARUCO_DICT_NAME   = cv2.aruco.DICT_6X6_250
# MARKER_LENGTH     = 0.0355   # meters
# MARKER_SEPARATION = 0.0095  # meters
# BOARD_ROWS        = 4
# BOARD_COLS        = 6

# Trying out differing values
ARUCO_DICT_NAME   = cv2.aruco.DICT_6X6_250
MARKER_LENGTH     = 0.026  # meters
MARKER_SEPARATION = 0.017  # meters
BOARD_ROWS        = 1
BOARD_COLS        = 1

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
board = cv2.aruco.GridBoard(
    (BOARD_COLS, BOARD_ROWS),
    MARKER_LENGTH,
    MARKER_SEPARATION,
    aruco_dict
)

def main():
    args = parse_args()

    # ---------- 1) Read intrinsics ----------
    if(args.update_intrinsics):
        k4a = PyK4A(Config())
        k4a.start()

        camera_matrix = k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
        dist_coeffs   = k4a.calibration.get_distortion_coefficients(
                        CalibrationType.COLOR)[:5]

        print("Camera matrix:\n", camera_matrix)
        print("Distortion coeffs:", dist_coeffs.ravel())

        fs = cv2.FileStorage("azure_kinect_intrinsics.yml", cv2.FILE_STORAGE_WRITE)
        fs.write("cameraMatrix", camera_matrix)
        fs.write("distCoeffs",  dist_coeffs)
        fs.release()
        print("✅ Saved to azure_kinect_intrinsics.yml")
    else:
        fs = cv2.FileStorage("azure_kinect_intrinsics.yml", cv2.FILE_STORAGE_READ)
        camera_matrix = fs.getNode("cameraMatrix").mat()
        dist_coeffs   = fs.getNode("distCoeffs").mat()
        fs.release()

        if camera_matrix is None or dist_coeffs is None:
            raise RuntimeError("❌ Failed to read intrinsics from azure_kinect_intrinsics.yml")

    # ---------- 2) Load RGB image ----------
    frame = cv2.imread(args.rgb)
    if frame is None:
        raise RuntimeError(f"❌ Failed to read RGB image: {args.rgb}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------- 3) Detect markers ----------
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        raise RuntimeError("❌ No ArUco markers detected")

    # ---------- 4) Estimate pose for single marker ----------
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, MARKER_LENGTH, camera_matrix, dist_coeffs
    )
    if rvecs is None or len(rvecs) == 0:
        raise RuntimeError("❌ Pose estimation failed")

    # Use the first marker (assuming only one is visible)
    rvec_board = rvecs[0].reshape(3, 1)
    tvec_board = tvecs[0].reshape(3, 1)


    frame = frame.copy()
    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_board, tvec_board, 0.10)

    cv2.imshow("Pose", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ---------- 5) Compute camera in board frame ----------
    R_b2c, _ = cv2.Rodrigues(rvec_board)
    R_c2b    = R_b2c.T
    t_cam_in_board = -R_c2b @ tvec_board
    rvec_c2b, _    = cv2.Rodrigues(R_c2b)

    print("Camera in Board CS  rvec:", rvec_c2b.ravel())
    print("Camera in Board CS  tvec:", t_cam_in_board.ravel(), "m")

    target_id = 18
    if target_id in ids:
        idx = np.where(ids == target_id)[0][0]
        rvec_board = rvecs[idx]
        tvec_board = tvecs[idx]
    else:
        raise RuntimeError(f"❌ Marker ID {target_id} not found")

    # ---------- 6) Save extrinsics ----------
    full_path = os.path.join(args.folder_path, args.file_name)
    fs_ext = cv2.FileStorage(full_path, cv2.FILE_STORAGE_WRITE)
    fs_ext.write("rvec", rvec_board)
    fs_ext.write("tvec", tvec_board)
    fs_ext.release()
    print("✅ Saved to {full_path}")

    # # ---------- 7) (Optional) Load depth ----------
    # if args.depth:
    #     if args.depth.endswith(".npy"):
    #         depth = np.load(args.depth)
    #     else:
    #         depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
    #     if depth is None:
    #         raise RuntimeError(f"❌ Failed to load depth image: {args.depth}")
    #     print("✅ Depth image loaded:", depth.shape)

if __name__ == "__main__":
    main()
