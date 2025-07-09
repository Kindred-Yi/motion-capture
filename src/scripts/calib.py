#!/usr/bin/env python

from pyk4a import PyK4A, Config, CalibrationType
import cv2
import numpy as np
import time

# ---------- 0) GridBoard size ----------
ARUCO_DICT_NAME  = cv2.aruco.DICT_6X6_250
MARKER_LENGTH    = 0.035   # m
MARKER_SEPARATION= 0.0091  # m
BOARD_ROWS       = 4
BOARD_COLS       = 6
CAPTURE_COUNT    = 50

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
board = cv2.aruco.GridBoard(
    (BOARD_COLS, BOARD_ROWS),
    MARKER_LENGTH,
    MARKER_SEPARATION,
    aruco_dict
)

# ---------- 1) Connect to camera and get intrinsics ----------
k4a = PyK4A(Config())
k4a.start()

# Get camera intrinsics
color_camera_matrix = k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
color_dist_coeffs = k4a.calibration.get_distortion_coefficients(CalibrationType.COLOR)[:5]

print("✅ Camera started and intrinsics loaded.")

# ---------- 2) Capture frames and estimate poses ----------
rvecs = []
tvecs = []
last_frame = None

for i in range(CAPTURE_COUNT):
    print(f"Capturing frame {i + 1}/{CAPTURE_COUNT}...")
    try:
        cap = k4a.get_capture()
        if cap.color is None:
            print("❌ No color frame captured, skipping.")
            continue

        frame = cap.color[:, :, :3]  # BGRA → BGR
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        last_frame = frame # Keep the latest frame for visualization

        # Detect markers
        aruco_params = cv2.aruco.DetectorParameters()
        aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None or len(ids) == 0:
            print("❌ No ArUco markers detected, skipping.")
            continue

        # Estimate board pose
        valid, rvec, tvec = cv2.aruco.estimatePoseBoard(
            corners, ids, board, color_camera_matrix, color_dist_coeffs, None, None)

        if valid > 0:
            rvecs.append(rvec)
            tvecs.append(tvec)
            print(f"✅ Pose captured successfully.")
        else:
            print("❌ Board pose estimation failed, skipping.")

    except Exception as e:
        print(f"An error occurred during capture: {e}")

    time.sleep(0.1) # Small delay between captures

if not rvecs or not tvecs:
    k4a.stop()
    raise RuntimeError("Could not capture any valid poses after 50 attempts.")

# ---------- 3) Average the poses ----------
# Note: Averaging rvecs directly is a simplification. For small variations,
# this is acceptable. The more rigorous method involves converting to
# quaternions or rotation matrices, averaging, and converting back.
avg_rvec = np.mean(rvecs, axis=0)
avg_tvec = np.mean(tvecs, axis=0)

print("\n--- Averaged Pose ---")
print(f"Computed from {len(rvecs)} valid captures.")
print("Averaged rvec:", avg_rvec.ravel())
print("Averaged tvec:", avg_tvec.ravel())

# ---------- 4) Visualize and save results ----------
vis_frame = last_frame.copy() 
cv2.aruco.drawDetectedMarkers(vis_frame, corners, ids) # Show markers from last frame
cv2.drawFrameAxes(vis_frame, color_camera_matrix, color_dist_coeffs,
                    avg_rvec, avg_tvec, 0.10)  # Draw axis from averaged pose

cv2.imshow("Averaged Pose", vis_frame)
print("\nDisplaying final averaged pose. Press any key to save and exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()

k4a.stop()

# Compute camera pose in board coordinate system from averaged vectors
R_b2c, _  = cv2.Rodrigues(avg_rvec)
R_c2b     = R_b2c.T
t_cam_in_board = -R_c2b @ avg_tvec
rvec_c2b, _ = cv2.Rodrigues(R_c2b)

print("\nCamera in Board CS rvec:", rvec_c2b.ravel())
print("Camera in Board CS tvec:", t_cam_in_board.ravel(), "m")

# Save final averaged extrinsics (Board to Camera)
fs_ext = cv2.FileStorage("azure_kinect_extrinsics.yml", cv2.FILE_STORAGE_WRITE)
fs_ext.write("rvec", avg_rvec)
fs_ext.write("tvec", avg_tvec)
fs_ext.release()
print("\n✅ Saved board-to-camera extrinsics to azure_kinect_extrinsics.yml")

# Save final averaged extrinsics (Camera to Board)
fs_ext_c2b = cv2.FileStorage("azure_kinect_extrinsics_cam_to_boardCS.yml", cv2.FILE_STORAGE_WRITE)
fs_ext_c2b.write("rvec_c2b", rvec_c2b)
fs_ext_c2b.write("t_cam_in_board", t_cam_in_board)
fs_ext_c2b.release()
print("✅ Saved camera-to-board extrinsics to azure_kinect_extrinsics_cam_to_boardCS.yml")