#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import os
from pyk4a import PyK4A, Config, CalibrationType, ColorControlMode
from pyk4a import ColorResolution # Import ColorResolution

# --- Define ArUco Board details ---
ARUCO_DICT_NAME = cv2.aruco.DICT_6X6_250
BOARD_ROWS = 4
BOARD_COLS = 6
MARKER_LENGTH = 0.0354  # meters
MARKER_SEPARATION = 0.0091  # meters

EXPOSURE = 8310  # microseconds, adjust as needed

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
board = cv2.aruco.GridBoard((BOARD_COLS, BOARD_ROWS), MARKER_LENGTH, MARKER_SEPARATION, aruco_dict)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

def main():
    parser = argparse.ArgumentParser(description="Capture and validate images for calibration.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the folder where images will be saved."
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="720p",
        choices=["720p", "1080p", "1440p", "1536p", "2160p", "3072p"],
        help="Color camera resolution (e.g., 720p, 1080p)."
    )

    
    args = parser.parse_args()

    # --- Create output directories ---
    color_dir = os.path.join(args.output, "color")
    depth_dir = os.path.join(args.output, "depth")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    print(f"✅ Saving images to: {args.output}")

    # --- Map resolution string to PyK4A enum ---
    resolution_map = {
        "720p": ColorResolution.RES_720P,
        "1080p": ColorResolution.RES_1080P,
        "1440p": ColorResolution.RES_1440P,
        "1536p": ColorResolution.RES_1536P,
        "2160p": ColorResolution.RES_2160P,
        "3072p": ColorResolution.RES_3072P,
    }
    selected_resolution = resolution_map.get(args.resolution.lower())

    if selected_resolution is None:
        print(f"❌ Invalid resolution: {args.resolution}. Using default 720p.")
        selected_resolution = ColorResolution.RES_720P

    # --- Initialize Camera and get intrinsics ---
    # Pass the selected resolution to the Config object
    k4a = PyK4A(Config(color_resolution=selected_resolution))

    k4a.start()

    k4a.exposure = EXPOSURE  # Set exposure time

    # Save camera intrinsics for the second script
    camera_matrix = k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
    dist_coeffs = k4a.calibration.get_distortion_coefficients(CalibrationType.COLOR)[:5]
    
    fs = cv2.FileStorage("azure_kinect_intrinsics.yml", cv2.FILE_STORAGE_WRITE)
    fs.write("cameraMatrix", camera_matrix)
    fs.write("distCoeffs", dist_coeffs)
    fs.release()
    print("✅ Camera intrinsics saved to azure_kinect_intrinsics.yml")

    img_count = 0
    while True:
        # --- Get a capture ---
        capture = k4a.get_capture()
        if capture.color is None or capture.transformed_depth is None:
            continue
        
        frame = capture.color[:, :, :3] # Get BGR
        depth_image = capture.transformed_depth # Get 16-bit depth image
        
        # --- Display instructions on the live feed ---
        display_frame = frame.copy()
        cv2.putText(display_frame, "Press [SPACE] to capture, [Q] to quit", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Feed", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            print("\nAttempting to capture...")
            
            # --- Detect markers ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            
            valid_pose = False
            if ids is not None and len(ids) > 0:
                valid, rvec, tvec = cv2.aruco.estimatePoseBoard(
                    corners, ids, board, camera_matrix, dist_coeffs, None, None
                )
                if valid > 0:
                    valid_pose = True

            if valid_pose:
                # --- Pose estimated, ask user for confirmation ---
                preview_frame = frame.copy()
                cv2.aruco.drawDetectedMarkers(preview_frame, corners, ids)
                cv2.drawFrameAxes(preview_frame, camera_matrix, dist_coeffs, rvec, tvec, 0.10)
                cv2.putText(preview_frame, "SAVE [S] or DISCARD [D]?", (10, 70), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Confirm Capture", preview_frame)
                
                while True:
                    confirm_key = cv2.waitKey(0) & 0xFF
                    if confirm_key == ord('s'):
                        print(f"✅  User approved. Saving image {img_count}.")
                        
                        # Define paths with .png for both
                        color_path = os.path.join(color_dir, f"{img_count:04d}_color.png")
                        depth_path = os.path.join(depth_dir, f"{img_count:04d}_depth.png")
                        
                        # Save the original color frame
                        cv2.imwrite(color_path, frame)
                        
                        # Save the 16-bit depth image as a PNG
                        cv2.imwrite(depth_path, depth_image)
                        
                        img_count += 1
                        break
                        
                    elif confirm_key == ord('d'):
                        print("❌  User discarded capture.")
                        break
                cv2.destroyWindow("Confirm Capture")

            else:
                # --- No markers or pose found, show the user why ---
                print("❌ Failure: Board not detected or pose invalid. Press any key to continue.")
                cv2.imshow("Capture Failed", frame)
                cv2.waitKey(0)
                cv2.destroyWindow("Capture Failed")

    k4a.stop()
    cv2.destroyAllWindows()
    print("\nProgram finished.")

if __name__ == "__main__":
    main()