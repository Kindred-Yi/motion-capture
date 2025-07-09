#!/usr/bin/env python3

import argparse
import cv2
import numpy as np
import os
from pyk4a import PyK4A, Config, CalibrationType

# --- Define ArUco marker details ---
ARUCO_DICT_NAME = cv2.aruco.DICT_6X6_250
MARKER_LENGTH = 0.0354  # meters
target_marker_id = 12

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_NAME)
aruco_params = cv2.aruco.DetectorParameters()
aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    """
    Adjust brightness and contrast of an image.
    brightness: [-100, 100]
    contrast: [-100, 100]
    """
    beta = brightness
    alpha = 1.0 + (contrast / 100.0)
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def main():
    parser = argparse.ArgumentParser(description="Capture and validate images for calibration.")
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the folder where images will be saved."
    )
    args = parser.parse_args()

    # --- Create output directories ---
    color_dir = os.path.join(args.output, "color")
    depth_dir = os.path.join(args.output, "depth")
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    print(f"✅ Saving images to: {args.output}")

    # --- Initialize Camera and get intrinsics ---
    k4a = PyK4A(Config())
    k4a.start()
    
    camera_matrix = k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
    dist_coeffs = k4a.calibration.get_distortion_coefficients(CalibrationType.COLOR)[:5]
    
    fs = cv2.FileStorage("azure_kinect_intrinsics.yml", cv2.FILE_STORAGE_WRITE)
    fs.write("cameraMatrix", camera_matrix)
    fs.write("distCoeffs", dist_coeffs)
    fs.release()
    print("✅ Camera intrinsics saved to azure_kinect_intrinsics.yml")

    img_count = 0
    brightness = 0
    contrast = 0

    while True:
        capture = k4a.get_capture()
        if capture.color is None or capture.transformed_depth is None:
            continue
        
        frame = capture.color[:, :, :3]
        depth_image = capture.transformed_depth

        display_frame = frame.copy()
        display_frame = adjust_brightness_contrast(display_frame, brightness, contrast)

        cv2.putText(display_frame, "Press [SPACE] to capture, [Q] to quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Brightness: {brightness}  Contrast: {contrast}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Live Feed", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('w'):
            brightness += 5
            print(f"⬆️ Brightness: {brightness}")
        elif key == ord('s'):
            brightness -= 5
            print(f"⬇️ Brightness: {brightness}")
        elif key == ord('e'):
            contrast += 5
            print(f"➡️ Contrast: {contrast}")
        elif key == ord('d'):
            contrast -= 5
            print(f"⬅️ Contrast: {contrast}")
        elif key == ord(' '):
            print("\nAttempting to capture...")

            adjusted_for_detection = adjust_brightness_contrast(frame, brightness, contrast)
            gray = cv2.cvtColor(adjusted_for_detection, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = detector.detectMarkers(gray)
            print(f"Detected marker IDs: {ids}")

            if ids is not None and len(ids) > 0:
                for i, marker_id in enumerate(ids):
                    marker_id = marker_id[0]
                    print(f"Marker ID: {marker_id}")

                    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                        [corners[i]], MARKER_LENGTH, camera_matrix, dist_coeffs
                    )
                    rvec = rvecs[0]
                    tvec = tvecs[0]

                    print(f"rvec: {rvec}")
                    print(f"tvec: {tvec}")

                    cv2.aruco.drawDetectedMarkers(adjusted_for_detection, corners, ids)
                    cv2.drawFrameAxes(adjusted_for_detection, camera_matrix, dist_coeffs, rvec, tvec, 0.10)

                cv2.putText(adjusted_for_detection, "Press [S] to save, [D] to discard", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow("Confirm Capture", adjusted_for_detection)

                while True:
                    confirm_key = cv2.waitKey(0) & 0xFF
                    if confirm_key == ord('s'):
                        print(f"✅ User approved. Saving image {img_count}.")

                        color_path = os.path.join(color_dir, f"{img_count:04d}_color.png")
                        depth_path = os.path.join(depth_dir, f"{img_count:04d}_depth.png")

                        cv2.imwrite(color_path, frame)
                        cv2.imwrite(depth_path, depth_image)

                        img_count += 1
                        break
                    elif confirm_key == ord('d'):
                        print("❌ User discarded capture.")
                        break
                cv2.destroyWindow("Confirm Capture")

            else:
                print("❌ Failure: Marker not detected or pose invalid. Press any key to continue.")
                cv2.imshow("Capture Failed", adjusted_for_detection)
                cv2.waitKey(0)
                cv2.destroyWindow("Capture Failed")

    k4a.stop()
    cv2.destroyAllWindows()
    print("\nProgram finished.")

if __name__ == "__main__":
    main()
