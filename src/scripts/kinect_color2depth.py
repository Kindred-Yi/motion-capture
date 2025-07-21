import numpy as np
import cv2
import sys
from pyk4a import Config, PyK4A, CalibrationType, K4AException

def save_transformation_matrix_to_yml(matrix, filename="kinect_depth_to_color.yml"):
    """
    Saves a 4x4 NumPy transformation matrix to a YAML file using OpenCV.

    Args:
        matrix (np.ndarray): The 4x4 transformation matrix to save.
        filename (str): The name of the output YAML file.
    """
    try:
        # Open the file storage for writing
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        if not fs.isOpened():
            raise IOError(f"Failed to open {filename} for writing.")

        # Write the matrix to the file with a descriptive key
        fs.write("T_color_depth", matrix)
        
        # Release the file
        fs.release()
        
        print(f"✅ Successfully saved transformation matrix to '{filename}'")

    except Exception as e:
        print(f"❌ Error: Could not save the matrix to '{filename}'. Reason: {e}")
        sys.exit(1)

def get_and_save_depth_to_color_transformation():
    """
    Connects to an Azure Kinect device, retrieves the extrinsic transformation
    from the depth camera's coordinate system to the color camera's, converts
    it to meters, and saves it to a YAML file.
    """
    try:
        # Initialize the camera with default settings.
        # This will automatically find the first connected device.
        k4a = PyK4A(Config())
        k4a.open()
        print("✅ Azure Kinect device connected successfully.")

    except K4AException as e:
        print(f"❌ Error: Failed to connect to Azure Kinect device.")
        print(f"   Please ensure the camera is connected and drivers are installed.")
        print(f"   SDK Error: {e}")
        sys.exit(1)
        
    try:
        # --- CORRECTED METHOD ---
        # Access the extrinsic calibration data directly.
        # This provides the rotation (R) and translation (t) to transform points
        # from the DEPTH camera frame to the COLOR camera frame.
        extrinsics = k4a.calibration.extrinsics[CalibrationType.DEPTH][CalibrationType.COLOR]
        R = extrinsics['r']  # 3x3 Rotation matrix
        t = extrinsics['t']  # 3x1 Translation vector (in millimeters)

        # Assemble the 4x4 transformation matrix from the rotation and translation
        T_color_depth_mm = np.eye(4, dtype=np.float32)
        T_color_depth_mm[:3, :3] = R
        T_color_depth_mm[:3, 3] = t

        # The translation vector 't' is in millimeters.
        # We must convert it to meters for consistency with other tools like Open3D.
        T_color_depth_meters = np.copy(T_color_depth_mm)
        T_color_depth_meters[0:3, 3] /= 1000.0

        print("\nTransformation Matrix (T_color_depth) in meters:")
        print(T_color_depth_meters)
        
        # Save the final matrix to a YAML file
        save_transformation_matrix_to_yml(T_color_depth_meters)

    except Exception as e:
        print(f"❌ Error: An unexpected error occurred while fetching calibration data: {e}")
    
    finally:
        # Always ensure the camera connection is closed
        k4a.close()
        print("\nCamera connection closed.")


if __name__ == "__main__":
    get_and_save_depth_to_color_transformation()
