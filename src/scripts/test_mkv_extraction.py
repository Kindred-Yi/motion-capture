import cv2
from pyk4a import PyK4APlayback

# --- Path to your ORIGINAL MKV file ---
video_path = r"D:/HAND_Human_Human_Study/Kinect/output_11am_3_710.mkv"

try:
    playback = PyK4APlayback(video_path)
    print(f"Successfully opened original file: {video_path}")
    print("Playing color and depth... Press 'q' to quit.")

    while True:
        try:
            # Get a single capture (color, depth, and IR images)
            capture = playback.get_next_capture()

            # --- You now have access to both color and depth data ---
            # The color image (a NumPy array)
            color_image = capture.color
            # The depth image (a NumPy array with millimeter values)
            depth_image = capture.depth

            if color_image is not None and depth_image is not None:
                # Display the color image
                cv2.imshow("Original Color", color_image)
                
                # To visualize the depth image, we normalize it to be visible
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.05),
                    cv2.COLORMAP_JET
                )
                cv2.imshow("Depth Visualization", depth_colormap)

            # Wait for 1 millisecond, and exit if the user presses 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except EOFError:
            # This error means we've reached the end of the file
            print("End of file.")
            break

    # Clean up
    playback.close()
    cv2.destroyAllWindows()

except Exception as e:
    print(f"Failed to open or read the original MKV file. Error: {e}")