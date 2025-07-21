# Motion Capture ReadMe V2
First we want to first get calibration data for the hand eye calibration. Please not that the Command Line arugments needs to be changed, these are just examples

1) First we set up the environment
```
source .venv/Scripts/activate # Any venv will work, just need to set up the environment first
```

2) Now We are going to take pictures for the calibration. Look at the window pop up and follow directions to take pictures
```
python src/scripts/take_pictures.py --output src/data/frames_test
```

3) For each color to depth pair, run the calib_with_image.py below, changing the name each time. It is important to make sure to save in the correct directory name as that is how EyehandCalibration.py finds it

```
python src/scripts/calib_with_images.py --rgb src/data/frames_test/color/0000_color.png -f src/data/frames_test/calibration_results/ -n extrinsics_0.yml
```

Use the point cloud visualization to see if good or not. Or else, play around with MARKER_SEPARATION and MARKER_LENGTH until in the point cloud the red dot is at the right place on the aruco tag. A good rule of thumb is MARKER_LENGTH changes how far the red dot is to the kinect camera while MARKER_SEPARATION moves the red dot around on that same distance

```
python src/scripts/trans_to_point_cloud.py --rgb src/data/frames_test/color/0000_color.png --depth src/data/frames_test/depth/0000_depth.png  --extrinsic src/data/frames_test/calibration_results/extrinsics_0.yml
```
4) Before running the below, make sure that you have your calibration results in a folder called "calibration_results" as well as the optitrack data in a folder called "optitrack" be in the --inputs directory in the next command. Often, these folders are just put in the same folder as the color and depth folders


```
python src/scripts/EyehandCalibration.py --inputs src/data/frames_test/
```

5) Visualize the data for 1 frame through combine_point_cloud_optitrack_1_frame.py. Use -h if unsure of the command line arguments
```
python src/scripts/combine_point_cloud_optitrack_1_frame.py --rgb src/data/frames_test/combination/color/00000.jpg --depth src/data/frames_test/combination/depth/00000.png -crc src/data/frames_calibration2/hand_eye_calibration_result.yml -o src/data/frames_test/test_recording.csv -c Kinect_cam
```

6) Run the kinect_optitrack_syncer.py program. This has no command line arguments. Instead, you have to change the global variables in this
```
python src/scripts/kinect_optitrack_syncer.py
```


7) Code not finished yet due to need for testing in Calibration. However, general code is already there. All that is left to do is to make sure every kinect frame has 4 optitrack frames as well as read the json file from kinect_optitrack_syncer
```
python src/scripts/combine_point_cloud_optitrack_sequence.py -c "D:\HAND_Human_Human_Study\Kinect\mkv_2_colordepth\7-15 4pm 1\color" -d "D:\HAND_Human_Human_Study\Kinect\mkv_2_colordepth\7-15 4pm 1\depth" -o "D:\HAND_Human_Human_Study\OptiTrack\peanut butter 7-15 4 1.csv" -crc src/data/frames_calibration2/hand_eye_calibration_result.yml --cam_body_name Kinect_cam --intrinsic azure_kinect_intrinsics.yml
```



# below is old readme


V2
```
source .venv/Scripts/activate
```


```
python src/scripts/take_pictures.py --output src/data/frames_test
```

For each color to depth pair, run the calib_with_image.py below, changing the name each time. It is important to make sure to save in the correct directory name as that is how EyehandCalibration.py finds it

```
python src/scripts/calib_with_images.py --rgb src/data/frames_test/color/0000_color.png -f src/data/frames_test/calibration_results/ -n extrinsics_0.yml
```

Use the point cloud visualization to see if good or not. Or else, play around with  MARKER_SEPARATION and MARKER_LENGTH until in the point cloud the red dot is at the right place on the aruco tag

```
python src/scripts/trans_to_point_cloud.py --rgb src/data/frames_test/color/0000_color.png --depth src/data/frames_test/depth/0000_depth.png  --extrinsic src/data/frames_test/calibration_results/extrinsics_0.yml
```

Before running the below, make sure that you have your calibration results in a folder called "calibration_results" as well as the optitrack data in a folder called "optitrack" be in the --inputs directory in the next command


```
python src/scripts/EyehandCalibration.py --inputs src/data/frames_test/
```


```
python src/scripts/combine_point_cloud_optitrack_1_frame.py --rgb src/data/frames_test/combination/color/00000.jpg --depth src/data/frames_test/combination/depth/00000.png -crc src/data/frames_calibration2/hand_eye_calibration_result.yml -o src/data/frames_test/test_recording.csv -c Kinect_cam
```

```
python src/scripts/combine_point_cloud_optitrack_sequence.py -c "D:\HAND_Human_Human_Study\Kinect\mkv_2_colordepth\7-15 4pm 1\color" -d "D:\HAND_Human_Human_Study\Kinect\mkv_2_colordepth\7-15 4pm 1\depth" -o "D:\HAND_Human_Human_Study\OptiTrack\peanut butter 7-15 4 1.csv" -crc src/data/frames_calibration2/hand_eye_calibration_result.yml --cam_body_name Kinect_cam --intrinsic azure_kinect_intrinsics.yml
```

V1

``
Jeffrey command prompt stuff

cd "Program Files\Azure Kinect SDK v1.4.1\tools"
``

``
"k4arecorder.exe"  -c 720p -r 15 -l 5 "%USERPROFILE%\OneDrive\Documents\Peoples and Robots Laboratory Research\motion-capture\output.mkv" # this is what I run in Poweshell to get recording
``

VSCODE Part now

``
python3 ./Open3D/examples/python/reconstruction_system/sensors/azure_kinect_mkv_reader.py --input ./output.mkv --output ./Open3D/frames # this is for running the program in my root directory
``

a)
``
python src/scripts/calib_with_images.py --rgb "C:\Users\jeffr\OneDrive\Documents\Peoples and Robots Laboratory Research\motion-capture\Open3D\frames\color\00000.jpg" --depth "C:\Users\jeffr\OneDrive\Documents\Peoples and Robots Laboratory Research\motion-capture\Open3D\frames\depth\00000.png"
``

b)
``
python src/scripts/calib_with_batch_images.py --folder Open3D/frames
``

``
python src/scripts/trans_to_point_cloud.py --rgb "C:\Users\jeffr\OneDrive\Documents\Peoples and Robots Laboratory Research\motion-capture\Open3D\frames\color\00000.jpg" --depth "C:\Users\jeffr\OneDrive\Documents\Peoples and Robots Laboratory Research\motion-capture\Open3D\frames\depth\00000.png" --extrinsic src/data/frames_calibration2/hand_eye_calibration_result.yml
``


# motion-capture

Set up python virtual environment:
```
python3.12 -m venv ~/venv/open3d
source ~/venv/open3d/bin/activate
```

Building Azure-Kinect-Sensor-SDK: Follow [here](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/blob/develop/docs/building.md).
 It usually works for ubuntu 20.04, for higher versions, refer to [here](https://github.com/microsoft/Azure-Kinect-Sensor-SDK/issues/1790)

Building Open3d library: Follow [here](https://www.open3d.org/docs/release/compilation.html#compilation)

Note: To build Open3D from source with K4A support, set BUILD_AZURE_KINECT=ON at CMake config step. That is,
```
cmake -DBUILD_AZURE_KINECT=ON -DOTHER_FLAGS ..
```

Recode:
```
cd /workspace/src
k4arecorder -c 720p -r 15 -l 5 output.mkv
```
Transform mkv file to rgbd images:
```
python /workspace/src/Open3D/examples/python/reconstruction_system/sensors/azure_kinect_mkv_reader.py --input /workspace/src/output.mkv --output /workspace/src/Open3D/frames
```
Generate point cloud:
```
python trans_to_point_cloud.py --rgb /workspace/src/Open3D/frames/color/00000.jpg --depth /workspace/src/Open3D/frames/depth/00000.png --output /workspace/src/Open3D/data/output00000.pcd
```
