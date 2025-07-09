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

Jeffrey testing commands

k4arecorder -c 720p -r 15 -l 5 output.mkv

python ./Open3D/examples/python/reconstruction_system/sensors/azure_kinect_mkv_reader.py --input ./output.mkv --output ./Open3D/frames

python src/scripts/calib_with_images.py --rgb /home/hci-lab/repos/motion-capture/Open3D/frames/color/00100.jpg --depth /home/hci-lab/repos/motion-capture/Open3D/frames/depth/00100.png