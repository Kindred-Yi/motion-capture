#!/usr/bin/env python3

import open3d as o3d
import sys

if len(sys.argv) < 2:
    print("Usage: python view_pcd.py your_file.pcd")
    sys.exit(1)

pcd = o3d.io.read_point_cloud(sys.argv[1])
pcd = pcd.voxel_down_sample(voxel_size=0.005)
coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
o3d.visualization.draw_geometries([pcd, coordinate_frame])
