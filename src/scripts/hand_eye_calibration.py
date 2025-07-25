# _*_coding:utf-8_*_
import os
import cv2
from math import *
import numpy as np
import transforms3d
 
class Calibration:
    def __init__(self):
        # Intrinsic camera matrix
        self.K = np.array([[912.2337646484375, 0.0, 637.2276000976562],
                           [0.0, 912.12548828125, 377.7027893066406],
                           [0.0, 0.0, 1.0]], dtype=np.float64)
        # Distortion coefficients (assumed zero)
        self.distortion = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        # Chessboard parameters
        self.target_x_number = 10
        self.target_y_number = 7
        self.target_cell_size = 25  # in millimeters
 
    # Compute rotation matrix from Euler angles (XYZ order)
    def angle2rotation(self, x, y, z):
        Rx = np.array([[1, 0, 0], [0, cos(x), -sin(x)], [0, sin(x), cos(x)]])
        Ry = np.array([[cos(y), 0, sin(y)], [0, 1, 0], [-sin(y), 0, cos(y)]])
        Rz = np.array([[cos(z), -sin(z), 0], [sin(z), cos(z), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        return R
 
    # Compute transformation matrix from tool pose (in degrees and mm)
    def end_base(self, tx, ty, tz, rx, ry, rz):
        # Convert to radians if needed
        thetaX = rx / 180 * pi
        thetaY = ry / 180 * pi
        thetaZ = rz / 180 * pi
 
        # Compute rotation and translation matrix from end-effector to base
        R_end2base = self.angle2rotation(rx, ry, rz)
        T_end2base = np.array([[tx], [ty], [tz]])
        Matrix_end2base = np.column_stack([R_end2base, T_end2base])
        Matrix_end2base = np.row_stack((Matrix_end2base, np.array([0, 0, 0, 1])))
 
        # Extract rotation and translation matrices
        R_end2base = Matrix_end2base[:3, :3]
        T_end2base = Matrix_end2base[:3, 3].reshape((3, 1))
 
        # Invert to get base to end-effector transform
        Matrix_end2base_inv = np.linalg.inv(Matrix_end2base)
        R_base2end = Matrix_end2base_inv[:3, :3]
        T_base2end = Matrix_end2base_inv[:3, 3].reshape((3, 1))
 
        return R_end2base, T_end2base, R_base2end, T_base2end
 
    # Compute base-to-end transform from pose vector
    def pose_vectors_to_base2end_transforms(self, pose_vector):
        R_end2base = self.euler_to_rotation_matrix(pose_vector[3], pose_vector[4], pose_vector[5], unit='rad')
        t_end2base = pose_vector[:3]
 
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = R_end2base
        pose_matrix[:3, 3] = t_end2base
 
        pose_matrix_inv = np.linalg.inv(pose_matrix)
        R_base2end = pose_matrix_inv[:3, :3]
        t_base2end = pose_matrix_inv[:3, 3].reshape(3, 1)
 
        return R_base2end, t_base2end
 
    # Convert Euler angles to rotation matrix using transforms3d
    def euler_to_rotation_matrix(self, rx, ry, rz, unit='deg'):
        if unit == 'deg':
            rx = np.radians(rx)
            ry = np.radians(ry)
            rz = np.radians(rz)
 
        Rx = transforms3d.axangles.axangle2mat([1, 0, 0], rx)
        Ry = transforms3d.axangles.axangle2mat([0, 1, 0], ry)
        Rz = transforms3d.axangles.axangle2mat([0, 0, 1], rz)
 
        rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
        return rotation_matrix
 
    # Main calibration pipeline
    def process(self, cvimgs_file, endtxts_file):
        R_target2camera_list = []
        T_target2camera_list = []
        R_base2end_list = []
        T_base2end_list = []
 
        object_points = []
        image_points = []
 
        with open(endtxts_file, "r") as endtxts:
            end_txts = endtxts.readlines()
 
        for end_txt in end_txts:
            fields = end_txt.strip().split(",")
            x, y, z = float(fields[0]), float(fields[1]), float(fields[2])
            rx, ry, rz = float(fields[3]), float(fields[4]), float(fields[5])
            img_name = fields[-1]
            img_file = f"{cvimgs_file}/{img_name}"
 
            print("x, y, z, rx, ry, rz:", x, y, z, rx, ry, rz)
            print("img_file:", img_file)
 
            img = cv2.imread(img_file)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.target_x_number, self.target_y_number), None)
 
            if ret:
                object_point = np.zeros((self.target_x_number * self.target_y_number, 3), np.float32)
                object_point[:, :2] = np.mgrid[0:self.target_x_number, 0:self.target_y_number].T.reshape(-1, 2)
                object_point *= self.target_cell_size
 
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
 
                object_points.append(object_point)
                image_points.append(corners_refined)
 
                retval, rvec, tvec = cv2.solvePnP(
                    object_point,
                    corners_refined,
                    self.K,
                    distCoeffs=self.distortion[:4])
 
                print("rvec, tvec:", rvec, tvec)
 
                rotation_mat, _ = cv2.Rodrigues(rvec)
                Matrix_target2camera = np.column_stack((rotation_mat, tvec))
                Matrix_target2camera = np.row_stack((Matrix_target2camera, np.array([0, 0, 0, 1])))
 
                R_target2camera = Matrix_target2camera[:3, :3]
                T_target2camera = Matrix_target2camera[:3, 3].reshape((3, 1))
                print("R_target2camera, T_target2camera :", R_target2camera, T_target2camera)
 
                R_target2camera_list.append(R_target2camera)
                T_target2camera_list.append(T_target2camera)
 
                cv2.drawChessboardCorners(img, (self.target_x_number, self.target_y_number), corners_refined, ret)
                cv2.imshow(f'img_name', img)
                cv2.waitKey(1000)
 
                R_end2base, T_end2base, R_base2end, T_base2end = self.end_base(x, y, z, rx, ry, rz)
                print("R_end2base, T_end2base:", R_end2base, T_end2base)
                print("R_base2end, T_base2end:", R_base2end, T_base2end)
 
                R_base2end_list.append(R_base2end)
                T_base2end_list.append(T_base2end)
 
        print("R_base2end_list, T_base2end_list", R_base2end_list, T_base2end_list, len(R_base2end_list))
        print("R_target2camera_list, T_target2camera_list:", R_target2camera_list, T_target2camera_list, len(R_target2camera_list))
 
        # Compute camera-to-base transformation using hand-eye calibration
        R_camera2base, T_camera2base = cv2.calibrateHandEye(R_base2end_list, T_base2end_list,
                                                            R_target2camera_list, T_target2camera_list)
 
        print("R_camera2base, T_camera2base:", R_camera2base, T_camera2base)
 
        return R_camera2base, T_camera2base, R_base2end_list, T_base2end_list, R_target2camera_list, T_target2camera_list
 
    # Verifies the calibration result by composing transformation matrices
    def check_result(self, R_cb, T_cb, R_gb, T_gb, R_tc, T_tc):
        for i in range(len(R_gb)):
            RT_base2end = np.column_stack((R_gb[i], T_gb[i]))
            RT_base2end = np.row_stack((RT_base2end, np.array([0, 0, 0, 1])))
            print("RT_base2end:", RT_base2end)
 
            RT_camera_to_base = np.column_stack((R_cb, T_cb))
            RT_camera_to_base = np.row_stack((RT_camera_to_base, np.array([0, 0, 0, 1])))
            print("RT_camera_to_base:", RT_camera_to_base)
 
            RT_target_to_camera = np.column_stack((R_tc[i], T_tc[i]))
            RT_target_to_camera = np.row_stack((RT_target_to_camera, np.array([0, 0, 0, 1])))
            RT_camera2target = np.linalg.inv(RT_target_to_camera)
            print("RT_camera2target:", RT_camera2target)
 
            # Compose final transform: chessboard (target) to robot base
            RT_target_to_end = RT_base2end @ RT_camera_to_base @ RT_target_to_camera
            print(f"Verification result {i}:")
            print(RT_target_to_end)
            print('')
 
if __name__ == "__main__":
    cvimgs_file = "cvimgs"
    endtxts_file = "end_txts/end.txt"
    np.set_printoptions(suppress=True)  # Avoid scientific notation for NumPy output
    calibrator = Calibration()
    R_camera2base, T_camera2base, R_base2end_list, T_base2end_list, R_target2camera_list, T_target2camera_list = calibrator.process(cvimgs_file, endtxts_file)
    # Uncomment below to run result verification
    # calibrator.check_result(R_camera2base, T_camera2base, R_base2end_list, T_base2end_list, R_target2camera_list, T_target2camera_list)