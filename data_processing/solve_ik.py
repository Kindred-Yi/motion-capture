import os
import time
import datetime
import socket
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as p
import pybullet_data

physicsClient = p.connect(p.GUI)  
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Upload the URDF files for the arm and hand
arm = p.loadURDF("assets/arm/panda_delto.urdf", basePosition=[0.0, 0.0, 0.0], baseOrientation=[0, 0, 0.7071068, 0.7071068], useFixedBase=True)
hand = p.loadURDF("assets/hand/delto_gripper_3f.urdf")

# Default joint positions for the arm and hand
ARM_REST = [0.4,
        -0.49826458111314524,
        -0.01990020486871322,
        -2.4732269941140346,
        -0.01307073642274261,
        2.00396583422025,
        1.1980939705504309]
HAND_REST = [0, 0, np.pi/12, np.pi/12,
            -np.pi/6, 0 , np.pi/12, np.pi/12,
            np.pi/6, 0 , np.pi/12, np.pi/12]

# Get the indices of the fingertip joints
fingertip_idx = [4, 9, 14]

# Function to set joint positions for the arm and hand, skipping fixed joints
def set_joint_positions(robot, joint_positions):
    jid = 0
    for i in range(len(joint_positions)):
        if p.getJointInfo(robot, jid)[2] != p.JOINT_FIXED:
            p.resetJointState(robot, jid, joint_positions[i])
        else:
            jid += 1
            if p.getJointInfo(robot, jid)[2] == p.JOINT_FIXED:
                jid += 1
            p.resetJointState(robot, jid, joint_positions[i])
        jid += 1

def get_current_joint_positions(robot):
    # Should ignore fixed joints
    q = []
    for i in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, i)[2] != p.JOINT_FIXED:
            q.append(p.getJointState(robot, i)[0])
    return q

def get_joint_limits(robot):
    joint_lower_limits = []
    joint_upper_limits = []
    joint_ranges = []
    for i in range(p.getNumJoints(robot)):
        joint_info = p.getJointInfo(robot, i)
        if joint_info[2] == p.JOINT_FIXED:
            continue
        joint_lower_limits.append(joint_info[8])
        joint_upper_limits.append(joint_info[9])
        joint_ranges.append(joint_info[9] - joint_info[8])
    return joint_lower_limits, joint_upper_limits, joint_ranges

def solve_arm_ik(robot, wrist_pos, wrist_orn):
    ik_result = p.calculateInverseKinematics(bodyUniqueId=arm,
                                            endEffectorLinkIndex=9,
                                            targetPosition=wrist_pos,
                                            targetOrientation=wrist_orn,
                                            lowerLimits=arm_lower_limits, 
                                            upperLimits=arm_upper_limits, 
                                            jointRanges = arm_joint_ranges, 
                                            restPoses = ARM_REST, 
                                            # currentPosition = prev_joints,
                                            maxNumIterations=1000, 
                                            residualThreshold=0.001)
    return ik_result

def solve_fingertip_ik(robot, fingertip_pos):
    tip_poses = []
    for i,fid in enumerate(fingertip_idx):
        tip_pos = fingertip_pos[i]
        tip_poses.append(tip_pos)
    target_q = []
    for i in range(3):
        target_q = target_q + list(p.calculateInverseKinematics(bodyUniqueId=hand, 
                                                                endEffectorLinkIndex=fingertip_idx[i], 
                                                                targetPosition=tip_poses[i], 
                                                                lowerLimits=hand_lower_limits, 
                                                                upperLimits=hand_upper_limits, 
                                                                jointRanges = hand_joint_ranges, 
                                                                restPoses=HAND_REST, 
                                                                # currentPosition = curr_joints,
                                                                maxNumIterations=40, 
                                                                residualThreshold=0.001))[4*i:4*(i+1)]
    return target_q

# Initialize the arm and hand
arm_lower_limits, arm_upper_limits, arm_joint_ranges = get_joint_limits(arm)
hand_lower_limits, hand_upper_limits, hand_joint_ranges = get_joint_limits(hand)

# Create a fixed joint constraint between the arm and hand
cid = p.createConstraint(
        parentBodyUniqueId=arm,   parentLinkIndex=9,
        childBodyUniqueId=hand,   childLinkIndex=-1,
        jointType=p.JOINT_FIXED,  jointAxis=[0,0,0],
        parentFramePosition=[0,0,0],          
        childFramePosition=[0,0,0],
        parentFrameOrientation=[0,0,0,1],
        childFrameOrientation=[0,0,0,1])

# Set the initial joint positions for the arm and hand
set_joint_positions(arm, ARM_REST)

#hand_xyz = np.asarray(p.getLinkState(arm, 9)[0])
#hand_orn = Rotation.from_quat(p.getLinkState(arm, 9)[1])


set_joint_positions(hand, HAND_REST)



eef_data = np.load("eef_629.npz")
wrist_pose = eef_data["right_wrist_tfs"]
tip_poses_world = eef_data["right_tip_poses"]

# print(p.getJointInfo(hand, 4))  # Check joint info for the first joint

# hand_xyz = np.asarray(p.getLinkState(arm, 9)[0])
# hand_orn = Rotation.from_quat(p.getLinkState(arm, 9)[1])
# print("Hand position:", hand_xyz)
# print("Hand orientation:", hand_orn)
# print("Hand orientation:", hand_orn.as_euler('xyz', degrees=True))


for i in range(100000):
    wrist_pos = wrist_pose[i, :3] + 0.2
    wrist_orn = Rotation.from_quat(wrist_pose[i, 3:])
    arm_q = solve_arm_ik(arm, wrist_pos, wrist_orn)
    set_joint_positions(arm, arm_q)

    hand_xyz = np.asarray(p.getLinkState(arm, 9)[0])
    hand_orn = Rotation.from_quat(p.getLinkState(arm, 9)[1])

    tip_poses_wrist = wrist_orn.apply(tip_poses_world[i]) + wrist_pos

    hand_q = solve_fingertip_ik(hand, tip_poses_wrist)
    
    set_joint_positions(hand, hand_q)

    p.stepSimulation()
    time.sleep(1. / 240.)

