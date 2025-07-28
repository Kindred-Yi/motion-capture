import numpy as np
import pybullet as p
import pybullet_data
import time
from lively import (
    Solver, Translation, ScalarRange,
    SmoothnessMacroObjective, CollisionAvoidanceObjective,
    PositionMatchObjective, OrientationMatchObjective,
    State, Transform
)
from lively import Rotation as livelyRotation
from lxml import etree
from scipy.spatial.transform import Rotation

# Load fingertip and wrist data
eef_data = np.load("eef_629.npz")
wrist_pose = eef_data["right_wrist_tfs"]
tip_poses_world = eef_data["right_tip_poses"]

# Start PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
robot = p.loadURDF(
    "assets/franka_description/robots/frankaEmikaPanda.urdf",
    basePosition=[0.0, 0.0, 0.0],
    baseOrientation=[0, 0, 0, 1],
    useFixedBase=True
)

def create_primitive_shape(pb, mass, shape, dim, color=(0.6, 0, 0, 1), 
                           collidable=True, init_xyz=(0, 0, 0),
                           init_quat=(0, 0, 0, 1)):
    # shape: p.GEOM_SPHERE or p.GEOM_BOX or p.GEOM_CYLINDER
    # dim: halfExtents (vec3) for box, (radius, length)vec2 for cylinder, (radius) for sphere
    # init_xyz vec3 being initial obj location, init_quat being initial obj orientation
    visual_shape_id = None
    collision_shape_id = -1
    if shape == p.GEOM_BOX:
        visual_shape_id = pb.createVisualShape(shapeType=shape, halfExtents=dim, rgbaColor=color)
        if collidable:
            collision_shape_id = pb.createCollisionShape(shapeType=shape, halfExtents=dim)
    elif shape == p.GEOM_CYLINDER:
        visual_shape_id = pb.createVisualShape(shape, dim[0], [1, 1, 1], dim[1], rgbaColor=color)
        if collidable:
            collision_shape_id = pb.createCollisionShape(shape, dim[0], [1, 1, 1], dim[1])
    elif shape == p.GEOM_SPHERE:
        visual_shape_id = pb.createVisualShape(shape, radius=dim[0], rgbaColor=color)
        if collidable:
            collision_shape_id = pb.createCollisionShape(shape, radius=dim[0])

    sid = pb.createMultiBody(baseMass=mass, baseInertialFramePosition=[0, 0, 0],
                             baseCollisionShapeIndex=collision_shape_id,
                             baseVisualShapeIndex=visual_shape_id,
                             basePosition=init_xyz, baseOrientation=init_quat)
    return sid

# Load combined URDF for solver
xml_string = etree.tostring(etree.parse('./panda_tesollo.xml')).decode()

# Define Lively objectives
objectives = {
    "position1": PositionMatchObjective(name="F1", link="F1_TIP", weight=15.0),
    "position2": PositionMatchObjective(name="F2", link="F2_TIP", weight=15.0),
    "position3": PositionMatchObjective(name="F3", link="F3_TIP", weight=15.0),
    "wrist_pos": PositionMatchObjective(name="WristPos", link="delto_base_link", weight=5.0),
    "wrist_ori": OrientationMatchObjective(name="WristOri", link="delto_base_link", weight=15.0)
}

# Create solver
solver = Solver(
    urdf=xml_string,
    objectives=objectives,
    root_bounds=[ScalarRange(0.0, 0.0)] * 6
)

# Get controllable joint indices
controllable_joints = [
    i for i in range(p.getNumJoints(robot))
    if p.getJointInfo(robot, i)[2] != p.JOINT_FIXED
]

# Desired joint order
desired_order = [
    "panda_joint1", "panda_joint2", "panda_joint3",
    "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7",
    "F1M1", "F1M2", "F1M3", "F1M4",
    "F2M1", "F2M2", "F2M3", "F2M4",
    "F3M1", "F3M2", "F3M3", "F3M4"
]

def apply_offset(position, quat):
    local_offset = np.array([0.0, 0.02, 0.03])
    # apply offset to world coordinates
    rot = Rotation.from_quat(quat)
    world_offset = rot.apply(local_offset)

    # calculate new position in world coordinates
    new_position = position + world_offset

    rot_y = Rotation.from_euler('y', np.radians(30), degrees=False)    # rotate 30° around world y-axis
    rot_x = Rotation.from_euler('x', np.radians(-30), degrees=False)   # rotate -30° around world x-axis
    rot_original = Rotation.from_quat(quat)

    rot_new = rot_original * rot_y * rot_x  

    new_quat = rot_new.as_quat()  # [x, y, z, w]

    return new_position, new_quat

scales = np.array([0.9, 0.8, 0.7])

c_code = [[1,0,0,1], [0,1,0,1], [0,0,1,1], [1,1,0,1]]
vis_sp = []
for i in range(3):
    vis_sp.append(create_primitive_shape(p, 0.1, p.GEOM_SPHERE, [0.02], color=c_code[i]))

vis_arm = create_primitive_shape(p, 0.1, p.GEOM_SPHERE, [0.03], color=[1, 0.5, 0, 1]) 

# Solve and visualize for each frame
for i in range(min(len(wrist_pose), len(tip_poses_world))):
    wrist_pos = wrist_pose[i, :3]
    wrist_quat = wrist_pose[i, 3:]

    # wrist_pos, wrist_quat = apply_offset(wrist_pos_raw, wrist_orn_raw)

    tips = tip_poses_world[i]

    p.resetBasePositionAndOrientation(vis_arm, wrist_pos, (0, 0, 0, 1))

    for j in range(3):
        p.resetBasePositionAndOrientation(vis_sp[j], tips[j], (0, 0, 0, 1))

    goals = {
        "position1": Translation(*tips[0]),
        "position2": Translation(*tips[1]),
        "position3": Translation(*tips[2]),
        "wrist_pos": Translation(*wrist_pos),
        "wrist_ori": livelyRotation(*wrist_quat)
    }

    state = solver.solve(goals=goals, weights={}, time=0.0)

    joint_states = [state.joints[k] for k in sorted(state.joints, key=lambda x: desired_order.index(x))]
    for j, joint_val in enumerate(joint_states[:len(controllable_joints)]):
        p.resetJointState(robot, controllable_joints[j], joint_val)


    p.stepSimulation()
    time.sleep(1. / 120.) 

p.disconnect()
