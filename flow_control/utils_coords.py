"""
Some util functions for dealing with transformations.
"""
import os
import numpy as np
from scipy.spatial.transform import Rotation as R

#
# Duplicates of robot_io.utils.utils, clean up?
#

def to_canonical_action(action):
    if isinstance(action, (list,tuple)):
        if len(action) == 5:
            raise ValueError("5-DoF action given")

        if len(action) == 8:
            new_action = dict(motion=(action[0:3], action[3:7], action[7]), ref="abs")
            return new_action

    elif isinstance(action, dict):
        assert "motion" in action and "ref" in action
        assert len(action["motion"][0]) == 3
        assert len(action["motion"][1]) == 4
        assert isinstance(action["motion"][2], (int, float))
        return action
    else:
        raise ValueError("Unrecognized action")

def action_to_current_state(env, grip_action="query"):
    """
    Get the action that will take us to the pose of current robot
    Args:
        robot: robot class
    Returns:
        action: {motion, ref}
    """
    tcp_pos, tcp_orn = env.robot.get_tcp_pos_orn()
    if grip_action == "query":
        # grp_act = env.get_action("gripper")
        raise NotImplementedError
    else:
        grp_act = grip_action
    action = dict(motion=(tcp_pos, tcp_orn, grip_action), ref="abs")
    return action

def pos_orn_to_matrix(pos, orn):
    """
    Arguments:
        position (x,y,z)
        orientation (q_x, q_y, q_z, w)
    Returns:
        mat: 4x4 homogeneous transformation
    """
    assert len(pos) == 3
    assert len(orn) == 4
    mat = np.eye(4)
    mat[:3, 3] = pos
    mat[:3, :3] = R.from_quat(orn).as_matrix()
    return mat


def matrix_to_pos_orn(mat):
    """
    Arguments:
        mat: 4x4 homogeneous transformation
    Returns:
        tuple:
            position: (x, y, z)
            orientation: quaternion (q_x, q_y, q_z, w)
    """
    pos = mat[:3, 3]
    orn = R.from_matrix(mat[:3, :3]).as_quat()
    return pos, orn


def rec_pprint(obj):
    str = ""
    if isinstance(obj, (tuple, list, np.ndarray)):
        str += "[" + ", ".join([rec_pprint(x) for x in obj]) + "]"
    elif isinstance(obj, float):
        return f"{obj:0.4f}"
    elif isinstance(obj, int):
        return f"{obj:06X}"
    else:
        raise ValueError(f"type {type(obj)} not recognied")
    return str


def get_pos_orn_diff(trf_a, trf_b):
    """
    Get the difference in poses.
    Argumetns:
        trf_a: 4x4 matrix
        trf_b: 4x4 matrix

    Returns:
        diff_pos: scalar float (l2 norm)
        diff_rot: scalar float (magnitude between orientations)
    """
    diff_pos = np.linalg.norm(trf_a[:3, 3] - trf_b[:3, 3], 2)
    diff_rot = R.from_matrix(trf_a[:3, :3] @ np.linalg.inv(trf_b[:3, :3])).magnitude()
    return diff_pos, diff_rot


def get_action_dist(env, servo_action):

    if isinstance(servo_action, dict):
        assert servo_action["ref"] == "abs"
        goal_pos = servo_action["motion"][0]
        goal_orn = servo_action["motion"][1]
    else:
        goal_pos = servo_action[0:3]
        goal_orn = servo_action[3:7]

    cur_pos, cur_orn = env.robot.get_tcp_pos_orn()
    l_2 = np.linalg.norm(np.array(goal_pos) - cur_pos)
    orn_diff = (R.from_quat(goal_orn)*R.from_quat(cur_orn).inv()).magnitude()
    # this is mixing of units, [m] and magnitude
    return l_2 + orn_diff


def print_pose_diff(trf_a, trf_b):
    diff_pos = np.linalg.norm(trf_a[:3, 3] - trf_b[:3, 3], 2)
    print(rec_pprint(diff_pos))


def permute_pose_grid(tcp_base, tcp_orn):
    """get target poses by permuting current pose along axes."""
    delta = 0.04
    for i in (0, 1, 2):
        for j in (1, -1):
            target_pose = list(tcp_base)
            target_pose[i] += j * delta
            yield target_pose, tcp_orn


def get_unittest_renderer():
    """
    Get a suitable renderer dependent on computer environment.
    e.g. for unit-testing.
    """
    if "CI" in os.environ:
        return "tiny"
    return "debug"
