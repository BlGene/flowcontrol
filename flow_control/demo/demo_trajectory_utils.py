import os
import copy
import shutil
import logging
from itertools import groupby

import numpy as np
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R

from robot_io.envs.playback_env import PlaybackEnv
from flow_control.utils_coords import pos_orn_to_matrix, matrix_to_pos_orn


from pdb import set_trace

def split_recording(recording):
    """
    Split a recording based on waypoint names.
    """
    rec = PlaybackEnv(recording).to_list()

    rec_files = [rec_el.file for rec_el in rec]

    if "wp_name" in rec[0].data["info"].item():
        wp_names = [rec_el.data["info"].item()["wp_name"] for rec_el in rec]
        print("loaded waypoint names")
    else:
        wp_names = None

    wp_dones = [wp_name.endswith("_done") for wp_name in wp_names]
    for i in range(len(wp_dones)-1):
        if wp_dones[i] is True and wp_dones[i+1] is True:
            wp_dones[i] = False

    split_at = np.where(wp_dones)[0]
    split_at = [-1,] + split_at.tolist() + [len(wp_dones)]
    segments = list(zip(np.array(split_at[:-1])+1, 1+np.array(split_at[1:])))

    for i, seg in enumerate(segments):
        # check
        # print(wp_names[slice(*seg)])

        seg_save_dir = recording + f"_seg{i}"

        if os.path.isdir(seg_save_dir):
            shutil.rmtree(seg_save_dir)

        os.makedirs(seg_save_dir)
        extra_files = ["camera_info.npz", "env_info.json"]
        extra_files = [os.path.join(recording, efn) for efn in extra_files]
        seg_files = rec_files[slice(*seg)]
        for fn in seg_files + extra_files:
            fn_new = fn.replace(recording, seg_save_dir)
            shutil.copy(fn, fn_new)

        print(f"segment {i}: done copying: {len(seg_files)} files")


def get_demo_continous(pos_vec, vel_cont_threshold=.02):
    # vel_cont_threshold is [m/iter] if mean vel above this assume

    # check if the demonstration is conintous video or individual frames
    vel_vec = np.diff(pos_vec, axis=0)
    vel_scl = np.linalg.norm(vel_vec, axis=1)

    if np.mean(vel_scl) > vel_cont_threshold:
        print("Auto-detected: non-continous trajectory")
        return False

    return True


def get_keep_from_wpnames(wp_names):
    if wp_names is None:
        return None

    keep_wpnames = {}
    for i in range(len(wp_names)-1):
        if wp_names[i+1] != wp_names[i]:
            if wp_names[i].endswith("_close"):
                print(f"Skipping keyframe @ {i}: gripper closing {wp_names[i]}")
                # as the gripper is closed, servoing with respect to the object
                # here will not work, so we skip this step.
                continue
            keep_wpnames[i] = dict(name=wp_names[i])

    return keep_wpnames


def get_keep_from_gripper(gripper_actions):
    GRIPPER_OPEN, GRIPPER_CLOSE = 1.0, -1.0  # assume normalized actions

    def check_gripper_closes(idx):
        # check that we transition open -> close & iter segment
        if gripper_actions[idx] == GRIPPER_OPEN and gripper_actions[idx+1] == GRIPPER_CLOSE:
            return True
        return False

    def check_gripper_opens(idx):
        # check that we transition open -> close & iter segment
        if gripper_actions[idx] == GRIPPER_CLOSE and gripper_actions[idx+1] == GRIPPER_OPEN:
            return True
        return False

    gripper_change_steps = np.where(np.diff(gripper_actions))[0].tolist()

    keep_gripper = {}
    for change_i in gripper_change_steps:
        if check_gripper_opens(change_i):
            change_name = "gripper_open"
        elif check_gripper_closes(change_i):
            change_name = "gripper_close"
        else:
            raise ValueError
        keep_gripper[change_i] = dict(name=change_name)

    return keep_gripper


def get_keep_from_motion(pos_vec, vel_stable_threshold=.002):
    # vel_stable_threshold [m/iter] if vel below this assume stable
    vel_vec = np.diff(pos_vec, axis=0)
    vel_scl = np.linalg.norm(vel_vec, axis=1)

    # This first loop gets minimal regions
    active = False
    start, stop = -1, -1
    min_regions = []
    for i in range(len(vel_scl)):
        if vel_scl[i] < vel_stable_threshold:
            if active:
                stop = i
            else:
                active = True
                start, stop = i, i
        else:
            if active:
                min_regions.append((start, stop))
                active = False
                start, stop = -1, -1

    # This second loop gets minimal value
    vel_stable = []
    for start, stop in min_regions:
        try:
            min_idx = start + 1 + np.argmin(vel_scl[start:stop])
        except ValueError:
            min_idx = 0
        if len(vel_stable) == 0 or vel_stable[-1] != min_idx:
            vel_stable.append(min_idx)
    return vel_stable


def check_names_grip(wp_names, gripper_change_steps):
    """
    check to see if the number of grip changes according to waypoint names is
    the same as grip changes according to actions.
    """
    wp_names_grouped = [key for key, _group in groupby(wp_names)]
    wp_names_gripper_changes = [1 if gn.endswith("_close") or gn.endswith("_open") else 0 for gn in wp_names_grouped]
    assert len(gripper_change_steps) == sum(wp_names_gripper_changes)


def filter_by_move_anchors(keep_wpnames, wp_names, filter_rel):
    """
    remove sequential keep frames.
    """
    # using waypoint names keeps frames are frames at the end of trajectory segments
    # if one keep frame is a relative motion, take

    for idx in list(keep_wpnames.keys()):
        if filter_rel[idx] and not filter_rel[idx+1]:
            print(f"Shifting keyframe @ {idx}: {idx} is relative, use {idx+1}")
            del keep_wpnames[idx]
            keep_wpnames[idx+1] = dict(name=wp_names[idx], info="pushed-by-rel")


def get_rel_motion(start_m, finish_m):
    # T such that F = T @ S
    t_rel = finish_m @ np.linalg.inv(start_m)
    return t_rel


def get_dist(rel_m):
    """
    Get the distance of a motion.
    """
    pos, orn = matrix_to_pos_orn(rel_m)
    return np.linalg.norm(pos) + R.from_quat(orn).magnitude()


def filter_by_motions(keep_cmb, tcp_pos, tcp_orn, gripper_actions, threshold=.001):
    keep_keys = list(keep_cmb.keys())
    for idx_a, idx_b in zip(keep_keys[:-1], keep_keys[1:]):
        start_m = pos_orn_to_matrix(tcp_pos[idx_a], tcp_orn[idx_a])
        finish_m = pos_orn_to_matrix(tcp_pos[idx_b], tcp_orn[idx_b])
        rel_m = get_rel_motion(start_m, finish_m)
        score = get_dist(rel_m) + float(gripper_actions[idx_a] != gripper_actions[idx_b])
        if score < threshold:
            print(f"Removing keyframe @ {idx_a}: too close to {idx_b}")
            del keep_cmb[idx_a]
    print()

def filter_by_gripper_motion(gripper_pos, gripper_change_steps, diff_t=.005, min_duration=5):
    raise NotImplementedError
    # check that the gripper velocity is below diff_t for at least min_duration
    gripper_abs_vel = np.abs(np.diff(gripper_pos))
    stable = np.concatenate(([True,], gripper_abs_vel < diff_t))
    grip_stable = []
    grip_ends = []
    for i in range(len(stable)):
        snext = np.all(stable[i:min(i+min_duration, len(stable))])
        grip_stable.append(snext)
        if grip_stable[-2:] == [0, 1]:
            grip_ends.append(i-1)
    # fix edge case, gripper dosen't stop in demo
    max_frame  = len(gripper_pos)-1
    num_frames = len(gripper_pos)
    if len(grip_ends) < len(gripper_change_steps):
        grip_ends.append(max_frame)
    grip_unstable = list(zip(gripper_change_steps, grip_ends))
    grip_stable_arr = np.ones(num_frames)
    for start,stop in grip_unstable:
        grip_stable_arr[start:stop] = 0
    return grip_stable_arr

def is_grip_step(keep_cmb_entry):
    name = keep_cmb_entry["name"]
    if name.startswith("gripper_"):
        return True


def set_skip_at_start(keep_cmb, servo_after_start):
    for i, k in enumerate(keep_cmb):
        if i < servo_after_start:
            keep_cmb[k]["skip"] = False
        else:
            break

def set_skip_from_gripper(keep_cmb, gripper_actions, max_dist=10, t_close=2, t_open=2):
    """
    Decide to servo if we are t_close, or t_open steps away from a grasping action.
    """
    assert gripper_actions.dtype != np.dtype('O')
    assert gripper_actions.ndim == 1
    step_since_grasp = max_dist
    step_since_open = max_dist
    step_since_close = max_dist

    # Iterate backward and save dist to grasp
    for key in reversed(sorted(keep_cmb)):
        name = keep_cmb[key]["name"]
        if name.startswith("gripper_open"):
            step_since_open = 0
            step_since_grasp = 0
        elif name.startswith("gripper_close"):
            step_since_close = 0
            step_since_grasp = 0
        else:
            step_since_grasp = min(step_since_grasp+1, max_dist)
            step_since_close = min(step_since_close+1, max_dist)
            step_since_open = min(step_since_open+1, max_dist)

        keep_cmb[key]["grip_dist"] = step_since_grasp
        if step_since_open  < t_open:
            keep_cmb[key]["skip"] = False
        if step_since_close < t_close:
            keep_cmb[key]["skip"] = False


def get_servo_anchors(move_anchors):
    """
    Move anchors are what we are moving relative to
    Servo anchors are what we want to servo relative to.
    These are nearly always the same, except for edge cases
    e.g. the first actionis abs, but we still want ot servo.

    Returns:
        servo_anchors, array w/ -1-> don't know, -2, rel motion
    """
    servo_anchors = copy.deepcopy(move_anchors)

    # edge case, first actionis abs, but we still want ot servo
    if servo_anchors[0] == "abs":
        if isinstance(servo_anchors[1], int):
            # maybe add a check here
            servo_anchors[0] = servo_anchors[1]
        else:
            logging.warning("Expected to see objct in second frame")

    for i, anchor_old in enumerate(servo_anchors):
        anchor_new = -1
        if isinstance(anchor_old, int):
            anchor_new = anchor_old
        servo_anchors[i] = anchor_new

    return servo_anchors

def set_skip_from_anchors(keep_cmb, move_anchors):
    # This appears not to be needed at the moment.

    servo_anchors = get_servo_anchors(move_anchors)
    for i, k in enumerate(keep_cmb):

        #if servo_anchors[k] == -2:
        #    keep_cmb[k]["skip"] = True

        # by default don't servo in other locations
        if "skip" in keep_cmb[k]:
            continue
        keep_cmb[k]["skip"] = True
    return servo_anchors

def set_trajectory_actions(keep_cmb, segment_steps, tcp_pos, tcp_orn, gripper_actions,
                           abs_waypoints={}):
    """
    This function:
        2. sets abs and pre motions for following the trajectory by dead-reckoning.
           makes sure these are of type float so that they are json serializable.
    """

    prior_key = None
    for key in sorted(keep_cmb):
        pre = []
        if prior_key is None:
            prior_key = key
            # Absolute motion to initial demo position
            pre.append(dict(motion=[tcp_pos[key].tolist(), tcp_orn[key].tolist(),
                            gripper_actions[key]], ref="abs", name=keep_cmb[key]["name"]))
            keep_cmb[key]["pre"] = pre
            continue


        same_segment = segment_steps[key] == segment_steps[prior_key]
        if not same_segment:
            rel_grip = dict(motion=[[0,0,0], [0,0,0,1], gripper_actions[key]], ref="rel",
                            name=keep_cmb[prior_key]["name"])
            pre.append(rel_grip)

        if keep_cmb[key]["name"] in abs_waypoints:
            pre.append(dict(motion=[tcp_pos[key].tolist(), tcp_orn[key].tolist(),
                           gripper_actions[key]], ref="abs",
                           name=keep_cmb[key]["name"]))
        else:
            # relative actions should be the norm
            # after servoing we should only do relative actions
            start_m = pos_orn_to_matrix(tcp_pos[prior_key], tcp_orn[prior_key])
            finish_m = pos_orn_to_matrix(tcp_pos[key], tcp_orn[key])
            rel_m = get_rel_motion(start_m, finish_m)
            rel_pos_orn = [list(x) for x in matrix_to_pos_orn(rel_m)]
            pre.append(dict(motion=[rel_pos_orn[0], rel_pos_orn[1],
                           gripper_actions[key]], ref="rel",
                           name=keep_cmb[key]["name"]))

        keep_cmb[key]["pre"] = pre
        prior_key = key
        # double check that we retain all keep steps
        #assert(np.all([k in keep_cmb.keys() for k in gripper_change_steps]))


def print_keep_frames(keep_cmb):
    print("fr.#  name".ljust(21),"servo   trj-act    grip_dist")
    print("-"*50)
    for k,v in keep_cmb.items():
        print(f"{k}".ljust(5),f"{v['name']}".ljust(15),
              f"{'       ' if v['skip'] else 'servo  '}",
              #f"pre={len(v['pre'])}",
            "->".join([a["ref"] for a in v['pre']]).ljust(10),
              f"{v['grip_dist']}".ljust(10),
              )
    print()




# The following two functions are no used. They were written for the problem of
# PtP motion not being linear enough. The solution for now was to handle this
# via more different waypoint names, there were needed because of the same problem
# The most sensible thing is probably to add a lin mode to the simulation.
def interpolate_trajectory_points(pos_end, orn_end, steps):
    raise NotImplementedError
    """
    Take a large trajectory motion and interpolate it into sub steps.
    Written for relative motions:
    Arguments:
        pos_end: (x, y, z)
        orn_end: (q_x, q_y, q_z, 1)
        steps: number of intermediate steps to generate
    """
    assert len(pos_end) == 3
    assert len(orn_end) == 4

    pos_start = (0, 0, 0)
    orn_start = (0, 0, 0, 1)

    pos_l = np.linspace(pos_start, pos_end, steps+1)[1:]

    slerp = Slerp([0, 1], R.from_quat([orn_start, orn_end]))
    orn_l = slerp(np.linspace(0, 1, steps+1)[1:])
    for i in range(steps):
        print(pos_l[i], get_dist([*pos_l[i], *orn_l[i].as_quat()]))


def interpolate_trajectory(keep_cmb):
    """
    Given a trajectory, add linearly interpolated intermediate steps.
    """
    raise NotImplementedError
    for k in keep_cmb:
        print("index", k)
        if "pre" in keep_cmb[k]:
            pre = keep_cmb[k]["pre"]
            if "rel" in pre:
                rel = pre["rel"]
                dist = get_dist(rel)
                if dist > 0.05:
                    int_steps = round(dist//0.05)+1
                    print(dist, int_steps)
                    interpolate_trajectory_points(rel[0:3], rel[3:7], int_steps)

            else:
                pass
                #print(keep_cmb[k]["pre"])
