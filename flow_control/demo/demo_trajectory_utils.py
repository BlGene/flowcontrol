import numpy as np
from scipy.spatial.transform import Rotation as R


def get_demo_continous(pos_vec, vel_cont_threshold = .02):
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
        else:
            return False

    def check_gripper_opens(idx):
        # check that we transition open -> close & iter segment
        if gripper_actions[idx] == GRIPPER_CLOSE and gripper_actions[idx+1] == GRIPPER_OPEN:
            return True
        else:
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


def get_keep_from_motion(pos_vec, vel_stable_threshold = .002):
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


def filter_by_anchors(keep_wpnames, wp_names, filter_rel):
    # using waypoint names keeps frames are frames at the end of trajectory segments
    # if one keep frame is a relative motion, take
    for idx in keep_wpnames:
        if filter_rel[idx] == True and filter_rel[idx+1] != True:
            print(f"Shifting keyframe @ {idx}: {idx} is relative, use {idx+1}")
            del keep_wpnames[idx]
            keep_wpnames[idx+1] = dict(name=wp_names[idx], info="pushed-by-rel")


def get_rel_motion(start_pos, start_orn, finish_pos, finish_orn):
    # position
    pos_diff = finish_pos - start_pos
    ord_diff = R.from_quat(finish_orn).inv() * R.from_quat(start_orn)
    #assert ord_diff.magnitude() < .35, ord_diff.magnitude() # for now
    return pos_diff.tolist() + ord_diff.as_quat().tolist()


def filter_by_motions(keep_cmb, tcp_pos, tcp_orn, gripper_actions):
    keep_keys = list(keep_cmb.keys())
    for idx_a, idx_b in zip(keep_keys[:-1],keep_keys[1:]):
        rel_m = get_rel_motion(tcp_pos[idx_a], tcp_orn[idx_a], tcp_pos[idx_b], tcp_orn[idx_b])
        score = np.linalg.norm(rel_m[0:3])+ R.from_quat(rel_m[3:8]).magnitude()
        score += gripper_actions[idx_a] != gripper_actions[idx_b]
        if score < .001:
            print(f"Removing keyframe @ {idx_a}: too close to {idx_b}")
            del keep_cmb[idx_a]


def set_grip_dist(keep_cmb, segment_steps, tcp_pos, tcp_orn, gripper_actions, max_dist=10):
    """
    This function:
        1. sets grip_dist, the distance in keyframes till the next grasp operation.
        2. sets abs and pre motions for following the trajectory by dead-reckoning.
    """
    step_since_grasp = max_dist
    # Iterate backward and save dist to grasp
    for key in reversed(sorted(keep_cmb)):
        name = keep_cmb[key]["name"]
        if name.startswith("gripper_"):
            step_since_grasp = 0
        else:
            step_since_grasp = min(step_since_grasp+1, max_dist)
        keep_cmb[key]["grip_dist"] = step_since_grasp

    prior_key = None
    for key in sorted(keep_cmb):
        if prior_key is None:
            prior_key = key
            continue
        pre_dict = {}

        same_segment = segment_steps[key] == segment_steps[prior_key]
        if not same_segment:
            pre_dict["grip"] = gripper_actions[key]

        if keep_cmb[prior_key]["grip_dist"] < 2:
            rel_motion = get_rel_motion(tcp_pos[prior_key], tcp_orn[prior_key],
                                        tcp_pos[key], tcp_orn[key])
            pre_dict["rel"] = rel_motion
        else:
            abs_motion = [*tcp_pos[key], *tcp_orn[key]]
            pre_dict["abs"] = abs_motion

        keep_cmb[key]["pre"] = pre_dict
        prior_key = key

        # double check that we retain all keep steps
        #assert(np.all([k in keep_cmb.keys() for k in gripper_change_steps]))


def set_anchors(keep_cmb, anchors):
    for k in keep_cmb:
        if anchors[k] == "rel":
            anchor = "rel"
        else:
            anchor = "object"
        keep_cmb[k]["anchor"] = anchor
