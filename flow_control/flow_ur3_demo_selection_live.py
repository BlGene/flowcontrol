"""
Module for testing UR3.
"""
import os
import logging
import numpy as np

import hydra.utils

from robot_io.recorder.simple_recorder import SimpleRecorder

from flow_control.servoing.module import ServoingModule
from flow_control.runner import evaluate_control
from flow_control.servoing.playback_env_servo import PlaybackEnvServo
import json
from sklearn.preprocessing import minmax_scale
import cv2
import time

log = logging.getLogger(__name__)


def goto_home_safe(robot):
    home_pos = np.array([0.2988764, 0.11044048, 0.15792169])
    home_orn = np.array([9.99999242e-01, -1.23099822e-03, 1.18825773e-05, 3.06933556e-05])
    pos, orn = robot.get_tcp_pos_orn()

    print(f"Current orn: {orn}")

    if pos[2] > home_pos[2]:
        # current pos higher than home pos, add minimal delta_z
        delta_z = 0.03
    else:
        delta_z = abs(pos[2] - home_pos[2])

    # delta_z = max(0.0, pos[2] - home_pos[2])

    pos_up = pos + np.array([0.0, 0.0, delta_z])

    hpos = np.array([0.2988764, 0.11044048, pos_up[2]])
    pos_up2 = hpos

    print(f"pos UP: {pos_up}")
    print(f"pos UP 2: {pos_up2}")
    print(f"home Pos: {home_pos}")

    robot.move_cart_pos(pos_up, orn, ref="abs", blocking=True, path="lin")
    robot.move_cart_pos(pos_up2, home_orn, ref="abs", blocking=True, path="lin")
    robot.move_cart_pos(home_pos, home_orn, ref="abs", blocking=True, path="lin")


def move_to_neutral_desk(robot):
    """
    Move the robot to a neutral position, by first moving along the z-axis only, then the rest.
    """
    robot.open_gripper()

    neutral_pos, neutral_orn = (0.30, 0.11, 0.16), (1, 0, 0, 0)  # queried from move_to_neutral()
    cur_pos, _ = robot.get_tcp_pos_orn()

    # return home high first
    # pos_up = np.array([0, 0, cur_pos[2] - neutral_pos[2]])
    # orn_up = np.array([0, 0, 0, 1])
    # robot.move_cart_pos(pos_up, orn_up, ref="rel", path="ptp")

    robot.move_cart_pos(neutral_pos, neutral_orn, ref="abs", path="ptp")


def similarity_from_reprojection(servo_module, live_rgb, demo_rgb, demo_mask, return_images=False):
    # evaluate the similarity via flow reprojection error
    flow = servo_module.flow_module.step(demo_rgb, live_rgb)
    warped = servo_module.flow_module.warp_image(live_rgb / 255.0, flow)
    error = np.linalg.norm((warped - (demo_rgb / 255.0)), axis=2) * demo_mask
    error = error.sum() / demo_mask.sum()
    mean_flow = np.linalg.norm(flow[demo_mask], axis=1).mean()
    if return_images:
        return error, mean_flow, flow, warped
    return error, mean_flow


def normalize_errors(sim_scores, mean_flows):
    sim_l = sim_scores
    mean_flows_l = mean_flows
    w = .5
    print("debug: normalizing", np.max(sim_l), np.max(mean_flows_l))
    sim_scores_norm = np.mean((1 * minmax_scale(sim_l), w * minmax_scale(mean_flows_l)), axis=0) / (1 + w)
    return sim_scores_norm


def select_best_recording(recordings, rec_names, current_rgb, part_info, part):
    control_config = dict(mode="pointcloud-abs", threshold=0.25)
    servo_module = ServoingModule(recordings[0], control_config=control_config, start_paused=False, plot=False)

    errors = np.ones((len(recordings)))
    mean_flows = np.zeros((len(recordings)))
    
    for demo_idx, rec in enumerate(recordings):
        playback = PlaybackEnvServo(rec, load='keep')

        demo_rgb = playback[part_info[rec_names[demo_idx]][part][0]].cam.get_image()[0]
        demo_mask = playback.fg_masks[part_info[rec_names[demo_idx]][part][0]]

        error, mean_flow = similarity_from_reprojection(servo_module, current_rgb,
                                                        demo_rgb, demo_mask, return_images=False)
        errors[demo_idx] = error
        mean_flows[demo_idx] = mean_flow

    errors_norm = normalize_errors(errors, mean_flows)
    best_idx = np.argmin(errors_norm)

    return best_idx


@hydra.main(config_path="/home/argusm/lang/robot_io/conf", config_name="ur3_teleop.yaml")
def main(cfg):
    """
    Try running conditional servoing.
    """
    logging.basicConfig(level=logging.DEBUG, format="")

    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    # return to neutral in a safer way.
    # move_to_neutral_desk(robot)
    goto_home_safe(robot)

    root_dir = "/home/argusm/Desktop/Demonstrations/2023-01-24"

    # Check if these are in the same order
    recordings = sorted([os.path.join(root_dir, rec) for rec in os.listdir(root_dir)
                         if os.path.isdir(os.path.join(root_dir, rec))])
    rec_names = sorted([rec for rec in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, rec))])

    with open(f'{root_dir}/parts.json', 'r') as f_obj:
        part_info = json.load(f_obj)

    time.sleep(.5)

    part_map = {0: 'locate', 1: 'grasp', 2: 'insert'}

    simp_rec = SimpleRecorder(env)

    initial_align = True

    for idx in range(3):

        state, _, _, _ = env.step(None)
        current_rgb = state['rgb_gripper']

        best_idx = select_best_recording(recordings, rec_names, current_rgb, part_info, part_map[idx])

        load_kp = part_info[rec_names[best_idx]][part_map[idx]]

        print(f"Loading Keypoints: {load_kp}")

        control_config = dict(mode="pointcloud-abs", threshold=0.25)
        servo_module = ServoingModule(recordings[best_idx], control_config=control_config, start_paused=True,
                                      plot=True, plot_save_dir='./plots', load=load_kp)

        if idx > 0:
            initial_align = False

        state, _, _, _ = evaluate_control(env, servo_module, initial_align=initial_align, recorder=simp_rec)

    goto_home_safe(robot)


if __name__ == "__main__":
    main()
