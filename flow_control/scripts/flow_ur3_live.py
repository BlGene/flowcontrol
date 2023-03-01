"""
Module for testing UR3.
"""
import json
import logging
from pathlib import Path

import hydra.utils

from robot_io.recorder.simple_recorder import SimpleRecorder
from flow_control.servoing.module import ServoingModule
from flow_control.servoing.runner import evaluate_control
from flow_control.demo_selection.select_reprojection import select_recording_reprojection
from flow_control.utils_real_robot import move_to_neutral_safe


log = logging.getLogger(__name__)


@hydra.main(config_path="/conf", config_name="ur3_teleop.yaml")
def main(cfg):
    """
    Try running conditional servoing.
    """
    logging.basicConfig(level=logging.DEBUG, format="")
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    # return to neutral in a safer way.
    move_to_neutral_safe(robot)

    root_dir = Path("/home/argusm/Desktop/Demonstrations/2023-01-24")
    with open(root_dir / 'parts.json', 'r') as f_obj:
        part_info = json.load(f_obj)
    part_names = {0: 'locate', 1: 'grasp', 2: 'insert'}

    # Check recordings are in the same order
    recordings = sorted([root_dir / rec for rec in root_dir.iterdir() if (root_dir / rec).is_dir()])

    simp_rec = SimpleRecorder(env)
    start_paused = True
    initial_align = True
    control_config = dict(mode="pointcloud-abs", threshold=0.25)
    for part_idx in sorted(part_names.keys()):
        part_name = part_names[part_idx]
        state, _, _, _ = env.step(None)

        rec_idx, rec_kfs = select_recording_reprojection(recordings, state, part_info, part_name)
        print(f"Loading Keypoints: {rec_kfs}")
        if part_idx > 0:
            initial_align = False
            start_paused = False
        servo_module = ServoingModule(recordings[rec_idx], load=rec_kfs, control_config=control_config,
                                      start_paused=start_paused, plot=True, plot_save_dir='./plots', flow_module="RAFT")
        state, _, _, _ = evaluate_control(env, servo_module, initial_align=initial_align, recorder=simp_rec)

    move_to_neutral_safe(robot)


if __name__ == "__main__":
    main()
