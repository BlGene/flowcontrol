"""
Module for testing UR3.
"""
import logging

import numpy as np
from pathlib import Path

import hydra.utils

from robot_io.recorder.simple_recorder import SimpleRecorder

from flow_control.servoing.module import ServoingModule
from flow_control.servoing.runner import evaluate_control, act2inst
from flow_control.demo_selection.select_hloc import SelectionHloc

log = logging.getLogger(__name__)


def move_to_neutral_desk(robot):
    home_pos = np.array([0.2988764, 0.11044048, 0.15792169])
    home_orn = np.array([9.99999242e-01, -1.23099822e-03, 1.18825773e-05, 3.06933556e-05])
    pos, orn = robot.get_tcp_pos_orn()

    if pos[2] > home_pos[2]:
        # current pos higher than home pos, add minimal delta_z
        delta_z = 0.03
    else:
        delta_z = abs(pos[2] - home_pos[2])

    # delta_z = max(0.0, pos[2] - home_pos[2])
    pos_up = pos + np.array([0.0, 0.0, delta_z])
    hpos = np.array([0.2988764, 0.11044048, pos_up[2]])
    pos_up2 = hpos

    robot.move_cart_pos(pos_up, orn, ref="abs", blocking=True, path="lin")
    robot.move_cart_pos(pos_up2, home_orn, ref="abs", blocking=True, path="lin")
    robot.move_cart_pos(home_pos, home_orn, ref="abs", blocking=True, path="lin")

@hydra.main(config_path="../../../robot_io/conf", config_name="ur3_teleop.yaml")
def main(cfg):
    """
    Try running conditional servoing.
    """
    logging.basicConfig(level=logging.DEBUG, format="")

    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    # return to neutral in a safer way.
    move_to_neutral_desk(robot)

    root_dir = Path("/home/argusm/Desktop/Demonstrations/2023-01-24/")
    selection_hloc = SelectionHloc(root_dir)

    # Run preprocess if not run before
    # selection_hloc.preprocess()

    query_cam = env.camera_manager.gripper_cam
    episode_name, _, res_best = selection_hloc.get_best_demo(query_cam)
    rec_path = root_dir / episode_name

    live_state, _, _, live_info = env.step(None)

    control_config = dict(mode="pointcloud-abs", threshold=0.25)
    servo_module = ServoingModule(rec_path, control_config=control_config, start_paused=True,
                                  plot=True, plot_save_dir='./plots', flow_module='RAFT')

    servo_module.check_calibration(env)  # update servo module with T_tcp_cam
    servo_module.process_obs(live_state, live_info)  # updates info with world_tcp
    live_to_demo_action = servo_module.trf_to_abs_act(res_best['trf_est'], live_info)
    live_to_demo_action_i = act2inst(live_to_demo_action, path="lin", blocking=True)

    # First action to align live and demonstration
    state, _, _, _ = env.step(live_to_demo_action_i)

    simp_rec = SimpleRecorder(env)
    state, _, _, _ = evaluate_control(env, servo_module, recorder=simp_rec, initial_align=False)

    move_to_neutral_desk(robot)


if __name__ == "__main__":
    main()
