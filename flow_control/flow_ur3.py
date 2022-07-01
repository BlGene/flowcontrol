"""
Module for testing UR3.
"""
import os
import logging

import hydra.utils

from flow_control.servoing.module import ServoingModule
from flow_control.flow_control_main import evaluate_control


@hydra.main(config_path="/home/argusm/lang/robot_io/robot_io/conf", config_name="ur3_teleop.yaml")
def main(cfg):
    """
    Try running conditional servoing.
    """
    logging.basicConfig(level=logging.DEBUG, format="")

    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)

    # return to neutral in a safer way.
    robot.move_to_neutral_desk()

    #task = '/home/argusm/lmb/robot_recordings/flow/sick_wtt/16-41-43'
    task = '/home/argusm/lmb/robot_recordings/flow/sick_wtt/16-51-30'

    control_config = dict(mode="pointcloud-abs", threshold=0.25)
    servo_module = ServoingModule(task, control_config=control_config, start_paused=True,
                                  plot=True, save_dir=f'{task}/plots')

    state, _, _, _ = evaluate_control(env, servo_module)

    print("Saved run log to:", os.getcwd())
    # TODO(max): debug in simulation
    # if servo_module.view_plots and servo_module.view_plots.save_dir:
    #    print("Saved plots to:", servo_module.view_plots.save_dir)


if __name__ == "__main__":
    main()

