"""
Testing file for development, to experiments for iiwa.
"""
import math
import logging
import hydra

from gym_grasping.envs.iiwa_env import IIWAEnv
from flow_control.servoing.module import ServoingModule
from flow_control.flow_control_main import evaluate_control


def main_hw(start_paused=False):
    """
    The main function that loads the recording, then runs policy for the ur3.
    """

    hydra.initialize("../conf/robot/")
    cfg = hydra.compose("ur_interface.yaml")
    robot = hydra.utils.instantiate(cfg)

    logging.basicConfig(level=logging.INFO, format="")

    recording, episode_num = "/media/argusm/Seagate Expansion Drive/kuka_recordings/flow/vacuum", 5
    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/multi2", 1
    # recording, episode_num = "/media/kuka/sergio-ntfs/multi2/", 1

    control_config = dict(mode="pointcloud-abs", threshold=0.35)

    servo_module = ServoingModule(recording, control_config=control_config, start_paused=False, plot=True,
                                  save_dir=None)

    iiwa_env = IIWAEnv(act_type='continuous', freq=20,
                       obs_type='image_state_reduced',
                       img_flip_horizontal=True,
                       dv=0.0035, drot=0.025, use_impedance=True, max_steps=1e9,
                       reset_pose=(0, -0.56, 0.23, math.pi, 0, math.pi / 2), control='relative',
                       gripper_opening_width=109,
                       obs_dict=False)
    iiwa_env.reset()

    _, reward, _, _ = evaluate_control(iiwa_env, servo_module)
    print("reward:", reward, "\n")


if __name__ == "__main__":
    main_hw(start_paused=True)
