"""
Test functional behaviour through built-in policies.
"""
import os
import logging
import unittest

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.servoing.module import ServoingModule
from flow_control.tests.move_abs_estimate import get_target_poses, make_demo_dict

is_ci = "CI" in os.environ

if is_ci:
    obs_type = "state"
    renderer = "tiny"
else:
    obs_type = "image"
    renderer = "debug"


class MoveThenServo(unittest.TestCase):
    """
    Test a Pick-n-Place task.
    """
    def test_absolute(self):
        """test performance of scripted policy, with parallel gripper"""
        env = RobotSimEnv(task="flow_calib", robot="kuka",
                          obs_type=obs_type, renderer=renderer,
                          act_type='continuous', control="relative",
                          max_steps=600, initial_pose="close",
                          img_size=(256, 256))

        # record base frame
        base_state, _, _, base_info = env.step(None)
        tcp_base = env.robot.get_tcp_pose()
        tcp_angles = env.robot.get_tcp_angles()

        # initialize servo module
        base_action = [*tcp_base[:3, 3], tcp_angles[2], 1]
        demo_dict = make_demo_dict(env, base_state, base_info, base_action)
        control_config = dict(mode="pointcloud", threshold=0.40)

        servo_module = ServoingModule(demo_dict,
                                      control_config=control_config,
                                      plot=True, save_dir=None)

        servo_module.set_env(env)

        for target_pose, control in get_target_poses(env, tcp_base):
            action = [*target_pose, tcp_angles[2], 1]
            state2, _, _, info = env.step(action, control)  # go to pose

            max_steps = 30
            servo_action = None
            servo_control = None  # means default
            for counter in range(max_steps):
                state, reward, done, info = env.step(servo_action, servo_control)
                if done:
                    break

                servo_action, servo_done, servo_queue = servo_module.step(state, info)

                if servo_module.config.mode == "pointcloud-abs":
                    servo_action, servo_control = servo_module.abs_to_action(env, servo_action, info)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="")
    unittest.main()
