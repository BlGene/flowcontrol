"""
Test servoing by running it in simulation.

This means we record an base image, move to a target image and servo back to
the base image.
"""
import logging
import unittest

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.demo.playback_env_servo import PlaybackEnvServo
from flow_control.servoing.module import ServoingModule
from flow_control.tests.test_estimate import get_pos_orn_diff
from flow_control.utils_coords import rec_pprint, permute_pose_grid
from flow_control.utils_coords import get_unittest_renderer


class MoveThenServo(unittest.TestCase):
    """
    Test flow based servoing on a calibration patter.

    plot=True causes the live_plot viewer to not close in time for next test
    """
    def run_servo(self, mode):
        """test performance of scripted policy, with parallel gripper"""
        renderer = get_unittest_renderer()
        env = RobotSimEnv(task="flow_calib", robot="kuka",
                          obs_type="image_state", renderer=renderer,
                          act_type='continuous', control="relative",
                          max_steps=600, initial_pose="close",
                          img_size=(256, 256))

        # record base frame and initialize servo module
        fg_mask = (env.get_obs_info()["seg_mask"] == 2)
        demo_pb = PlaybackEnvServo.freeze(env, fg_mask=fg_mask)

        control_config = dict(mode=mode, threshold=0.40)
        servo_module = ServoingModule(demo_pb, control_config=control_config, plot=True, save_dir=None)
        servo_module.set_env(env)

        tcp_base = env.robot.get_tcp_pose()
        tcp_pos, tcp_orn = env.robot.get_tcp_pos_orn()

        # in this loop tcp base is the demo (goal) position
        # we should try to predict tcp_base using live world_tcp
        for target_pose, target_orn in permute_pose_grid(tcp_pos, tcp_orn):
            action = dict(motion=(target_pose, target_orn, 1), ref="abs")
            _, _, _, info = env.step(action)  # go to pose

            max_steps = 30
            servo_action = None
            servo_control = None  # means default
            for counter in range(max_steps):
                state, _, done, info = env.step(servo_action)
                if done:
                    break

                servo_action, _, servo_info = servo_module.step(state, info)

                diff_pos, _ = get_pos_orn_diff(tcp_base, info["world_tcp"])
                if diff_pos < .0015:  # 1mm
                    break

                print(rec_pprint(tcp_base[:3, 3] - info["world_tcp"][:3, 3]))

            servo_module.reset()

            print("action", action)
            self.assertLess(diff_pos, .011)

    # def test_01_relative(self):
    #    """
    #    yield incremental actions.
    #    """
    #    self.run_servo("pointcloud")

    def test_02_absolute(self):
        """
        yield absolute actions.
        """
        self.run_servo("pointcloud-abs")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="")
    unittest.main()
