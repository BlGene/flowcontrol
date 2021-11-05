"""
Test servoing by estimating relative poses from demonstrations.

This means we take to frames from a demo, fro which we also have poses
and use the images to try to estimate the relative poses.
"""
import logging
import unittest
from flow_control.servoing.module import ServoingModule
from flow_control.tests.test_estimate import get_pose_diff


class EstimateDemo(unittest.TestCase):
    """
    Estimate relative poses from demonstration trajectory.
    """
    def test_01_absolute(self):
        """Estimate relative poses from demonstration trajectory."""

        recording, episode_num = "/home/argusm/CLUSTER/robot_recordings/flow/vacuum", 2
        demo_start, demo_stop = 0, 30

        servo_module = ServoingModule(recording, episode_num,
                                      control_config=dict(mode="pointcloud-abs"),
                                      plot=False, save_dir=None)
        servo_module.set_env("demo")

        for demo_frame in range(demo_start, demo_stop, 1):
            state, info = servo_module.demo.get_state(demo_frame)
            servo_action, servo_done, servo_info = servo_module.step(state, info)

            if servo_module.config.mode == "pointcloud-abs":
                tcp_base_est = servo_module.abs_to_world_tcp(servo_info, info)

            tcp_base = servo_module.demo.world_tcp
            diff_pos, diff_rot = get_pose_diff(tcp_base, tcp_base_est)
            print(diff_pos)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="")
    unittest.main()
