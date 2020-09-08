"""
Record random views to test pose estimation system.
"""
import os
import math
import time
import json

import numpy as np
import cv2

from robot_io.kuka_iiwa.iiwa_controller import IIWAController
from robot_io.kuka_iiwa.wsg50_controller import WSG50Controller
from robot_io.cams.realsenseSR300_librs2 import RealsenseSR300
from gym_grasping.calibration.random_pose_sampler import RandomPoseSampler


class RandomViewRecorder(RandomPoseSampler):
    """
    Record random views to test pose estimation system.
    """

    def __init__(self,
                 save_dir="/media/kuka/Seagate Expansion Drive/"
                          "kuka_recordings/flow/pose_estimation/",
                 save_folder="default_recording",
                 num_samples=50):
        super().__init__()
        self.save_path = os.path.join(save_dir, save_folder)
        os.makedirs(self.save_path, exist_ok=True)
        self.num_samples = num_samples
        self.robot = IIWAController(use_impedance=False, joint_vel=0.3,
                                    gripper_rot_vel=0.5, joint_acc=0.3)
        gripper = WSG50Controller()
        gripper.home()
        self.cam = RealsenseSR300(img_type='rgb_depth')

    def create_dataset(self):
        '''the main dataset collection loop'''
        self.robot.send_cartesian_coords_abs_PTP((*self.center, math.pi, 0, math.pi / 2))
        time.sleep(4)

        poses = []
        # start file indexing with 0 and zero pad filenames
        for i in range(self.num_samples):
            pos = self.sample_pose()
            self.robot.send_cartesian_coords_abs_PTP(pos)
            time_0 = time.time()
            coord_unreachable = False
            while not self.robot.reached_position(pos):
                time.sleep(0.1)
                time_1 = time.time()
                if (time_1 - time_0) > 5:
                    coord_unreachable = True
                    break
            if coord_unreachable:
                continue

            # save pose file
            pose = self.robot.get_joint_info()
            poses.append(pose[:6])
            json_fn = os.path.join(self.save_path, 'pose_{0:04d}.json').format(i)
            with open(json_fn, 'w') as file:
                pose_dict = {'x': pose[0], 'y': pose[1], 'z': pose[2],
                             'rot_x': pose[3], 'rot_y': pose[4],
                             'rot_z': pose[5], 'depth_scaling': 0.000125}
                json.dump(pose_dict, file)

            # save rgb and depth file
            rgb, depth = self.cam.get_image(crop=False)
            rgb_fn = os.path.join(self.save_path, 'rgb_{0:04d}.png'.format(i))
            cv2.imwrite(rgb_fn, rgb[:, :, ::-1])
            depth /= 0.000125
            depth = depth.astype(np.uint16)
            depth_fn = os.path.join(self.save_path, 'depth_{0:04d}.png'.format(i))
            cv2.imwrite(depth_fn, depth)

            # plot
            cv2.imshow("win", rgb[:, :, ::-1])
            cv2.waitKey(1)
            print(i)

        np.savez(os.path.join(self.save_path, 'poses.npz'),
                 poses=poses,
                 center=self.center)


def main():
    '''create a dataset'''
    pose_sampler = RandomViewRecorder(save_folder="sick_vacuum", num_samples=50)
    pose_sampler.create_dataset()


if __name__ == '__main__':
    main()
