"""
Record random views to test pose estimation system.
"""
import os
import math
import time
import json

import numpy as np
import cv2

from gym_grasping.robot_io.iiwa_controller import IIWAController
from gym_grasping.robot_io.gripper_controller_new import GripperController
from gym_grasping.robot_io.realsense_cam import RealsenseCam


class RandomPoseSampler:
    """
    Record random views to test pose estimation system.
    """

    def __init__(self,
                 save_dir="/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/pose_estimation/",
                 save_folder="default_recording",
                 num_samples=10,
                 center=np.array([0, -0.55, 0.2]),
                 theta_limits=(0, 2 * math.pi),
                 r_limits=(0.05, 0.15),
                 h_limits=(0, 0.1),
                 trans_limits=(-0.1, 0.1),
                 rot_limits=(-np.radians(10), np.radians(10)),
                 pitch_limit=(-np.radians(10), np.radians(10)),
                 roll_limit=(-np.radians(10), np.radians(10))):

        self.save_path = os.path.join(save_dir, save_folder)
        os.makedirs(self.save_path, exist_ok=True)
        self.num_samples = num_samples
        self.robot = IIWAController(use_impedance=False, joint_vel=0.3,
                                    gripper_rot_vel=0.5, joint_acc=0.3)
        gripper = GripperController()
        gripper.home()
        self.cam = RealsenseCam(img_type='rgb_depth')
        self.center = center
        self.theta_limits = theta_limits
        self.r_limits = r_limits
        self.h_limits = h_limits
        self.trans_limits = trans_limits
        self.rot_limits = rot_limits
        self.pitch_limit = pitch_limit
        self.roll_limit = roll_limit

    def sample_pose(self):
        '''sample a  random pose'''
        theta = np.random.uniform(*self.theta_limits)
        rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [np.sin(theta), np.cos(theta), 0],
                            [0, 0, 1]])
        vec = rot_mat @ np.array([1, 0, 0])
        r = np.random.uniform(*self.r_limits)
        vec = vec * r
        theta_offset = np.random.uniform(*self.rot_limits)
        trans = np.cross(np.array([0, 0, 1]), vec)
        trans = trans * np.random.uniform(*self.trans_limits)
        h = np.array([0, 0, 1]) * np.random.uniform(*self.h_limits)
        x = self.center + vec + trans + h
        pitch = np.random.uniform(*self.pitch_limit)
        roll = np.random.uniform(*self.roll_limit)
        return (*x, math.pi + pitch, roll, theta + math.pi / 2 + theta_offset)

    def create_dataset(self):
        '''the main dataset collection loop'''
        self.robot.send_cartesian_coords_abs((*self.center, math.pi, 0, math.pi/2))
        time.sleep(4)

        poses = []
        # start file indexing with 0 and zero pad filenames
        for i in range(self.num_samples):
            pos = self.sample_pose()
            self.robot.send_cartesian_coords_abs(pos)
            t0 = time.time()
            coord_unreachable = False
            while not self.robot.reached_position(pos):
                time.sleep(0.1)
                t1 = time.time()
                if (t1 - t0) > 5:
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
            # plot
            cv2.imwrite(depth_fn, depth)
            cv2.imshow("win", rgb[:, :, ::-1])
            cv2.waitKey(1)
            print(i)

        np.savez(os.path.join(self.save_path, 'poses.npz'),
                 poses=poses,
                 center=self.center)


def main():
    '''create a dataset'''
    pose_sampler = RandomPoseSampler(save_folder="ECU", num_samples=50)
    pose_sampler.create_dataset()


if __name__ == '__main__':
    main()
