"""
Record random views to test pose estimation system.
"""
import os
import math
import sys
import time
import json
import logging
import datetime

from tqdm import tqdm
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
                 save_dir="/media/argusm/Seagate Expansion Drive/"
                          "kuka_recordings/flow/pose_estimation/",
                 # save_dir="/home/kuka/pose_estimation",
                 save_folder="sensor",
                 object_height=.02,  # in [m]
                 num_samples=50):

        super().__init__(object_height)
        self.save_dir = os.path.join(save_dir, save_folder)
        os.makedirs(self.save_dir, exist_ok=True)

        self.num_samples = num_samples
        self.start_samples = self.get_existing_samples()

        self.robot = IIWAController(use_impedance=False, joint_vel=0.3,
                                    gripper_rot_vel=0.5, joint_acc=0.3)
        gripper = WSG50Controller()
        gripper.home()
        self.cam = RealsenseSR300(img_type='rgb_depth')
        self.depth_scaling = 8000.0

    def get_existing_samples(self):
        import glob
        path = os.path.join(self.save_dir, 'depth_*.png')
        depth_fns = glob.glob(path)
        if len(depth_fns) == 0:
            return 0
        last_file = sorted(depth_fns)[-1]
        last_file = last_file.replace(path.replace("*.png", ""), "")
        last_file = last_file.replace(".png", "")
        last_file = int(last_file)
        return last_file

    def save_info(self):
        # save info
        info_fn = os.path.join(self.save_dir, "info.json".format(self.ep_counter))
        env_info = self.env.get_info()
        env_info["time"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        env_info["depth_scaling"] = self.depth_scaling
        with open(info_fn, 'w') as f_obj:
            json.dump(env_info, f_obj)

    def create_dataset(self):
        '''the main dataset collection loop'''
        self.save_info()
        self.robot.send_cartesian_coords_abs_PTP((*self.center, math.pi, 0, math.pi / 2))
        time.sleep(4)

        poses = []
        # start file indexing with 0 and zero pad filenames
        for i in tqdm(range(self.start_samples, self.num_samples)):
            pos = list(self.sample_pose())
            pos = tuple(np.clip(pos, self.robot.workspace_min, self.robot.workspace_max))
            # pos = (pos[0], pos[1], pos[2], math.pi, 0, math.pi / 2)  # only positions
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
                logging.warning("Coordinates not reached.")
                # continue

            time.sleep(.5)  # wait, so that image is not blurry.

            # save pose file
            pose = self.robot.get_info()["tcp_pose"]
            poses.append(pose[:6])
            json_fn = os.path.join(self.save_dir, 'pose_{0:04d}.json').format(i)
            with open(json_fn, 'w') as file:
                pose_dict = {'x': pose[0], 'y': pose[1], 'z': pose[2],
                             'rot_x': pose[3], 'rot_y': pose[4],
                             'rot_z': pose[5]}
                json.dump(pose_dict, file)

            # save rgb and depth file
            rgb, depth = self.cam.get_image(crop=False)
            rgb_fn = os.path.join(self.save_dir, 'rgb_{0:04d}.png'.format(i))
            print(rgb_fn)
            cv2.imwrite(rgb_fn, rgb[:, :, ::-1])
            depth = (depth * self.depth_scaling).round().astype(np.uint16)
            depth_fn = os.path.join(self.save_dir, 'depth_{0:04d}.png'.format(i))
            cv2.imwrite(depth_fn, depth)

            # plot
            cv2.imshow("win", rgb[:, :, ::-1])
            cv2.waitKey(1)
            print(i)


def print_poses():
    ps = RandomPoseSampler()
    for i in range(50):
        pose = ps.sample_pose()
        print(pose)


def main():
    logging.basicConfig(level=logging.DEBUG, format="")
    '''create a dataset'''
    pose_sampler = RandomViewRecorder()
    pose_sampler.create_dataset()


if __name__ == '__main__':
    main()

