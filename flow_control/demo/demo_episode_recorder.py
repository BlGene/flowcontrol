"""
Records demo episodes from sim or real robot.
"""
import os
import re
import math
import json
import datetime

import numpy as np
from gym import Wrapper
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from PIL import Image

class Recorder(Wrapper):
    """
    Records demo episodes from sim or real robot.
    """
    def __init__(self, env, obs_type, save_dir):
        super(Recorder, self).__init__(env)
        assert self.obs_type in ['image_state', "img_color", "image_state_reduced"]
        self.obs_type = obs_type
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        try:
            self.ep_counter = max([int(re.findall(r'\d+', f)[0]) for f in os.listdir(save_dir) if f[-4:] == ".npz"]) + 1
        except ValueError:
            self.ep_counter = 0
        print("Recording episode:", self.ep_counter)

        self.actions = []
        self.observations = []
        self.infos = []

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.actions.append(action)
        self.observations.append(observation)
        self.infos.append(info)
        return observation, reward, done, info

    def reset(self):
        if len(self.actions) > 0:
            self.save()
            self.ep_counter += 1

        self.actions = []
        self.observations = []
        self.infos = []

        observation = self.env.reset()
        return observation

    def save_info(self):
        # save info
        info_fn = os.path.join(self.save_dir, "episode_{}_info.json".format(self.ep_counter))
        env_info = self.env.get_info()
        env_info["time"] = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        #env_info["T_tcp_cam"] = self.env.cam.get_extrinsic_calibration()

        with open(info_fn, 'w') as f_obj:
            json.dump(env_info, f_obj)

    def save_data(self):
        """
        Save data to files.
        """
        path = os.path.join(self.save_dir, "episode_{}").format(self.ep_counter)
        np.savez_compressed(path,
                            actions=self.actions,
                            steps=len(self.actions),
                            observations=self.observations,
                            infos=self.infos)
        print("saved:", path)
        os.mkdir(path)
        for i, obs in enumerate(self.observations):
            img = obs["rgb_gripper"]
            #cv2.imwrite(os.path.join(path, "img_{:04d}.png".format(i)), img[:, :, ::-1])
            Image.fromarray(img).save(os.path.join(path, "img_{:04d}.png".format(i)))
        print("saved {} w/ length {}".format(path, len(self.actions)))

    def save(self):
        self.save_info()
        self.save_data()


def start_recording_sim(save_dir="./tmp_recordings/default", episode_num=1,
                        mouse=False):
    """
    Record from simulation.
    """
    iiwa = RobotSimEnv(task='pick_n_place', renderer='egl', act_type='continuous',
                       initial_pose='close', max_steps=200, control='absolute-iter',
                       obs_type='image_state', sample_params=False,
                       img_size=(256, 256))

    env = Recorder(env=iiwa, obs_type='image_state_reduced', save_dir=save_dir)
    uenv = env.unwrapped
    policy = True if hasattr(uenv._task, "policy") else False

    env.reset()
    if mouse:
        from robot_io.input_devices.space_mouse import SpaceMouse
        mouse = SpaceMouse(act_type='continuous')

    max_episode_len = 200
    for e in range(episode_num):
        try:
            for i in range(max_episode_len):
                if policy:
                    action, policy_done = uenv._task.policy(env, None)
                elif mouse:
                    action = mouse.handle_mouse_events()
                    mouse.clear_events()
                _, _, done, info = env.step(action)
                # cv2.imshow('win', info['rgb_unscaled'][:, :, ::-1])
                # cv2.waitKey(30)

                if done:
                    break

            env.reset()

        except KeyboardInterrupt:
            break


def start_recording(save_dir='/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/default', max_steps=1e6):
    """
    record from real robot
    """
    import cv2
    from gym_grasping.envs.iiwa_env import IIWAEnv
    from robot_io.input_devices.space_mouse import SpaceMouse

    max_steps = int(max_steps)
    iiwa = IIWAEnv(act_type='continuous', freq=20, obs_type='image_state_reduced',
                   dv=0.01, drot=0.04, joint_vel=0.05,  # trajectory_type='lin',
                   gripper_rot_vel=0.3, joint_acc=0.3, use_impedance=True,
                   reset_pose=(0, -0.56, 0.26, math.pi, 0, math.pi / 2))

    env = Recorder(env=iiwa, obs_type='image_state_reduced', save_dir=save_dir)
    env.reset()

    mouse = SpaceMouse(act_type='continuous', initial_gripper_state='open')
    while 1:
        try:
            for i in range(max_steps):
                print(i, max_steps)
                action = mouse.handle_mouse_events()
                mouse.clear_events()
                _, _, _, info = env.step(action)

                cv2.imshow('win', info['rgb_unscaled'][:, :, ::-1])
                if cv2.waitKey(1) == ord('s'):
                    print("Stopping recording")
                    break
            env.reset()
        except KeyboardInterrupt:
            break


def load_episode(filename):
    """
    load a single episode
    """
    rd = np.load(recording_fn, allow_pickle=True)
    rgb = np.array([obs["rgb_gripper"] for obs in rd["observations"]])
    depth = np.array([obs["depth_gripper"] for obs in rd["observations"]])
    state = [obs["robot_state"] for obs in rd["observations"]]
    actions = rd["actions"]
    return actions, state, rgb, depth


#def load_episode_batch():
#    """
#    load a batch of episodes, and show how many are solved.
#    """
#    folder = "/media/kuka/Seagate Expansion Drive/kuka_recordings/dr/2018-12-18-12-35-21"
#    solved = 0
#    for i in range(96):
#        file = folder + "/episode_{}.npz".format(i)
#        episode = load_episode(file)
#        if episode[6]:
#            solved += 1
#    print(i / solved)


def show_episode(file):
    """
    plot a loaded episode
    """
    import cv2
    _, _, rgb, depth = load_episode(file)
    for i in range(200):
        # print(robot_state_full[i])
        # cv2.imshow("win", img_obs[i][:,:,::-1])
        cv2.imshow("win1", rgdb[i, :, :, ::-1])
        cv2.imshow("win2", depth[i]/np.max(depth[i]))
        print(depth[i])
        cv2.waitKey(0)
        # cv2.imwrite("/home/kuka/lang/robot/master_thesis/figures/example_task/image_{}.png".format(i), kinect_obs[i])






if __name__ == "__main__":
    # show_episode('/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/pick/episode_0.npz')

    # save_dir = './tmp_recordings/pick_n_place'
    # start_recording_sim(save_dir)

    # save_dir = '/media/argusm/Seagate Expansion Drive/kuka_recordings/flow/tmp'
    save_dir = '/home/argusm/kuka_recordings/flow/tmp'
    start_recording(save_dir)
