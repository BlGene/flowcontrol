import math

from pygments.formatters import img
import cv2
from gym_grasping.robot_envs.iiwa_env import IIWAEnv
import numpy as np
import time
from gym import Wrapper
from gym import spaces
import os
import datetime
from os import listdir
from os.path import isfile, join
import re
from a2c_ppo_acktr.play_model import Model, build_env, render_obs
from gym_grasping.robot_io.space_mouse import SpaceMouse


class Recorder(Wrapper):
    def __init__(self, env, obs_type, save_dir):
        super(Recorder, self).__init__(env)
        self.ep_counter = 0
        self.save_dir = save_dir
        try:
            os.mkdir(self.save_dir)
            with(open(os.path.join(self.save_dir, "info.txt"), 'w')) as f:
                f.write("time of recording: " + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '\n')

        except FileExistsError:
            self.ep_counter = max([int(re.findall(r'\d+', f)[0]) for f in listdir(save_dir) if f[-4:] == ".npz"]) + 1
        print(self.ep_counter)
        self.initial_configuration = None
        self.actions = []
        self.robot_state_observations = []
        self.robot_state_full = []
        self.img_obs = []
        self.depth_imgs = []
        self.unscaled_imgs = []
        self.obs_type = obs_type
        # self.observation_space = spaces.Box(low=0, high=255,
        #                                     shape=(84, 84, 3), dtype='uint8')
        assert self.obs_type in ["img_state", "img_color", "img_state_reduced"]

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.actions.append(action)
        self.robot_state_observations.append(observation['robot_state'])
        self.robot_state_full.append(info['robot_state_full'])
        self.img_obs.append(observation['img'])
        if self.obs_type == "img_color":
            observation = observation['img']
        self.depth_imgs.append(info['depth'])
        self.unscaled_imgs.append(info['rgb_unscaled'])
        return observation, reward, done, info

    def reset(self):
        if len(self.img_obs) > 0:
            self.save()
            self.ep_counter += 1
        self.initial_configuration = None
        self.actions = []
        self.robot_state_observations = []
        self.robot_state_full = []
        self.img_obs = []
        self.depth_imgs = []
        self.unscaled_imgs = []

        observation = self.env.reset()
        self.initial_configuration = self.env._get_obs()[1]['robot_state_full'][:4]
        if self.obs_type == "img_color":
            observation = observation['img']
        return observation

    def save(self):
        path = os.path.join(self.save_dir, "episode_{}").format(self.ep_counter)
        np.savez(path,
                 initial_configuration=self.initial_configuration,
                 actions=self.actions,
                 steps=len(self.actions),
                 robot_state_observations=self.robot_state_observations,
                 robot_state_full=self.robot_state_full,
                 img_obs=self.img_obs,
                 depth_imgs=self.depth_imgs,
                 rgb_unscaled=self.unscaled_imgs)
        os.mkdir(path)
        for i, img in enumerate(self.unscaled_imgs):
            cv2.imwrite(os.path.join(path, "img_{}.png".format(i)), img[:, :, ::-1])


def start_recording():
    iiwa = IIWAEnv(act_type='continuous', freq=20, obs_type='img_state_reduced', dv=0.01, drot=0.2, use_impedance=True,
                   use_real2sim=False,  max_steps=400)

    save_dir = '/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/control_test/'


    env = Recorder(env=iiwa, obs_type='img_state_reduced', save_dir=save_dir)
    env.reset()
    mouse = SpaceMouse(act_type='continuous')
    max_episode_len = 200
    while 1:
        try:
            for i in range(max_episode_len):
                print(i,max_episode_len)
                action = mouse.handle_mouse_events()
                mouse.clear_events()
                ob, _, done, info = env.step(action)
                #cv2.imshow("win", cv2.resize(ob['img'][:, :, ::-1], (300, 300)))
                cv2.imshow('win', info['rgb_unscaled'][:, :, ::-1])
                cv2.waitKey(1)
            env.reset()
        except KeyboardInterrupt:
            break


def load_episode(filename):
    data = np.load(filename)
    initial_configuration = data["initial_configuration"]
    actions = data["actions"]
    state_obs = data["robot_state_observations"]
    robot_state_full = data["robot_state_full"]
    img_obs = data["img_obs"]
    # kinect_obs = data["kinect_imgs"]
    depth = data['depth_imgs']
    rgb_unscaled = data['rgb_unscaled']
    steps = data["steps"]
    return initial_configuration, actions, state_obs, robot_state_full, img_obs, depth, rgb_unscaled, steps


def load_episode_batch():
    folder = "/media/kuka/Seagate Expansion Drive/kuka_recordings/dr/2018-12-18-12-35-21"
    solved = 0
    for i in range(96):
        file = folder + "/episode_{}.npz".format(i)
        ep = load_episode(file)
        if ep[6]:
            solved += 1
    print(i / solved)


def show_episode(file):
    initial_configuration, actions, state_obs, robot_state_full, img_obs, depth, rgb_unscaled, steps = load_episode(file)
    print(initial_configuration)
    for i in range(200):
        # print(robot_state_full[i])
        # cv2.imshow("win", img_obs[i][:,:,::-1])
        cv2.imshow("win1", rgb_unscaled[i, :,:,::-1])
        cv2.imshow("win2", depth[i]/np.max(depth[i]))
        print(depth[i])
        cv2.waitKey(0)
        # cv2.imwrite("/home/kuka/lang/robot/master_thesis/figures/example_task/image_{}.png".format(i), kinect_obs[i])


if __name__ == "__main__":
    # show_episode('/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/pick/episode_0.npz')
    start_recording()
