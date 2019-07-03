"""
Testing file for development, to experiment with evironments.
"""
import sys
import argparse
import json
import time
import traceback
from collections import Counter, OrderedDict
from pdb import set_trace
from math import pi
import numpy as np

import pybullet as p
import gym
import gym_grasping
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from gym_grasping.envs import CurriculumEnvLog
from gym_grasping.envs.grasping_env import GraspingEnv
from gym_grasping.scripts.play_grasping import play
from gym_grasping.scripts.viewer import Viewer

import matplotlib.pyplot as plt


# load flow module once only
import cv2
from gym_grasping.flow_control.flow_module import FlowModule
flow_module = FlowModule()
flow_recording_fn = "../flow_control/bolt_recordings/episode_2/episode_2_img.npz"
flow_recording = np.load(flow_recording_fn)["img"]

state_recording_fn = "../flow_control/bolt_recordings/episode_2/episode_2.npz"
state_recording = np.load(state_recording_fn)
ee_positions = state_recording["ee_positions"]
gr_positions = state_recording["gripper_states"]


import matplotlib.gridspec as gridspec
from collections import deque
class ViewPlots:
    def __init__(self, size=(1,1), threshold=.1):
        plt.ion()
        self.fig = plt.figure(figsize=(8,3))#figsize = size)
        gs = gridspec.GridSpec(*size)
        gs.update(wspace=0.001, hspace=0.001) # set the spacing between axes.
        plt.subplots_adjust(wspace=0.5, hspace=0, left=0, bottom=0, right=1, top=1)

        self.num_plots = 2
        self.horizon_timesteps = 30 * 5
        self.ax1 = plt.subplot(gs[0,0])
        self.ax = [self.ax1, self.ax1.twinx()]

        self.cur_plots = [None for _ in range(self.num_plots)]
        self.t = 0
        self.data = [deque(maxlen=self.horizon_timesteps) for _ in range(self.num_plots)]

        self.names = ["loss","demo frame num"]
        hline = self.ax[0].axhline(y=threshold,color="k")

    def __del__(self):
        plt.ioff()
        plt.close()


    def step(self, *obs):
        for point, series in zip(obs, self.data):
            series.append(point)

        self.t += 1
        xmin, xmax = max(0, self.t - self.horizon_timesteps), self.t

        for plot in self.cur_plots:
            if plot is not None:
                plot.remove()

        for i in range(self.num_plots):
            c = 'C{}'.format(i)
            l = self.names[i]
            self.cur_plots[i], = self.ax[i].plot(range(xmin, xmax), list(self.data[i]),color=c,label=l)
            self.ax1.set_xlim(xmin, xmax)

        self.ax[0].legend(loc='lower left')
        self.ax[1].legend(loc='upper left')

        self.fig.tight_layout()
        self.fig.canvas.draw()





def sample(sample, env=None, task_name="bolt", mouse=True, ):

    if env is None:
        env = GraspingEnv(task=task_name, renderer='tiny', act_type='continuous',
                          max_steps=1e9)#object_pose=(0.07141055, -0.48649803, 0.15, 0))

    #env = CurriculumEnvSimple(robot='kuka', task='stack', renderer='debug', act_type='continuous', initial_pose='close',
    #                          max_steps=3, obs_type='image_color', use_dr=False)
    #env = gym.make("KukaBlock_dr-v0")
    policy = env._task.get_policy(mode='play')(env)

    # was this for no jerks at end of policy?
    policy_defaults = policy.get_defaults()

    control_names = env.robot.control_names
    if policy_defaults is not None:
        defaults = policy.get_defaults()
    else:
        defaults = [0,]*len(control_names)

    motors_ids = []
    for cn, dv in zip(control_names, defaults):
        motors_ids.append(p.addUserDebugParameter(cn, -1, 1, dv))
    motors_ids.append(p.addUserDebugParameter("debug", -1, 1, 0))

    if mouse:
        from gym_grasping.robot_io.space_mouse import SpaceMouse
        mouse = SpaceMouse(act_type='continuous')

    debug = False
    done = False
    counter = 0
    min_loss = 10
    mean_flows = []
    base_frame = 8
    threshold = .4
    base_image = flow_recording[base_frame]
    base_pos = ee_positions[base_frame]
    grip_state = gr_positions[base_frame]
    num_demo_frames = flow_recording.shape[0]
    view_plots = ViewPlots(threshold=threshold)

    while counter < 1e5:
        # Controls
        if mouse:
            action = mouse.handle_mouse_events()
            mouse.clear_events()
        else:
            action = []
            for motor_id in motors_ids:
               action.append(p.readUserDebugParameter(motor_id))
            debug = action[-1] > .5

        action = policy.act(env, action)

        if counter == 0:
            action = sample + [0,0,1]

        if action == [0, 0, 0, 0, 1] and counter > 20:
            #mean_flow = [0,0]  # disable translation
            z = pos_diff[2] * 5
            action = [mean_flow[0], -mean_flow[1], z, mean_rot, grip_state]

        # Stepping
        state, reward, done, info = env.step(action)
        # print("reward:", reward)
        if debug == 1.0:
            env.reset()
        if reward == 1.0:
            done = False
            if reward == 1:
                pass
            env.reset()
        if done:
            env.reset(data=True)

        # Computation
        # copied from curriculum_env.py, move to grasping_env?
        ee_pos = list(env._p.getLinkState(env.robot.robot_uid, env.robot.flange_index)[0])
        ee_pos[2] += 0.02

        # compute flow and translation
        flow = flow_module.step(state, base_image)
        mean_flow = np.mean(flow[:,:,0], axis=(0,1)), np.mean(flow[:,:,1], axis=(0,1))
        mean_flows.append(mean_flow)

        # compute rotation
        rot_field = np.sum(flow*flow_module.inv_field, axis=2)
        mean_rot = np.mean(rot_field,axis=(0,1))

        # compute height
        radial_field = np.sum(flow*flow_module.inv_rad_field, axis=2)
        mean_rad = np.mean(radial_field,axis=(0,1))

        pos_diff = base_pos - ee_pos

        loss = np.linalg.norm(mean_flow) +  np.abs(mean_rot)
        min_loss = min(loss, min_loss)
        if loss < threshold:
            if base_frame+1 == num_demo_frames:
                print("Done", counter)
                break
            # this is basically a step function
            base_frame += 1
            base_image = flow_recording[base_frame]
            base_pos = ee_positions[base_frame]
            grip_state = gr_positions[base_frame]
            print("increment: ", base_frame, "/", num_demo_frames)

        #print(loss,)
        view_plots.step(loss, base_frame)

        # show flow
        flow_img = flow_module.computeImg(flow, dynamic_range=False)
        img = np.concatenate((state[:,:,::-1], base_image[:,:,::-1],flow_img[:,:,::-1]),axis=1)
        cv2.imshow('window', cv2.resize(img, (300*3,300)))
        cv2.waitKey(1)

        counter += 1

    del view_plots

    return mean_flows

if __name__ == "__main__":
    from itertools import combinations_with_replacement
    # do one step calibration first, everything else staturates anyway.
    samples = list(combinations_with_replacement([-1, 1, 0], 2))

    mean_flows_collection = []
    for i in samples:
        mean_flows = sample(list(i))
        mean_flows_collection.append(mean_flows)

    for i, mf in zip(samples, mean_flows_collection):
        val = np.mean(mf[25:30],axis=0)
        print(i,val)
        plt.plot(mf)

    plt.show()

