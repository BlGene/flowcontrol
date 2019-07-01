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
flow_recording_fn = "../flow_control/bolt_recordings/episode_1/episode_1_img.npz"
flow_recording = np.load(flow_recording_fn)["img"]
base_frame = flow_recording[0]


def sample(sample, env=None, task_name="bolt", mouse=True, ):

    if env is None:
        env = GraspingEnv(task=task_name, renderer='tiny', act_type='continuous',
                          max_steps=1e9 )#object_pose=(0.07141055, -0.48649803, 0.15, 0))


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
    mean_flows = []

    while counter < 1e5:
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
            print("c", np.linalg.norm(mean_flow))
            action = [mean_flow[0], -mean_flow[1]] + [0,0,1]

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

        # compute flow
        flow = flow_module.step( state,base_frame)
        mean_flow = np.mean(flow[:,:,0], axis=(0,1)), np.mean(flow[:,:,1], axis=(0,1))
        mean_flows.append(mean_flow)

        # show flow
        flow_img = flow_module.computeImg(flow, dynamic_range=False)
        #print(flow_module.field.max())
        #print(flow_field
        #rot_field = np.linalg.norm(flow_module.field * flow,axis=2) * 40
        #print(rot_field.m)
        #flow_img = np.stack((rot_field,rot_field,rot_field),axis=2).astype(np.uint8)



        img = np.concatenate((state[:,:,::-1], base_frame[:,:,::-1],flow_img[:,:,::-1]),axis=1)
        cv2.imshow('window', cv2.resize(img, (300*3,300)))
        cv2.waitKey(1)

        counter += 1

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

