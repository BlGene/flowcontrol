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
import matplotlib.pyplot as plt

import gym
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
import gym_grasping
from gym_grasping.envs import CurriculumEnvLog
from gym_grasping.envs.grasping_env import GraspingEnv
from gym_grasping.scripts.play_grasping import play
from gym_grasping.scripts.viewer import Viewer

import cv2
from gym_grasping.flow_control.flow_module import FlowModule

import matplotlib.gridspec as gridspec
from collections import deque
class ViewPlots:
    def __init__(self, size=(2,1), threshold=.1):
        plt.ion()
        self.fig = plt.figure(figsize=(8*size[1],3*size[0]))
        gs = gridspec.GridSpec(2,3)
        gs.update(wspace=0.001, hspace=.3) # set the spacing between axes.
        plt.subplots_adjust(wspace=0.5, hspace=0, left=0, bottom=0, right=1, top=1)

        self.num_plots = 4
        self.horizon_timesteps = 50
        self.ax1 = plt.subplot(gs[0,:])
        self.low_1 = plt.subplot(gs[1,0])
        self.low_2 = plt.subplot(gs[1,1])
        self.low_3 = plt.subplot(gs[1,2])

        self.ax = [self.ax1, self.ax1.twinx(), self.ax1.twinx()]
        self.ax.append(self.ax[-1])

        self.cur_plots = [None for _ in range(self.num_plots)]
        self.t = 0
        self.data = [deque(maxlen=self.horizon_timesteps) for _ in range(self.num_plots)]

        self.names = ["loss","demo frame", "demo z","live z"]
        hline = self.ax[0].axhline(y=threshold,color="k")

        # images stuff
        self.low_1_h = self.low_1.imshow(np.zeros((256,256)))
        self.low_1.set_axis_off()
        self.low_1.set_title("live state")
        self.low_2_h = self.low_2.imshow(np.zeros((256,256)))
        self.low_2.set_axis_off()
        self.low_2.set_title("demo state")
        self.low_3_h = self.low_3.imshow(np.zeros((256,256)))
        self.low_3.set_axis_off()
        self.low_3.set_title("flow")
        plt.show()

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
            res = self.ax[i].plot(range(xmin, xmax), list(self.data[i]),color=c,label=l)
            self.cur_plots[i], = res
            self.ax1.set_xlim(xmin, xmax)

        self.ax1.legend(handles=self.cur_plots, loc='upper center')

        self.fig.tight_layout()
        self.fig.canvas.draw()


def evaluate_control(recording, perturbation, env=None, task_name="stack", threshold=0.4, max_steps=1000, mouse=False, plot=True):
    # perturbation goes from -1,1

    # internal parameters
    perturb_actions = False  # perturb gripper position or perturb object pose
    forward_flow = False  # default is backward(demo->obs)
    flow_mask = True  # default is masking
    fitting_control = True  # use least squares fit of FG points
    # TODO(max): maybe erosion code could be removed
    mask_erosion = False
    num_mask = 32 # the number of pixels to retain after erosion
    base_frame = 8  # starting frame
    to_shade = 0.6

    # load files
    flow_recording_fn = "./{}/episode_1_img.npz".format(recording)
    mask_recording_fn = "./{}/episode_1_mask.npz".format(recording)
    state_recording_fn = "./{}/episode_1.npz".format(recording)
    flow_recording = np.load(flow_recording_fn)["img"]
    mask_recording = np.load(mask_recording_fn)["mask"]
    state_recording = np.load(state_recording_fn)
    ee_positions = state_recording["ee_positions"]
    gr_positions = state_recording["gripper_states"]
    size = flow_recording.shape[1:3]
    print("Image shape from recording", size)

    # load flow net (needs image size)
    flow_module = FlowModule(size=size)

    # load env (needs
    if env is None:
        if perturb_actions:
            env = GraspingEnv(task=task_name, renderer='debug', act_type='continuous',
                              max_steps=1e9,
                              img_size=size)

        if perturb_actions is False:
            object_pose = [-0.01428776+to_shade, -0.52183914,  0.15, pi]
            object_pose[0] += perturbation[0]*0
            object_pose[3] += perturbation[1]*pi/2
            print("perturbation", perturbation)
            #object_pose = (.071+perturbation[0]*0.02, -.486+perturbation[1]*0.02, 0.15, 0)
            env = GraspingEnv(task=task_name, renderer='tiny', act_type='continuous',
                              max_steps=1e9, object_pose=object_pose,
                              img_size=size)
    # select frame
    base_image = flow_recording[base_frame]
    base_mask = mask_recording[base_frame]
    base_pos = ee_positions[base_frame]
    grip_state = gr_positions[base_frame]
    max_demo_frame = flow_recording.shape[0] - 1
    if mask_erosion:
        from scipy import ndimage as ndi
        base_mask = ndi.distance_transform_edt(base_mask)
        mask_thr = np.sort(base_mask.flatten())[-num_mask]
        base_mask = base_mask > mask_thr

    if perturb_actions:
        perturb_actions = 20
    else:
        perturb_actions = 0
    if mouse:
        from gym_grasping.robot_io.space_mouse import SpaceMouse
        mouse = SpaceMouse(act_type='continuous')
    if plot:
        view_plots = ViewPlots(threshold=threshold)

    done = False
    for counter in range(max_steps):
        # Controls
        action = [0, 0, 0, 0, 1]
        if mouse:
            action = mouse.handle_mouse_events()
            mouse.clear_events()
        if counter < perturb_actions:
            action = perturbation + [0,0,1]

        if action == [0, 0, 0, 0, 1] and counter > perturb_actions:
            #mean_flow = [0,0]  # disable translation
            z = pos_diff[2] * 10 * 2 * 1.5
            action = [mean_flow[0], -mean_flow[1], z, mean_rot, grip_state]

        # hacky hack to move up if episode is done
        if base_frame == max_demo_frame:
            action = [0,0,1,0,0]

        # Environment Stepping
        state, reward, done, info = env.step(action)
        assert(state.shape == base_image.shape)
        if done:
            print("done. ", reward)
            break

        # Control computation
        # copied from curriculum_env.py, move to grasping_env?
        ee_pos = list(env._p.getLinkState(env.robot.robot_uid, env.robot.flange_index)[0])
        ee_pos[2] += 0.02

        if forward_flow:
            flow = flow_module.step(state, base_image)
        else:
            flow = flow_module.step(base_image, state)
        if flow_mask:
            assert forward_flow == False
            flow_tmp = flow.copy()
            flow_tmp[np.logical_not(base_mask)] = 0

        # Do the masked reference computation here.
        if fitting_control:
            # because I need to use the FG mask
            assert(forward_flow == False)

            x = np.linspace(-1,1,size[0])
            y = np.linspace(-1,1,size[1])

            xv,yv = np.meshgrid(x,y)
            # rotate
            field =  np.stack((yv,-xv),axis=2)

            #print("XXX shape", flow_module.field.shape)
            #set_trace()
            field[:,:,0]*=(84-1)/2*size[0]/256
            field[:,:,1]*=(84-1)/2*size[1]/256
            points = field

            observations = points + flow
            #select foreground
            points = points[base_mask]
            observations = observations[base_mask]
            points = np.pad(points, ((0, 0), (0, 2)), mode="constant")
            observations = np.pad(observations, ((0, 0), (0, 2)), mode="constant")
            from servoing import solve_transform
            from scipy.spatial.transform import Rotation as R
            guess = solve_transform(points, observations)
            r = R.from_dcm(guess[:3,:3])
            xyz = r.as_euler('xyz')
            z = xyz[2]
            # magical gain values for control, these could come form calibration
            mean_flow = guess[0,3]/23, guess[1,3]/23
            mean_rot = -7*z
        else:
            # previously I was using FG points for translation
            # radial field trick for rotation
            mean_flow = np.mean(flow_tmp[:,:,0], axis=(0,1)), np.mean(flow_tmp[:,:,1], axis=(0,1))
            # compute rotation
            rot_field = np.sum(flow*flow_module.inv_field, axis=2)
            mean_rot = np.mean(rot_field, axis=(0, 1))
        if not forward_flow:
            # then reverse direction of actions
            mean_flow = np.array(mean_flow) * -1
            mean_rot = mean_rot * -1
        # Outputs of this block: used at beginning of loop
        #   1. mean_flow
        #   2. mean_rot

        # demonstration stepping code
        pos_diff = base_pos - ee_pos
        loss = np.linalg.norm(mean_flow) + np.abs(mean_rot) + np.abs(pos_diff[2])*10
        if loss < threshold and base_frame < max_demo_frame:
            # this is basically a step function
            base_frame += 1
            base_image = flow_recording[base_frame]
            base_mask = mask_recording[base_frame]
            if mask_erosion:
                base_mask = ndi.distance_transform_edt(base_mask)
                mask_thr = np.sort(base_mask.flatten())[-num_mask]
                base_mask = base_mask > mask_thr
            base_pos = ee_positions[base_frame]
            grip_state = gr_positions[base_frame]
            print("demonstration: ", base_frame, "/", max_demo_frame)

        # plotting code
        if plot or plot_cv:
            # show flow
            flow_img = flow_module.computeImg(flow, dynamic_range=False)
            # show segmentatione edge
            if counter % 5 == 0:
                edge  = np.gradient(base_mask.astype(float))
                edge = (np.abs(edge[0])+ np.abs(edge[1])) > 0
                flow_img[edge] = (255, 0, 0)
        if plot:
            print("loss =",loss, action)
            # show loss, frame number
            view_plots.step(loss, base_frame, base_pos[2], ee_pos[2])
            view_plots.low_1_h.set_data(state)
            view_plots.low_2_h.set_data(base_image)
            view_plots.low_3_h.set_data(flow_img)
            plot_fn = f'./video/{counter:03}.png'
            #plt.savefig(plot_fn, bbox_inches=0)
        if plot_cv:
            img = np.concatenate((state[:,:,::-1], base_image[:,:,::-1], flow_img[:,:,::-1]),axis=1)
            cv2.imshow('window', cv2.resize(img, (300*3,300)))
            cv2.waitKey(1)

    if 'ep_length' not in info:
        info['ep_length'] = counter
    return state, reward, done, info

if __name__ == "__main__":
    import itertools

    task_name = "stack"
    recording = "stack_recordings/episode_118"
    threshold = 0.35 # .40 for not fitting_control

    #task_name = "block"
    #recording = "block_recordings/episode_41"
    #threshold = 0.35 # .40 for not fitting_control


    #task_name = "block"
    #recording = "block_yellow_recordings/episode_1"
    #threshold = 1.8 # .40 for not fitting_control

    samples = sorted(list(itertools.product([-1, 1, -.5, .5, 0], repeat=2)))[:7]

    if len(samples) > 10:  # statistics mode
        save = True
        plot = False
    else:  # dev mode
        save = False
        plot = True
        plot_cv = False

    num_samples = len(samples)
    results = []
    for i, s in enumerate(samples):
        print("starting",i,"/",num_samples)
        state, reward, done, info = evaluate_control(recording,
                                                     list(s),
                                                     task_name=task_name,
                                                     threshold=threshold,
                                                     plot=plot)
        res = dict(offset=s,
                   angle=0,
                   threshold=threshold,
                   reward=reward,
                   ep_length=info['ep_length'])

        results.append(res)
        if save:
            with open('./translation_backward.json',"w") as fo:
                json.dump(results, fo)

        break
