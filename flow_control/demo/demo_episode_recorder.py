"""
Records demo episodes from sim or real robot.
"""
import math

import numpy as np
from gym_grasping.envs.robot_sim_env import RobotSimEnv

from robot_io.recorder.simple_recorder import SimpleRecorder


def start_recording_sim(save_dir="./tmp_recordings/default", episode_num=1,
                        mouse=False):
    """
    Record from simulation.
    """
    env = RobotSimEnv(task='pick_n_place', renderer='egl', act_type='continuous',
                      initial_pose='close', max_steps=200, control='absolute-iter',
                      obs_type='image_state', sample_params=False,
                      img_size=(256, 256))

    rec = SimpleRecorder(env, save_dir=save_dir)
    policy = True if hasattr(env._task, "policy") else False

    env.reset()
    if mouse:
        from robot_io.input_devices.space_mouse import SpaceMouse
        mouse = SpaceMouse(act_type='continuous')

    max_episode_len = 200
    for e in range(episode_num):
        try:
            for i in range(max_episode_len):
                if policy:
                    action, policy_done = env._task.policy(env, None)
                elif mouse:
                    action = mouse.handle_mouse_events()
                    mouse.clear_events()

                obs, rew, done, info = env.step(action)
                save_action = dict(motion=(action[0:3], action[3], action[4]), ref=None)
                rec.step(save_action, obs, rew, done, info)

                # cv2.imshow('win', info['rgb_unscaled'][:, :, ::-1])
                # cv2.waitKey(30)

                if done:
                    break

            rec.save()
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
    env = IIWAEnv(act_type='continuous', freq=20, obs_type='image_state_reduced',
                  dv=0.01, drot=0.04, joint_vel=0.05,  # trajectory_type='lin',
                  gripper_rot_vel=0.3, joint_acc=0.3, use_impedance=True,
                  reset_pose=(0, -0.56, 0.26, math.pi, 0, math.pi / 2))

    #env = Recorder(env=iiwa, obs_type='image_state_reduced', save_dir=save_dir)
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
    rd = np.load(filename, allow_pickle=True)
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
        cv2.imshow("win1", rgb[i, :, :, ::-1])
        cv2.imshow("win2", depth[i] / np.max(depth[i]))
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
