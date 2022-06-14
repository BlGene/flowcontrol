"""
Records demo episodes from sim or real robot.
"""
import math

import os.path
import numpy as np
from gym_grasping.envs.robot_sim_env import RobotSimEnv

from robot_io.recorder.simple_recorder import SimpleRecorder


def record_sim(env, save_dir="./tmp_recordings/default",
               mouse=False, max_episode_len=200, save_first_action=True):
    """
    Record from simulation.
    Given transitions of (s, a, s' r), save (a, s', r).

    Arguments:
        env: the env to use, will use env.policy if avaliable
        save_dir: directory into which to save recording
        max_episode_len: number of steps to record for
        save_first_action: save the first action as (None, s', r)
    """

    if os.path.isdir(save_dir):
        # raise an error here to avoid concatenating steps from different episodes
        raise ValueError(f"Recording error, folder exists: f{save_dir}")

    rec = SimpleRecorder(env, save_dir=save_dir)
    policy = True if hasattr(env._task, "policy") else False

    if mouse:
        from robot_io.input_devices.space_mouse import SpaceMouse
        mouse = SpaceMouse(act_type='continuous')

    try:
        for _ in range(max_episode_len):
            if save_first_action:
                action, control = None, None
                save_first_action = False
            elif policy:
                action, control, _, p_info = env._task.policy(env)
            elif mouse:
                action = mouse.handle_mouse_events()
                mouse.clear_events()

            if action is not None or control is not None:
                assert control.dof == "xyzquatg"
                assert len(action) == 8
                save_action = dict(motion=(action[0:3], action[3:7], action[7]), ref=None)
            else:
                pseudo_act = env.robot.get_tcp_pos_orn()
                save_action = dict(motion=(pseudo_act[0], pseudo_act[1], 1), ref=None)
                p_info = {"wp_name":"locate-1", "move_anchor": "rel"}

            obs, rew, done, info = env.step(action, control)
            cmb_info = {**info, **p_info}
            rec.step(save_action, obs, rew, done, cmb_info)

            if done:
                break

        print("Recording ended with reward: ", rew)
        rec.save()
        env.reset()

    except KeyboardInterrupt:
        return

def record_real(save_dir='/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/default', max_steps=1e6):
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

    # env = Recorder(env=iiwa, obs_type='image_state_reduced', save_dir=save_dir)
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


def test_record_sim():
    env = RobotSimEnv(task='pick_n_place', renderer='egl', act_type='continuous',
                      initial_pose='close', max_steps=200, control='absolute-full',
                      obs_type='image_state', sample_params=False,
                      img_size=(256, 256))

    save_dir = './tmp_recordings/pick_n_place'
    record_sim(env, save_dir)


if __name__ == "__main__":
    test_record_sim()
