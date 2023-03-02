"""
Records demo_segment episodes from simulation
"""
import os.path
import numpy as np

from robot_io.recorder.simple_recorder import SimpleRecorder
from gym_grasping.envs.robot_sim_env import RobotSimEnv


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
        for i in range(max_episode_len):
            if save_first_action:
                action = None
                save_first_action = False
            elif policy:
                action, _, p_info = env._task.policy(env)
            elif mouse:
                action = mouse.handle_mouse_events()
                mouse.clear_events()

            if action is not None and not isinstance(action, dict):
                assert len(action) == 8
                save_action = dict(motion=(action[0:3], action[3:7], action[7]), ref=None)
            elif isinstance(action, dict):
                save_action = action
            else:
                pseudo_act = env.robot.get_tcp_pos_orn()
                save_action = dict(motion=(pseudo_act[0], pseudo_act[1], 1), ref=None)
                if i == 0:
                    wp_name = "start"
                else:
                    wp_name = "pseudeo"
                p_info = {"wp_name":wp_name, "move_anchor": "abs"}

            obs, rew, done, info = env.step(action)
            cmb_info = {**info, **p_info}
            rec.step(obs, save_action, None, rew, done, cmb_info)

            if done:
                break

        print("Recording ended with reward: ", rew)
        rec.save()
        env.reset()

    except KeyboardInterrupt:
        return


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
