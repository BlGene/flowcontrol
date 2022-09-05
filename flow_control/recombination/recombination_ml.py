from distutils import errors
from distutils.log import error
import os.path as osp
import os
import numpy as np

from scipy.spatial.transform import Rotation as R

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.demo.demo_episode_recorder import record_sim
from flow_control.flow_control_main import evaluate_control
from flow_control.servoing.module import ServoingModule


def get_best_segment(prob, idx1, idx2):
    seed_prob = prob[idx1, :]
    subset = seed_prob[0:idx2]
    best_idx = np.argmax(subset)

    return best_idx


def run_episode(seed, recs):
    param_info = {"task_selected": 'pick_n_place'}

    # Instantiate environment
    env = RobotSimEnv(task='recombination', renderer='debug', act_type='continuous',
                      initial_pose='close', max_steps=500, control='absolute-full',
                      img_size=(256, 256),
                      sample_params=False,
                      param_info=param_info,
                      seed=seed)

    reward = 0.0
    control_config = dict(mode="pointcloud-abs-rotz", threshold=0.25)

    for idx, rec in enumerate(recs):
        # Initialize servo_module
        sm = ServoingModule(rec, control_config=control_config,
                            plot=False, save_dir=None)
        initial_align = True
        if idx > 0:
            initial_align = False

        # Evaluate control
        _, reward, _, info = evaluate_control(env, sm, max_steps=100,
                                              save_dir=None, initial_align=initial_align)

        del sm

        if reward == 1.0:
            # We can exit now
            break

    del env

    print(f"Servoing completed in {info['ep_length']} steps")
    print(f"Reward: {reward}")

    return reward, info


def main():
    rec_path = './recombination/tmp_test_new_split1/ss'
    seg0 = sorted([osp.join(rec_path, rec) for rec in os.listdir(rec_path) if rec.endswith('seg0')])
    seg1 = sorted([osp.join(rec_path, rec) for rec in os.listdir(rec_path) if rec.endswith('seg1')])

    prob_path = './recombination/probs_ml_debug.npz'
    result_path = './recombination/rewards_ml'
    os.makedirs(result_path, exist_ok=True)

    # Load prob
    prob = np.load(prob_path)['arr_0']

    seeds = range(100, 120, 1)
    steps = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    num_steps = len(steps)

    rewards = np.zeros((len(seeds), num_steps))

    for idx, seed in enumerate(seeds):
        last_rec0, last_rec1 = None, None

        # Now you have all required errors_old
        for step_idx, value in enumerate(steps):
            best_idx = get_best_segment(prob, idx, value)
            rec0 = seg0[best_idx]
            rec1 = seg1[0]

            print(f"Recordings selected are: {rec0}, {rec1}")
            # continue
            if last_rec0 == rec0 and last_rec1 == rec1:
                # This was already tested, use the result
                rewards[idx, step_idx] = rewards[idx, step_idx - 1]
            else:
                # This needs to be tested
                rewards[idx, step_idx], info = run_episode(seed, [rec0, rec1])
                last_rec0 = rec0
                last_rec1 = rec1

            np.savez(osp.join(result_path, 'rewards_ml1.npz'), rewards)


if __name__ == "__main__":
    main()