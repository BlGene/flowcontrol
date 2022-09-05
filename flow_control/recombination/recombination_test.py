import json
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

def get_best_segments(error_front, error_rear, error_matrix, error_fn='min'):
    x, y = error_matrix.shape
    best_error_fn = {'error': np.inf, 'idx1': -1, 'idx2': -1}

    if error_fn == 'min':
        idx1 = np.argmin(error_front)
        idx2 = np.argmin(error_rear)
        best_error_fn['idx1'] = idx1
        best_error_fn['idx2'] = idx2

        return best_error_fn

    for i in range(x):
        for j in range(y):
            total_error_fn = np.inf
            if error_fn == 'sum':
                total_error_fn = error_matrix[i][j] + error_front[i] + error_rear[j]
            elif error_fn == 'prod':
                total_error_fn = error_front[i] * error_rear[j]

            if total_error_fn < best_error_fn['error']:
                best_error_fn['error'] = total_error_fn
                best_error_fn['idx1'] = i
                best_error_fn['idx2'] = j

    return best_error_fn

def run_episode(seed, recs, seed_vid_dir):
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

    frame_number = 0

    for idx, rec in enumerate(recs):
        # Initialize servo_module
        sm = ServoingModule(rec, control_config=control_config, plot=False, save_dir=None)

        initial_align = True
        if idx > 0:
            initial_align = False

        # Evaluate control
        _, reward, _, info = evaluate_control(env, sm, max_steps=130,
                                              save_dir=seed_vid_dir, initial_align=initial_align,
                                              frame_number=frame_number)
        frame_number = info['frame_number']
        del sm

        if reward == 1.0:
            # We can exit now
            break

    del env

    print(f"Servoing completed in {info['ep_length']} steps")
    print(f"Reward: {reward}")

    return reward, info

def main():
    # rec_path = './recombination/tmp_test_split_overlap/ss'
    rec_path = './recombination/tmp_test_new_split1'
    seg0 = sorted([osp.join(rec_path, rec) for rec in os.listdir(rec_path) if rec.endswith('seg0')])
    seg1 = sorted([osp.join(rec_path, rec) for rec in os.listdir(rec_path) if rec.endswith('seg1')])

    error_path = './recombination/errors_overlap'
    result_path = './recombination/rewards_abs'
    os.makedirs(result_path, exist_ok=True)

    # Load Error matrix and Rear Errors
    error_rear_path = osp.join(error_path, 'errors_rear.npz')
    error_matrix_path = osp.join(error_path, 'error_matrix.npz')

    errors_rear = np.load(error_rear_path)['arr_0']
    error_matrix = np.load(error_matrix_path)['arr_0']

    seeds = range(100, 120, 1)    
    steps = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75]
    num_steps = len(steps)

    for error_fn in ['min', 'sum', 'prod']:
        rewards = np.zeros((len(seeds), num_steps))
        selected_recordings = dict()

        for idx, seed in enumerate(seeds):
            seed_dir = osp.join(error_path, str(seed))
            error_front_path = osp.join(seed_dir, 'errors_front.npz')
            errors_front = np.load(error_front_path)['arr_0']

            seed_recordings = {}

            last_rec0, last_rec1 = None, None

            # Now you have all required errors_old
            for step_idx, value in enumerate(steps):
                ef = errors_front[0:value]
                er = errors_rear[0:value]
                em = error_matrix[0:value, 0:value]
                best_segments = get_best_segments(ef, er, em, error_fn)

                idx1, idx2 = best_segments['idx1'], best_segments['idx2']

                rec0, rec1 = seg0[idx1], seg1[idx2]
                print(f"Recordings selected are: {rec0}, {rec1}")
                # continue
                if last_rec0 == rec0 and last_rec1 == rec1:
                    # This was already tested, use the result
                    rewards[idx, step_idx] = rewards[idx, step_idx - 1]
                    print(f"skipped {seed} and {step_idx}")
                else:
                    # This needs to be tested
                    seed_vid_dir = os.path.join(result_path, f"{seed}", f"{step_idx}")
                    os.makedirs(seed_vid_dir, exist_ok=True)

                    seed_recordings[step_idx] = {'r1': rec0, 'r2': rec1}
                    rewards[idx, step_idx], info = run_episode(seed, [rec0, rec1], seed_vid_dir)
                    last_rec0 = rec0
                    last_rec1 = rec1
                np.savez(osp.join(result_path, f'rewards_{error_fn}_all.npz'), rewards)
            selected_recordings[seed] = seed_recordings

        with open(osp.join(result_path, f'selected_recordings_{error_fn}.json'), 'w') as outfile:
            json.dump(selected_recordings, outfile)

    
if __name__ == "__main__":
    main()