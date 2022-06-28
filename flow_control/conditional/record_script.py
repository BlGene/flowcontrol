"""
Test servoing for the shape sorting task.
"""
import os
import shutil
from scipy.spatial.transform import Rotation as R

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.demo.demo_episode_recorder import record_sim
from flow_control.flow_control_main import evaluate_control
from flow_control.servoing.module import ServoingModule


def record_multiple_episodes(num_episodes=10, n_digits=6):
    os.makedirs("./tmp_test", exist_ok=True)

    object_selected = "trapeze"
    #object_selected = "semicircle"
    #object_selected = "oval"

    orn_options = dict(
        rR=None
        #rN=R.from_euler("xyz", (0, 0, 0), degrees=True).as_quat(),
        #rZ=R.from_euler("xyz", (0, 0, 20), degrees=True).as_quat(),
        #rY=R.from_euler("xyz", (0, 90, 0), degrees=True).as_quat(),
        #rX=R.from_euler("xyz", (90, 0, 0), degrees=True).as_quat(),
        #rXZ=R.from_euler("xyz", (180, 0, 160), degrees=True).as_quat()
        )

    for name, orn in orn_options.items():
        for i in range(num_episodes):
            env = RobotSimEnv(task='shape_sorting', renderer='egl', act_type='continuous',
                              initial_pose='close', max_steps=200, control='absolute-full',
                              img_size=(256, 256),
                              sample_params=False,
                              param_info={"object_selected": object_selected},
                              seed=i)

            save_dir = f"./tmp_test/shape_sorting_{object_selected}_{name}_{i:0{n_digits}d}"
            if os.path.isdir(save_dir):
                # lsof file if there are NSF issues.
                shutil.rmtree(save_dir)
            record_sim(env, save_dir)
            del env


if __name__ == "__main__":
    record_multiple_episodes()
