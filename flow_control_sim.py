"""
Testing file for development, to experiment with evironments.
"""
import logging
import time
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from gym_grasping.flow_control.servoing_module import ServoingModule

def evaluate_control(env, recording, episode_num, start_index=0,
                     control_config=None, max_steps=1000,
                     plot=True):
    """
    Function that runs the policy.
    """
    assert env is not None

    # load the servo module
    servo_module = ServoingModule(recording,
                                  episode_num=episode_num,
                                  start_index=start_index,
                                  control_config=control_config,
                                  camera_calibration=env.camera.calibration,
                                  plot=plot, save_dir=None)
    servo_action = None
    servo_done = False
    done = False
    for counter in range(max_steps):
        # choose action
        action = None
        if counter > 0:  # inital frame dosen't have action
            action = servo_action

        # environment stepping
        state, reward, done, info = env.step(action)

        # compute action
        if isinstance(env, RobotSimEnv):
            # TODO(max): fix API change between sim and robot
            obs_image = state
        else:
            obs_image = info['rgb_unscaled']
        ee_pos = info['robot_state_full'][:6]  # take three position values
        servo_res = servo_module.step(obs_image, ee_pos, live_depth=info['depth'])
        servo_action, _, servo_done, serov_info = servo_res

        if done:
            break

    info['ep_length'] = counter

    return state, reward, done, info


def main():
    """
    The main function that loads the recording, then runs policy.
    """
    logging.basicConfig(level=logging.DEBUG, format="")

    recording, episode_num = "./tmp_test/pick_n_place", 0
    control_config = dict(mode="pointcloud",
                          gain_xy=50,
                          gain_z=100,
                          gain_r=15,
                          threshold=0.35)

    # TODO(max): save and load these value from a file.
    task_name = "pick_n_place"
    robot = "kuka"
    renderer = "debug"
    control = "relative"

    env = RobotSimEnv(task=task_name, robot=robot, renderer=renderer,
                      control=control, max_steps=500, show_workspace=False,
                      param_randomize=True, img_size=(256, 256))

    state, reward, done, info = evaluate_control(env, recording,
                                                 episode_num=episode_num,
                                                 control_config=control_config,
                                                 plot=True)
    print("reward:", reward)


if __name__ == "__main__":
    main()
