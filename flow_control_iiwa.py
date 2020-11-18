"""
Testing file for development, to experiment with evironments.
"""
import os
import math
import time
from gym_grasping.envs.iiwa_env import IIWAEnv
from gym_grasping.flow_control.servoing_module import ServoingModule
try:
    from robot_io.input_devices.space_mouse import SpaceMouse
except ImportError:
    pass


def evaluate_control(env, recording, episode_num, start_index=0,
                     control_config=None, max_steps=1000,
                     plot=True):
    """
    Function that runs the policy.
    """
    assert env is not None
    # load the servo module
    # TODO(max): rename base_frame to start_frame
    servo_module = ServoingModule(recording,
                                  episode_num=episode_num,
                                  start_index=start_index,
                                  control_config=control_config,
                                  camera_calibration=env.camera_calibration,
                                  plot=plot,
                                  save_dir="./plots_vacuum")

    servo_action = None
    done = False
    for counter in range(max_steps):
        # Compute controls (reverse order)
        action = [0, 0, 0, 0, 1]
        if servo_module.base_frame == servo_module.max_demo_frame or done:
            # for end move up if episode is done
            action = [0, 0, 1, 0, 0]
        elif counter > 0:
            action = servo_action
        elif counter == 0:
            # inital frame dosent have action
            pass
        else:
            pass

        # Environment Stepping
        state, reward, done, info = env.step(action)

        # take only the three spatial components
        ee_pos = info['robot_state_full'][:6]
        obs_image = info['rgb_unscaled']
        servo_action, _, _, info = servo_module.step(obs_image, ee_pos,
                                                     live_depth=info['depth'])

        if "action_abs_tcp" in info:
            value = tuple(info["action_abs_tcp"])
            # move up a bit first
            env.robot.send_cartesian_coords_rel_PTP((0, 0, .025, 0, 0, 0))
            time.sleep(2)
            env.robot.move_to_pose(value)

    if 'ep_length' not in info:
        info['ep_length'] = counter

    return state, reward, done, info


def go_to_default_pose():
    import cv2
    iiwa_env = IIWAEnv(act_type='continuous', freq=20,
                       obs_type='image_state_reduced',
                       dv=0.0035, drot=0.025, use_impedance=True, max_steps=1e9,
                       reset_pose=(0, -0.56, 0.23, math.pi, 0, math.pi / 2), control='absolute')
    _ = iiwa_env.reset()

    # load the first image from demo
    recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/sick_combine", 3
    img_fn = os.path.join(recording, "episode_{}/img_0000.png".format(episode_num))
    print(os.path.isfile(img_fn))
    print(img_fn)
    demo_img = cv2.imread(img_fn)

    cv2.imshow("demo", demo_img)

    while True:
        action = [0, 0, 0, 0, 1]
        _state, _reward, _done, info = iiwa_env.step(action)

        # ee_pos = action_queue['robot_state_full'][:6]
        obs_image = info['rgb_unscaled']
        cv2.imshow("win", obs_image[:, :, ::-1])
        cv2.waitKey(1)


def main():
    """
    The main function that loads the recording, then runs policy.
    """
    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/shape_insert", 15
    # start_index = 107
    # threshold = 0.1

    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/lego", 3
    # start_index = 100
    # threshold = 0.30

    # demo mit knacks, ok
    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/wheel", 9
    # start_index = 4
    # loss = 0.25

    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/wheel", 17
    # start_index = 1

    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/pick_stow", 2
    # start_index = 5

    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/transfer_orange", 0
    # start_index = 5

    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/navigate_blue_letter_block", 0
    # start_index = 1

    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/sick_vacuum", 4
    # start_index = 1

    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/car_block_1", 0
    # start_index = 1

    recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/vacuum", 2
    start_index = 2

    threshold = 0.35  # this was 0.35
    plot = True

    control_config = dict(mode="pointcloud",
                          gain_xy=50,
                          gain_z=100,
                          gain_r=15,
                          threshold=threshold,
                          use_keyframes=False,
                          cursor_control=True)

    iiwa_env = IIWAEnv(act_type='continuous', freq=20,
                       obs_type='image_state_reduced',
                       dv=0.0035, drot=0.025, use_impedance=True, max_steps=1e9,
                       reset_pose=(0, -0.56, 0.23, math.pi, 0, math.pi / 2), control='relative',
                       gripper_opening_width=109)

    # TOOD(max): add a check here that makes shure that the pointcloud mode matches the iiwa mode
    iiwa_env.reset()

    state, reward, done, info = evaluate_control(iiwa_env,
                                                 recording,
                                                 episode_num=episode_num,
                                                 start_index=start_index,
                                                 control_config=control_config,
                                                 plot=plot)
    print(state)
    print(reward)
    print(done)
    print(info)


if __name__ == "__main__":
    #go_to_default_pose()
    main()
