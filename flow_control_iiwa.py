"""
Testing file for development, to experiment with evironments.
"""
import os
import math
from gym_grasping.envs.iiwa_env import IIWAEnv
from gym_grasping.flow_control.servoing_module import ServoingModule
try:
    from robot_io.input_devices.space_mouse import SpaceMouse
except ImportError:
    pass


def evaluate_control(env, recording, episode_num, base_index=0,
                     control_config=None, max_steps=1000, use_mouse=False,
                     plot=True):
    """
    Function that runs the policy.
    """
    # load the servo module
    # TODO(max): rename base_frame to start_frame
    servo_module = ServoingModule(recording,
                                  episode_num=episode_num,
                                  start_index=base_index,
                                  control_config=control_config,
                                  camera_calibration=env.camera_calibration,
                                  plot=plot)

    # load env (needs
    if env is None:
        raise ValueError

    if use_mouse:
        mouse = SpaceMouse(act_type='continuous')

    servo_action = None
    done = False
    for counter in range(max_steps):
        # Compute controls (reverse order)
        action = [0, 0, 0, 0, 1]
        if use_mouse:
            action = mouse.handle_mouse_events()
            mouse.clear_events()
        elif servo_module.base_frame == servo_module.max_demo_frame or done:
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
        # if done:
        #     print("done. ", reward)
        #     break
        #
        # take only the three spatial components
        ee_pos = info['robot_state_full'][:6]
        obs_image = info['rgb_unscaled']
        servo_action, _, _ = servo_module.step(obs_image, ee_pos,
                                               live_depth=info['depth'])
        # if mode == "manual":
        #     use_mouse = True
        # else:
        #     use_mouse = False

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

        # ee_pos = info['robot_state_full'][:6]
        obs_image = info['rgb_unscaled']
        cv2.imshow("win", obs_image[:, :, ::-1])
        cv2.waitKey(1)


def main():
    """
    The main function that loads the recording, then runs policy.
    """
    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/shape_insert", 15
    # base_index = 107
    # threshold = 0.1

    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/lego", 3
    # base_index = 100
    # threshold = 0.30

    # demo mit knacks, ok
    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/wheel", 9
    # base_index = 4
    # loss = 0.25
    #

    recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/wheel", 17
    base_index = 1

    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/pick_stow", 2
    # base_index = 5

    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/transfer_orange", 0
    # base_index = 5

    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/navigate_blue_letter_block", 0
    # base_index = 1

    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/sick_vacuum", 4
    # base_index = 1

    recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/car_block_1", 0
    base_index = 1

    threshold = 0.35  # this was 0.35

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
                       reset_pose=(0, -0.56, 0.23, math.pi, 0, math.pi / 2), control='relative')

    # TOOD(max): add a check here that makes shure that the pointcloud mode matches the iiwa mode

    iiwa_env.reset()

    plot = True
    state, reward, done, info = evaluate_control(iiwa_env,
                                                 recording,
                                                 episode_num=episode_num,
                                                 base_index=base_index,
                                                 control_config=control_config,
                                                 plot=plot,
                                                 use_mouse=False)
    print(state)
    print(reward)
    print(done)
    print(info)


if __name__ == "__main__":
    # go_to_default_pose()
    main()
