"""
Testing file for development, to experiment with evironments.
"""
from gym_grasping.robot_envs.iiwa_env import IIWAEnv
from gym_grasping.flow_control.servoing_module import ServoingModule
import math

folder_format = "LUKAS"


def evaluate_control(env, recording, episode_num, base_index=0,
                     control_config=None, max_steps=1000, use_mouse=False,
                     plot=True):
    # load the servo module
    #TODO(max): rename base_frame to start_frame
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
        from gym_grasping.robot_io.space_mouse import SpaceMouse
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


if __name__ == "__main__":
    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/shape_insert", 15
    # base_index = 107
    # threshold = 0.1

    #recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/lego", 3
    #base_index = 100
    #threshold = 0.30


    # demo mit knacks, ok
    #recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/wheel", 9
    #base_index = 4
    #loss = 0.25
    #
    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/wheel", 14
    # base_index = 5

    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/pick_stow", 2
    # base_index = 5

    # recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/transfer_orange", 0
    # base_index = 5

    recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/navigate_blue_letter_block", 0
    base_index = 1

    threshold = 0.35

    control_config = dict(mode="pointcloud-abs",
                          gain_xy=50,
                          gain_z=50,
                          gain_r=15,
                          threshold=threshold,
                          use_keyframes=False,
                          cursor_control=True)

    iiwa_env = IIWAEnv(act_type='continuous', freq=20,
                       obs_type='img_state_reduced',
                       dv=0.0035, drot=0.025, use_impedance=True,
                       use_real2sim=False, max_steps=1e9,
                       rest_pose=(0, -0.56, 0.23, math.pi, 0, math.pi / 2), control='absolute')
    iiwa_env.reset()

    save = False
    plot = True

    state, reward, done, info = evaluate_control(iiwa_env,
                                                 recording,
                                                 episode_num=episode_num,
                                                 base_index=base_index,
                                                 control_config=control_config,
                                                 plot=plot,
                                                 use_mouse=False)
