"""
Testing file for development, to experiment with evironments.
"""
from gym_grasping.robot_envs.iiwa_env import IIWAEnv
from gym_grasping.flow_control.servoing_module import ServoingModule

folder_format = "LUKAS"

def evaluate_control(env, recording, episode_num, base_index=0, threshold=0.4, max_steps=1000, use_mouse=False, plot=True):
    # load the servo module
    #TODO(max): rename base_frame to start_frame
    servo_module = ServoingModule(recording,
                                  episode_num=episode_num,
                                  start_index=base_index,
                                  threshold=threshold,
                                  camera_calibration=env.camera_calibration,
                                  plot=plot)

    # load env (needs
    if env is None:
        raise ValueError

    if use_mouse:
        from gym_grasping.robot_io.space_mouse import SpaceMouse
        mouse = SpaceMouse(act_type='continuous')

    done = False
    for counter in range(max_steps):
        # Compute controls (reverse order)
        action = [0, 0, 0, 0, 1]
        if use_mouse:
            action = mouse.handle_mouse_events()
            mouse.clear_events()
        elif servo_module.base_frame == servo_module.max_demo_frame or done:
            # for end move up if episode is done
            action = [0,0,1,0,0]
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
        ee_pos = info['robot_state_full'][:3]
        obs_image = info['rgb_unscaled']
        servo_action = servo_module.step(obs_image, ee_pos, live_depth=info['depth'])
        # if mode == "manual":
        #     use_mouse = True
        # else:
        #     use_mouse = False

    if 'ep_length' not in info:
        info['ep_length'] = counter
    return state, reward, done, info


if __name__ == "__main__":
    #TODO(max): add variables here: expisode num, start frame.
    task_name = "stack"
    recording = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/stacking"
    episode_num = 3
    base_index = 0

    #recording = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/control_test"
    #episode_num = 1
    #base_index = 0

    recording, episode_num = "/media/kuka/Seagate Expansion Drive/kuka_recordings/flow/shape_insert", 10
    base_index = 200

    threshold = 0.18  # .40 for not fitting_control

    iiwa_env = IIWAEnv(act_type='continuous', freq=20, obs_type='img_state_reduced',
                       dv=0.0035, drot=0.025, use_impedance=True,
                       use_real2sim=False, max_steps=1e9)
    iiwa_env.reset()

    save = False
    plot = True

    state, reward, done, info = evaluate_control(iiwa_env,
                                                 recording,
                                                 episode_num=episode_num,
                                                 base_index=base_index,
                                                 threshold=threshold,
                                                 plot=plot,
                                                 use_mouse=False)
