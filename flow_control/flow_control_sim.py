"""
Testing file for development, to experiment with evironments.
"""
import logging
from pdb import set_trace
import numpy as np
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.servoing_module import ServoingModule


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
        # environment stepping
        state, reward, done, info = env.step(servo_action)

        # compute action
        if isinstance(env, RobotSimEnv):
            # TODO(max): fix API change between sim and robot
            obs_image = state
        else:
            obs_image = info['rgb_unscaled']
        ee_pos = info['robot_state_full'][:8]  # take three position values
        servo_res = servo_module.step(obs_image, ee_pos, live_depth=info['depth'])
        servo_action, servo_done, servo_info = servo_res

        do_abs = False
        if do_abs and servo_info:
            print("Servo info", servo_info)
            # if "grip" in servo_info:
            #    env.robot.apply_action([0, 0, 0, 0, servo_info["grip"]])
            #    for i in range(3):
            #        env.p.stepSimulation(physicsClientId=env.cid)
            #    set_trace()

            if "rel" in servo_info:
                control = env.robot.get_control("absolute")  # xyzrg by default
                c_pos, c_orn = env.p.getLinkState(env.robot.robot_uid, env.robot.flange_index, physicsClientId=env.cid)[0:2]
                new_pos = np.array(c_pos) + servo_info["rel"][0:3]
                rot = env.robot.desired_ee_angle
                grp = servo_action[-1]
                abs_action = [*new_pos, rot, grp]
                env.robot.apply_action(abs_action, control)

                for i in range(12):
                    env.p.stepSimulation(physicsClientId=env.cid)
                    mr = env.robot.get_motion_residual()
                    if i > 12 and mr < .002:
                        print("done in ", i)
                        break
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
                          threshold=0.45)  # .15 35 45

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
