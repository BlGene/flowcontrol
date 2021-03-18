"""
Testing file for development, to experiment with evironments.
"""
import time
import logging
from pdb import set_trace
import numpy as np
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.servoing_module import ServoingModule


def evaluate_control(env, recording, episode_num, start_index=0,
                     control_config=None, max_steps=1000, plot=True):
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
        if done:
            break

        # compute action
        if isinstance(env, RobotSimEnv):
            # TODO(max): fix API change between sim and robot
            obs_image = state
        else:
            obs_image = info['rgb_unscaled']
        ee_pos = info['robot_state_full'][:8]  # take three position values
        servo_res = servo_module.step(obs_image, ee_pos, live_depth=info['depth'])
        servo_action, servo_done, servo_info = servo_res

        # env.robot.show_action_debug()
        do_abs = True
        if not (do_abs and servo_info):
            continue

        # time.sleep(.5)

        if "grip" in servo_info:
            control = env.robot.get_control("absolute")
            pos = env.robot.desired_ee_pos
            rot = env.robot.desired_ee_angle
            grp = servo_info["grip"]
            abs_action = [*pos, rot, grp]

            for i in range(24):
                env.robot.apply_action(abs_action, control)
                env.p.stepSimulation(physicsClientId=env.cid)
            print("grip", servo_info["grip"], "done in", i)

        if "abs" in servo_info:
            control = env.robot.get_control("absolute")
            aim_tcp = servo_info["abs"][0:3]
            if control.frame == "tcp":
                new_pos = aim_tcp
            elif control.frame == "flange":
                c_pos, c_orn = env.p.getLinkState(env.robot.robot_uid, control.ik_index,
                                                  physicsClientId=env.cid)[0:2]
                cur_tcp = env.robot.get_tcp_pos()
                new_pos = np.array(c_pos) - cur_tcp + aim_tcp

            rot = env.robot.desired_ee_angle
            grp = servo_action[-1]
            abs_action = [*new_pos, rot, grp]
            env.robot.apply_action(abs_action, control)

            for i in range(300):
                env.p.stepSimulation(physicsClientId=env.cid)
                mr = env.robot.get_motion_residual()
                if i > 12 and mr < .002:
                    break
            print("abs", np.array(servo_info["abs"][0:3]).round(3), "done in", i)

        if "rel" in servo_info:
            control = env.robot.get_control("absolute")  # xyzrg by default
            c_pos, c_orn = env.p.getLinkState(env.robot.robot_uid, control.ik_index,
                                              physicsClientId=env.cid)[0:2]
            new_pos = np.array(c_pos) + servo_info["rel"][0:3]
            rot = env.robot.desired_ee_angle
            grp = servo_action[-1]
            abs_action = [*new_pos, rot, grp]
            env.robot.apply_action(abs_action, control)

            for i in range(300):
                env.p.stepSimulation(physicsClientId=env.cid)
                mr = env.robot.get_motion_residual()
                if i > 12 and mr < .002:
                    break
            print("rel", np.array(servo_info["rel"][0:3]).round(3), "done in", i)

        #
        if servo_module.demo.frame == 53:
            print(servo_info.keys(), "X"*20)
            set_trace()

    if servo_module.view_plots:
        del servo_module.view_plots
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
                          threshold=0.40)  # .15 35 45

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
    print("reward:", reward, "\n")


if __name__ == "__main__":
    main()
