"""
Test flow control in simulation, this is a bit deprecated.
"""
from collections import defaultdict
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib
matplotlib.use('TkAgg')
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from gym_grasping.flow_control.servoing_module import ServoingModule
try:
    from robot_io.input_devices.space_mouse import SpaceMouse
except ImportError:
    pass

GRIPPER_OPEN = 1


def dcm2cntrl(T, gripper=GRIPPER_OPEN):
    '''convert a transformation matrix into a robot control signal'''
    servo_dcm = R.from_dcm(T[:3, :3])
    pos_x, pos_y, pos_z = T[:3, 3]
    roll, pitch, yaw = servo_dcm.as_euler('xyz')
    action = [pos_x, pos_y, pos_z, gripper, yaw, pitch, roll]
    return action


def evaluate_control(env, recording, servo_module, max_steps=600, mouse=False):
    '''evaluate a recording'''
    if mouse:
        mouse = SpaceMouse(act_type='continuous')

    servo_action = None  # will be set in loop
    done = False
    for counter in range(max_steps):
        # Compute controls (reverse order)
        action = [0, 0, 0, 0, 1]
        if mouse:
            action = mouse.handle_mouse_events()
            mouse.clear_events()
        elif servo_module.base_frame == servo_module.max_demo_frame:
            # for end move up if episode is done
            action = [0, 0, 1, 0, 0]
        elif counter > 0:
            action = servo_action
        elif counter == 0:
            # inital frame dosent have servo action
            pass
        else:
            pass

        # Environment Stepping
        state, reward, done, info = env.step(action)
        if isinstance(state, dict):
            ee_pos = info['robot_state_full'][:3]
            state_image = state['img']
        else:
            # state extraction
            link_state = env.p.getLinkState(env.robot.robot_uid,
                                            env.robot.flange_index)
            ee_pos = list(link_state[0])
            ee_pos[2] += 0.02
            state_image = state

        servo_action, servo_trf, servo_done = \
            servo_module.step(state_image, ee_pos, live_depth=info["depth"])
        # this produces a transformation in the TCP frame (or camera).
        assert servo_module.frame == "TCP"
        action = dcm2cntrl(servo_trf)

        if servo_done:
            done = True

        # logging
        # state_dict = dict(state=state,
        #                  reward=reward,
        #                  done=done,
        #                  ee_pos=ee_pos)

        if done:
            servo_module.reset()
            env.reset()
            print("done. ", reward, counter)
            break

    if 'ep_length' not in info:
        info['ep_length'] = counter

    return dict(reward=reward, counter=counter)


def save_imitation_trajectory(save_id, collect):
    '''save a imitated trajectory'''
    assert isinstance(collect[0], dict)

    episode = defaultdict(list)

    for key in collect[0]:
        for step in collect:
            episode[key].append(step[key])

        episode[key] = np.array(episode[key])

    save_fn = f"./eval_t30/run_{save_id:03}.npz"
    np.savez(save_fn, **episode)


def test_stack_wo_textures():
    '''
    test stacking without textures
    '''
    task_name = "stack"
    recording = "stack_recordings/episode_118"
    episode_num = 1

    start_index = 20
    max_steps = 600
    img_size = (256, 256)

    control_config = dict(mode="pointcloud",
                          gain_xy=100,
                          gain_z=50,
                          gain_r=30,
                          threshold=0.20,
                          use_keyframes=False,
                          cursor_control=True)

    control_config = dict(mode="pointcloud-abs",
                          gain_xy=.5,
                          gain_z=1,
                          gain_r=1,
                          threshold=0.005,
                          use_keyframes=False,
                          cursor_control=False)

    control_config = dict(mode="flat",
                          gain_xy=50,
                          gain_z=30,
                          gain_r=-7,
                          threshold=0.20,
                          use_keyframes=True)

    env = RobotSimEnv(task=task_name, renderer='tiny', act_type='continuous',
                      max_steps=max_steps, img_size=img_size)

    if "cursor_control" in control_config \
       and not control_config["cursor_control"]:
        env.robot.dv = None
        env.robot._cursor_control = False

    servo_module = ServoingModule(recording, episode_num=episode_num,
                                  start_index=start_index,
                                  control_config=control_config,
                                  camera_calibration=env.camera_calibration,
                                  plot=True)

    num_samples = 10
    collect = []
    for i in range(num_samples):
        print("starting", i, "/", num_samples)
        res = evaluate_control(env, recording, servo_module,
                               max_steps=max_steps)
        collect.append(res)
        env.reset()
        servo_module.reset()


def test_stack_w_textures():
    '''test stacking with textures'''
    task_name = "stack"
    recording = "/media/kuka/Seagate Expansion Drive/kuka_recordings/" \
                "flow/stacking_sim/"
    episode_num = 0
    start_index = 0
    # threshold = .2  # .40 for not fitting_control
    max_steps = 2000
    img_size = (256, 256)
    env = RobotSimEnv(task=task_name, renderer='tiny', act_type='continuous',
                      max_steps=600, img_size=img_size)

    servo_module = ServoingModule(recording, episode_num=episode_num,
                                  start_index=start_index,
                                  camera_calibration=env.camera_calibration,
                                  plot=True)

    num_samples = 2
    collect = []
    for i in range(num_samples):
        print("starting", i, "/", num_samples)
        res = evaluate_control(env, recording, servo_module,
                               max_steps=max_steps)
        collect.append(res)
        env.reset()
        servo_module.reset()


if __name__ == "__main__":
    test_stack_wo_textures()
    # test_stack_w_textures()
