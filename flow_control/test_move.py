"""
Test functional beahviour through built-in policies.
"""
import os
import time
import logging
import unittest
from pdb import set_trace
import numpy as np
from scipy.spatial.transform import Rotation as R

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.servoing.module import ServoingModule, T_TCP_CAM
from gym_grasping.utils import state2matrix
from flow_control.rgbd_camera import RGBDCamera

is_ci = "CI" in os.environ

if is_ci:
    obs_type = "state"
    renderer = "tiny"
else:
    obs_type = "image"
    renderer = "debug"


def make_demo_dict(env, base_state, base_info, base_action):
    """
    create keep dict with info from state
    """
    env_info = env.get_info()
    rgb = base_state[np.newaxis, :]
    depth = base_info["depth"][np.newaxis, :]
    mask = base_info["seg_mask"][np.newaxis, :] == 2
    state = base_info["robot_state_full"][np.newaxis, :]
    keep_dict = {0: None}

    actions = np.array(base_action)[np.newaxis, :]

    demo_dict = dict(env_info=env_info,
                     rgb=rgb,
                     depth=depth,
                     mask=mask,
                     state=state,
                     keep_dict=keep_dict,
                     actions=actions)
    return demo_dict


class Move_absolute(unittest.TestCase):
    """
    Test a Pick-n-Place task.
    """

    def test_gripper(self):
        """test performance of scripted policy, with parallel gripper"""
        env = RobotSimEnv(task="flow_calib", robot="kuka",
                          obs_type=obs_type, renderer=renderer,
                          act_type='continuous', control="absolute",
                          max_steps=600, initial_pose="close",
                          img_size=(256, 256))

        base_state, _, _, base_info = env.step(None)
        base_tcp_pose = env.robot.get_tcp_pose()
        world_cam = env.camera.get_cam_mat()

        cam_tcp = np.linalg.inv(world_cam) @ base_tcp_pose

        tcp_angles = env.robot.get_tcp_angles()
        base_action = [*base_tcp_pose[:3, 3], tcp_angles[2], 1]

        delta = 0.04
        data = []
        for i in (0, 1, 2):
            for j in (1, -1):
                target_pose = list(base_tcp_pose[:3, 3])
                #target_pose[i] += j * delta
                target_pose += base_tcp_pose[:3, i] * j * delta
                control = env.robot.get_control("absolute")  #, min_iter=24)
                action = [*target_pose, tcp_angles[2], 1]

                state2, _, _, info = env.step(action, control)
                tcp_pose = env.robot.get_tcp_pose()
                cam_pose = env.camera.get_cam_mat()
                data.append(dict(action=action, state=state2, info=info,
                                 pose=tcp_pose,cam=cam_pose))

        demo_dict = make_demo_dict(env, base_state, base_info, base_action)
        control_config = dict(mode="pointcloud-abs",
                              gain_xy=50,
                              gain_z=100,
                              gain_r=15,
                              threshold=0.40)

        servo_module = ServoingModule(demo_dict,
                                      episode_num=0,
                                      start_index=0,
                                      control_config=control_config,
                                      camera_calibration=env.camera.calibration,
                                      plot=True, save_dir=None)
        print("loaded servo module")

        for i in range(len(data)):
            rgb = data[i]["state"]
            state = data[i]["info"]["robot_state_full"]
            depth = data[i]["info"]["depth"]
            action, done, info = servo_module.step(rgb, state, depth)
            align_trf = action[0]
            T_gt = data[i]["pose"] @ np.linalg.inv(base_tcp_pose)

            est_t = np.linalg.inv(cam_tcp) @ align_trf @ cam_tcp
            est_w = world_cam @ align_trf @ np.linalg.inv(world_cam)
            #np.linalg.inv(data[i]["cam"])
            #set_trace()
            print("T_a\n", align_trf.round(3))
            print("gt\n", T_gt.round(3))
            print()
            print("est\n", est_w.round(3))

            #set_trace()


class Camera_Matrix(unittest.TestCase):
    """
    Test a Pick-n-Place task.
    """

    def test_camera(self):
        return
        """test performance of scripted policy, with parallel gripper"""
        env = RobotSimEnv(task="flow_calib", robot="kuka",
                          obs_type=obs_type, renderer=renderer,
                          act_type='continuous', control="absolute",
                          max_steps=600, initial_pose="close",
                          img_size=(256, 256))

        base_state, _, _, base_info = env.step(None)
        base_tcp_pose = env.robot.get_tcp_pose()
        world_cam = env.camera.get_cam_mat()

        # tcp in camera coordinates
        cam_tcp = np.linalg.inv(world_cam) @ base_tcp_pose
        print("world_cam\n", world_cam.round(3))
        print("tcp_pose\n", base_tcp_pose.round(3))
        print("cam_tcp\n", cam_tcp.round(3))

        world_tcp =  world_cam @ cam_tcp  # tcp in world coordinates (from camera)

        # select leftmost pixel on the image
        obj_mask = base_info["seg_mask"] == 2
        min_y = np.where(np.any(obj_mask, axis=0))[0][0]
        min_x = np.where(obj_mask[:,min_y])[0][0]

        """
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(obj_mask)
        mask_img = np.zeros(obj_mask.shape,dtype=bool)
        mask_img[min_y, min_x] = True
        ax[1].imshow(mask_img)
        circle2 = plt.Circle(( min_x, min_y), 1.5, color='r', fill=False)
        ax[0].add_patch(circle2)
        plt.show()
        """

        mask = np.array([[min_x, min_y]])
        cam = RGBDCamera(env.camera.calibration)
        cam_point = cam.generate_pointcloud(base_state, base_info["depth"], mask)[0, :4]
        world_point = world_cam @ cam_point  # point in world cooridates

        # red line origin -> tcp
        env.p.addUserDebugLine([0, 0, 0], base_tcp_pose[:3, 3], lineColorRGB=[1, 0, 0],
                               lineWidth=2, physicsClientId=0)
        # green line origin -> camera
        env.p.addUserDebugLine([0, 0, 0], world_cam[:3, 3], lineColorRGB=[0, 1, 0],
                               lineWidth=2, physicsClientId=0)

        # blue line camera -> tcp
        env.p.addUserDebugLine(world_cam[:3, 3], world_tcp[:3, 3], lineColorRGB=[0, 0, 1],
                               lineWidth=2, physicsClientId=0)

        # teal line camera -> corner
        env.p.addUserDebugLine(world_cam[:3, 3], world_point[:3], lineColorRGB=[0, 1, 1],
                               lineWidth=2, physicsClientId=0)

        set_trace()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="")
    unittest.main()
