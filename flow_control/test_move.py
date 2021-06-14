"""
Test functional beahviour through built-in policies.
"""
import os
import time
import math
import logging
import unittest
from pdb import set_trace
import numpy as np
from scipy.spatial.transform import Rotation as R

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.servoing.module import ServoingModule, T_TCP_CAM
from gym_grasping.utils import state2matrix
from flow_control.rgbd_camera import RGBDCamera
import sys

is_ci = "CI" in os.environ

if is_ci:
    obs_type = "state"
    renderer = "tiny"
else:
    obs_type = "image"
    renderer = "debug"


def get_pose_diff(T_a, T_b):
    diff_pos = np.linalg.norm(T_a[:3, 3]-T_b[:3, 3], 2)
    diff_rot = R.from_matrix(T_a[:3, :3] @ np.linalg.inv(T_b[:3, :3])).magnitude()
    return diff_pos, diff_rot


def make_demo_dict(env, base_state, base_info, base_action):
    """
    create keep dict with info from state
    """
    demo_dict = dict(env_info=env.get_info(),
                     rgb=base_state[np.newaxis, :],
                     depth=base_info["depth"][np.newaxis, :],
                     mask=base_info["seg_mask"][np.newaxis, :] == 2,
                     state=base_info["robot_state_full"][np.newaxis, :],
                     keep_dict={0: None},
                     actions=np.array(base_action)[np.newaxis, :])
    return demo_dict


def get_target_poses(env, tcp_base):
    control = env.robot.get_control("absolute")  #, min_iter=24)

    delta = 0.04
    for i in (0, 1, 2):
        for j in (1, -1):
            target_pose = list(tcp_base[:3, 3])
            target_pose[i] += j * delta
            yield target_pose, control

"""
from gym_grasping.calibration.random_pose_sampler import RandomPoseSampler
def get_target_poses(env, tcp_base):
    control_in = dict(type='continuous', dof='xyzquatg', frame='tcp',
                      space="world", norm=False,
                      min_iter=12, max_iter=300, to_dist=.002)
    control = env.robot.get_control(control_in)  #, min_iter=24)

    sampler = RandomPoseSampler()

    for i in range(10):
        pose = sampler.sample_pose()
        yield pose, control
"""

def show_pointclouds(servo_module, rgb, depth, cam_live, cam_base):
    import open3d as o3d
    start_pc = servo_module.cam.generate_pointcloud2(rgb, depth)
    colors = start_pc[:, 4:7]/255.  # recorded image colors
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(start_pc[:, :3])
    pcd1.colors = o3d.utility.Vector3dVector(colors)
    pcd1.transform(cam_live)

    end_pc = servo_module.cam.generate_pointcloud2(servo_module.demo.rgb, servo_module.demo.depth)
    colors = end_pc[:, 4:7]/255.  # recorded image colors
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(end_pc[:, :3])
    pcd2.colors = o3d.utility.Vector3dVector(colors)
    pcd2.transform(cam_base)
    o3d.visualization.draw_geometries([pcd1, pcd2])


class MoveThenServo(unittest.TestCase):
    """
    Test a Pick-n-Place task.
    """
    def test_absolute(self):
        """test performance of scripted policy, with parallel gripper"""
        env = RobotSimEnv(task="flow_calib", robot="kuka",
                          obs_type=obs_type, renderer=renderer,
                          act_type='continuous', control="absolute",
                          max_steps=600, initial_pose="close",
                          img_size=(256, 256))

        # record base frame
        base_state, _, _, base_info = env.step(None)
        tcp_base = env.robot.get_tcp_pose()
        tcp_angles = env.robot.get_tcp_angles()
        # relative transformations
        cam_base = env.camera.get_cam_mat()
        cam_tcp = np.linalg.inv(cam_base) @ tcp_base

        data = []

        target_pose, control = next(get_target_poses(env, tcp_base))

        # go to state
        action = [*target_pose, tcp_angles[2], 1]
        state2, _, _, info = env.step(action, control)
        # and collect data
        tcp_pose = env.robot.get_tcp_pose()
        cam_pose = env.camera.get_cam_mat()
        data.append(dict(action=action, state=state2, info=info,
                         pose=tcp_pose,cam=cam_pose))

        # initialize servo module
        base_action = [*tcp_base[:3, 3], tcp_angles[2], 1]
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

        max_steps = 1000
        servo_action = None
        servo_control = None  # means default
        done = False
        for counter in range(max_steps):
            # environment stepping
            if done:
                break

            state, reward, done, info = env.step(servo_action, servo_control)
            # compute action
            if isinstance(env, RobotSimEnv):
                obs_image = state  # TODO(max): fix API change between sim and robot
            else:
                obs_image = info['rgb_unscaled']
            ee_pos = info['robot_state_full'][:8]  # take three position values
            servo_res = servo_module.step(obs_image, ee_pos, live_depth=info['depth'])
            servo_action, servo_done, servo_queue = servo_res

            if servo_module.config.mode == "pointcloud-abs":
                trf_est, grip_action  = servo_action
                robot_pose = env.robot.get_tcp_pose()
                goal_pos = cam_base @ trf_est @ np.linalg.inv(cam_base) @ tcp_base
                goal_pos = goal_pos[:3, 3]
                print(goal_pos)

                #env.p.removeAllUserDebugItems()
                #env.p.addUserDebugLine([0, 0, 0], robot_pose[:3, 3], lineColorRGB=[1, 0, 0],
                #                       lineWidth=2, physicsClientId=0)
                #env.p.addUserDebugLine([0, 0, 0], goal_pos, lineColorRGB=[0, 1, 0],
                #                       lineWidth=2, physicsClientId=0)

                goal_angle = math.pi / 4
                servo_action = goal_pos.tolist() + [goal_angle, grip_action]
                servo_control = env.robot.get_control("absolute")
                #set_trace()


class MoveThenEstimate(unittest.TestCase):
    def test_absolute(self):
        return
        """test performance of scripted policy, with parallel gripper"""
        env = RobotSimEnv(task="flow_calib", robot="kuka",
                          obs_type=obs_type, renderer=renderer,
                          act_type='continuous', control="absolute",
                          max_steps=600, initial_pose="close",
                          img_size=(256, 256),
                          param_randomize=False)

        # record base frame
        base_state, _, _, base_info = env.step(None)
        tcp_base = env.robot.get_tcp_pose()
        tcp_angles = env.robot.get_tcp_angles()
        # relative transformations
        cam_base = env.camera.get_cam_mat()

        T_cam_tcp = cam_base @ np.linalg.inv(tcp_base)

        live = []
        errors = []
        for target_pose, control in get_target_poses(env, tcp_base):
            # go to state
            action = [*target_pose, tcp_angles[2], 1]
            state2, _, _, info = env.step(action, control)
            # and collect data
            tcp_live = env.robot.get_tcp_pose()
            cam_live = env.camera.get_cam_mat()
            live.append(dict(action=action, state=state2, info=info,
                             pose=tcp_live,cam=cam_live))

            T_cam_tcp2 = cam_live @ np.linalg.inv(tcp_live)
            diff = T_cam_tcp2 @ np.linalg.inv(T_cam_tcp)
            err = np.linalg.norm(diff[:3, 3])
            errors.append(err)

        print("mean erorr", np.mean(errors))

        # initialize servo module
        base_action = [*tcp_base[:3, 3], tcp_angles[2], 1]
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
        # iterate over samples
        pos_errs = []
        rot_errs = []
        for i in range(len(live)):
            rgb = live[i]["state"]
            state = live[i]["info"]["robot_state_full"]
            depth = live[i]["info"]["depth"]
            cam_live = live[i]["cam"]
            action, done, info = servo_module.step(rgb, state, depth)

            T_align = action[0]
            # comparison in cam frame
            # T_gt = live[i]["cam"] @ np.linalg.inv(cam_base)
            # cam_live = live[i]["cam"]
            # T_est = cam_live @ T_align @ np.linalg.inv(cam_live)  # in world

            # create pointcloud
            #show_pointclouds(servo_module, rgb, depth, live[i]["cam"], cam_base)

            # comparison in tcp frame
            T_gt = live[i]["pose"] @ np.linalg.inv(tcp_base)
            tcp_live = live[i]["pose"]
            T_est = tcp_live @ T_align @ np.linalg.inv(tcp_live)  # in world

            diff_pos, diff_rot = get_pose_diff(T_gt, T_est)
            self.assertLess(diff_pos, .005)
            self.assertLess(diff_rot, .008)
            pos_errs.append(diff_pos)
            rot_errs.append(diff_rot)

        print("mean pos error", np.mean(pos_errs))
        print("mean rot error", np.mean(rot_errs))

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
        tcp_base = env.robot.get_tcp_pose()
        cam_base = env.camera.get_cam_mat()

        # tcp in camera coordinates
        cam_tcp = np.linalg.inv(cam_base) @ tcp_base
        print("cam_base\n", cam_base.round(3))
        print("tcp_pose\n", tcp_base.round(3))
        print("cam_tcp\n", cam_tcp.round(3))

        world_tcp =  cam_base @ cam_tcp  # tcp in world coordinates (from camera)

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
        world_point = cam_base @ cam_point  # point in world cooridates

        # red line origin -> tcp
        env.p.addUserDebugLine([0, 0, 0], tcp_base[:3, 3], lineColorRGB=[1, 0, 0],
                               lineWidth=2, physicsClientId=0)
        # green line origin -> camera
        env.p.addUserDebugLine([0, 0, 0], cam_base[:3, 3], lineColorRGB=[0, 1, 0],
                               lineWidth=2, physicsClientId=0)

        # blue line camera -> tcp
        env.p.addUserDebugLine(cam_base[:3, 3], world_tcp[:3, 3], lineColorRGB=[0, 0, 1],
                               lineWidth=2, physicsClientId=0)

        # teal line camera -> corner
        env.p.addUserDebugLine(cam_base[:3, 3], world_point[:3], lineColorRGB=[0, 1, 1],
                               lineWidth=2, physicsClientId=0)

        set_trace()

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="")
    unittest.main()
