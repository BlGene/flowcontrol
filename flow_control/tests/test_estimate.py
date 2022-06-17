"""
Test servoing by estimating relative poses.

This means we record a base image, move to a target pose, and testimage the
pose difference.
"""
import os
import math
import logging
import unittest
import numpy as np
from scipy.spatial.transform import Rotation as R

from gym_grasping.envs.robot_sim_env import RobotSimEnv
from robot_io.recorder.simple_recorder import SimpleRecorder, PlaybackEnvServo

from flow_control.servoing.module import ServoingModule
from flow_control.utils_coords import get_pos_orn_diff, print_pose_diff
from flow_control.utils_coords import permute_pose_grid




import copy
from pdb import set_trace

IS_CI = "CI" in os.environ

if IS_CI:
    RENDERER = "tiny"
else:
    RENDERER = "debug"





# from gym_grasping.calibration.random_pose_sampler import RandomPoseSampler
# def permute_pose_grid(env, tcp_base):
#    control_in = dict(type='continuous', dof='xyzquatg', frame='tcp',
#                      space="world", norm=False,
#                      min_iter=12, max_iter=300, to_dist=.002)
#    control = env.robot.get_control(control_in)  #, min_iter=24)
#    sampler = RandomPoseSampler()
#    for i in range(10):
#        pose = sampler.sample_pose()
#        yield pose, control


def show_pointclouds(servo_module, rgb, depth, cam_live, cam_base):
    """show pointclouds of fit."""
    import open3d as o3d
    start_pc = servo_module.demo_cam.generate_pointcloud2(rgb, depth)
    colors = start_pc[:, 4:7] / 255.  # recorded image colors
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(start_pc[:, :3])
    pcd1.colors = o3d.utility.Vector3dVector(colors)
    pcd1.transform(cam_live)

    end_pc = servo_module.demo_cam.generate_pointcloud2(servo_module.demo.rgb,
                                                        servo_module.demo.depth)
    colors = end_pc[:, 4:7] / 255.  # recorded image colors
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(end_pc[:, :3])
    pcd2.colors = o3d.utility.Vector3dVector(colors)
    pcd2.transform(cam_base)
    o3d.visualization.draw_geometries([pcd1, pcd2])


def move_absolute_then_estimate(env):
    """test performance of scripted policy, with parallel gripper"""
    # record base frame
    fg_mask = (env.get_obs_info()["seg_mask"] == 2)
    demo_pb = PlaybackEnvServo.freeze(env, fg_mask=fg_mask)

    cam_base = env.camera.get_cam_mat()
    tcp_base = env.robot.get_tcp_pose()
    tcp_angles = env.robot.get_tcp_angles()

    live_obs = []
    for target_pose, control in permute_pose_grid(env, tcp_base):
        # go to state
        action = [*target_pose, tcp_angles[2], 1]
        state2, _, _, info = env.step(action, control)
        # and collect data
        tcp_live = env.robot.get_tcp_pose()
        cam_live = env.camera.get_cam_mat()
        live_obs.append(dict(state=state2, info=info,
                             pose=tcp_live, cam=cam_live))

        # T_tcp_cam2 = cam_live @ np.linalg.inv(tcp_live)
        # diff = T_tcp_cam2 @ np.linalg.inv(T_tcp_cam)
        # err = np.linalg.norm(diff[:3, 3])
        # errors.append(err)
        # break

    # print("mean error", np.mean(errors))

    # initialize servo module
    control_config = dict(mode="pointcloud-abs", threshold=0.40)
    servo_module = ServoingModule(demo_pb,
                                  control_config=control_config,
                                  plot=True, save_dir=None)

    servo_module.set_env(env)

    pcds = []
    for live_i in live_obs:
        live_state = live_i["state"]
        live_info = live_i["info"]
        action, _, servo_info = servo_module.step(live_state, live_info)

        pos_change = (live_state["robot_state"]["tcp_pos"]-tcp_base[:3, 3]).tolist()
        print("position change: "+", ".join([f"{x:0.3f}" for x in pos_change]))


        # cam base -> estimate live_cam and live_tcp
        t_camdemo_camlive = servo_info["align_trf"]
        live_cam_est = cam_base @ t_camdemo_camlive
        diff_pos, diff_rot = get_pos_orn_diff(live_i["cam"], live_cam_est)
        print(f"live_cam {diff_pos:0.4f}, {diff_rot:0.4f}")
        assert diff_pos < .001  # 1mm
        assert diff_rot < .005

        live_tcp_est = live_cam_est @ np.linalg.inv(servo_module.T_tcp_cam)
        diff_pos, diff_rot = get_pos_orn_diff(live_i["pose"], live_tcp_est)
        assert diff_pos < .001  # 1mm
        assert diff_rot < .005
        print(f"live_tcp {diff_pos:0.4f}, {diff_rot:0.4f}")

        # live_tcp -> cam_base and tcp_base
        cam_base_est = live_i["pose"] @ servo_module.T_tcp_cam @ np.linalg.inv(t_camdemo_camlive)
        diff_pos, diff_rot = get_pos_orn_diff(cam_base, cam_base_est)
        assert diff_pos < .001  # 1mm
        assert diff_rot < .005
        print(f"cam_base {diff_pos:0.4f}, {diff_rot:0.4f}")

        tcp_base_est = cam_base_est @ np.linalg.inv(servo_module.T_tcp_cam)
        diff_pos, diff_rot = get_pos_orn_diff(tcp_base, tcp_base_est)
        assert diff_pos < .001  # 1mm
        assert diff_rot < .005
        print(f"tcp_base {diff_pos:0.4f}, {diff_rot:0.4f}")

        # using servo module
        tcp_base_est2 = servo_module.abs_to_world_tcp(servo_info, {"world_tcp": live_i["pose"]})
        diff_pos, diff_rot = get_pos_orn_diff(tcp_base, tcp_base_est2)
        assert diff_pos < .001  # 1mm
        assert diff_rot < .005
        print(f"tcp_base2 {diff_pos:0.4f}, {diff_rot:0.4f}\n")

        plot_bt = False
        if plot_bt:
            env.p.removeAllUserDebugItems()
            env.p.addUserDebugLine([0, 0, 0], live_i["pose"][:3, 3], lineColorRGB=[0, 1, 0],
                                   lineWidth=2, physicsClientId=0)  # green
            env.p.addUserDebugLine([0, 0, 0], live_tcp_est[:3, 3], lineColorRGB=[0, 0, 1],
                                   lineWidth=2, physicsClientId=0)  # blue
        plot_o3d = False
        if plot_o3d:
            import open3d as o3d
            start_pc = servo_module.demo_cam.generate_pointcloud2(live_state["rgb_gripper"],
                                                                  live_state["depth_gripper"])
            colors = start_pc[:, 4:7] / 255.  # recorded image colors
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(start_pc[:, :3])
            pcd1.colors = o3d.utility.Vector3dVector(colors)
            # pcd1.transform(live_i["cam"])  # live cam transforms objects to point
            pcd1.transform(live_cam_est)
            pcds.append(pcd1)
            o3d.visualization.draw_geometries(pcds)


        yield diff_pos, diff_rot

    if plot_o3d:
        o3d.visualization.draw_geometries(pcds)


class MoveThenEstimate(unittest.TestCase):
    """
    Move the robot, then estimate motion.
    """
    def test_move_absolute_then_estimate(self, is_sim=True):
        """with absolute motions."""
        if is_sim:
            env = RobotSimEnv(task="flow_calib", robot="kuka",
                              obs_type="image_state", renderer=RENDERER,
                              act_type='continuous', control="absolute",
                              max_steps=600, initial_pose="close",
                              img_size=(256, 256),
                              param_randomize=False)
        else:
            from gym_grasping.envs.iiwa_env import IIWAEnv
            env = IIWAEnv(act_type='continuous', freq=20, obs_type='image_state_reduced', dv=0.01,
                          drot=0.04, joint_vel=0.05,  # trajectory_type='lin',
                          gripper_rot_vel=0.3, joint_acc=0.3, use_impedance=True, safety_stop=True,
                          dof='5dof',
                          reset_pose=(0, -0.56, 0.26, math.pi, 0, math.pi / 2),
                          obs_dict=False)

        pos_errs = []
        rot_errs = []
        for diff_pos, diff_rot in move_absolute_then_estimate(env):
            self.assertLess(diff_pos, .02)
            self.assertLess(diff_rot, .02)
            pos_errs.append(diff_pos)
            rot_errs.append(diff_rot)

        print("mean pos error", np.mean(pos_errs))
        print("mean rot error", np.mean(rot_errs))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="")
    unittest.main()
