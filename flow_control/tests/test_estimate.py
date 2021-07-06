"""
Test functional behavior through built-in policies.
"""
import os
import math
import logging
import unittest
from copy import copy
import numpy as np
from scipy.spatial.transform import Rotation as R
from gym_grasping.envs.robot_sim_env import RobotSimEnv
from flow_control.servoing.module import ServoingModule

is_ci = "CI" in os.environ

if is_ci:
    obs_type = "state"
    renderer = "tiny"
else:
    obs_type = "image"
    renderer = "debug"


def get_pose_diff(trf_a, trf_b):
    diff_pos = np.linalg.norm(trf_a[:3, 3]-trf_b[:3, 3], 2)
    diff_rot = R.from_matrix(trf_a[:3, :3] @ np.linalg.inv(trf_b[:3, :3])).magnitude()
    return diff_pos, diff_rot


def make_demo_dict(env, base_state, base_info, base_action):
    """
    create keep dict with info from state
    """
    try:
        mask = base_info["seg_mask"] == 2
    except KeyError:
        mask = np.zeros_like(base_info["depth"], dtype=bool)

    demo_dict = dict(env_info=env.get_info(),
                     rgb=base_state[np.newaxis, :],
                     depth=base_info["depth"][np.newaxis, :],
                     mask=mask[np.newaxis, :],
                     state=base_info["robot_state_full"][np.newaxis, :],
                     keep_dict={0: None},
                     actions=np.array(base_action)[np.newaxis, :])
    return demo_dict


def get_target_poses(env, tcp_base):
    control = env.robot.get_control("absolute")  # min_iter=24)

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
    start_pc = servo_module.demo_cam.generate_pointcloud2(rgb, depth)
    colors = start_pc[:, 4:7]/255.  # recorded image colors
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(start_pc[:, :3])
    pcd1.colors = o3d.utility.Vector3dVector(colors)
    pcd1.transform(cam_live)

    end_pc = servo_module.demo_cam.generate_pointcloud2(servo_module.demo.rgb, servo_module.demo.depth)
    colors = end_pc[:, 4:7]/255.  # recorded image colors
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(end_pc[:, :3])
    pcd2.colors = o3d.utility.Vector3dVector(colors)
    pcd2.transform(cam_base)
    o3d.visualization.draw_geometries([pcd1, pcd2])


def move_absolute_then_estimate(env):
    """test performance of scripted policy, with parallel gripper"""
    # record base frame
    base_state = env.reset()
    base_info = env.get_obs_info()

    tcp_base = env.robot.get_tcp_pose()
    tcp_angles = env.robot.get_tcp_angles()
    cam_base = env.camera.get_cam_mat()
    # T_cam_tcp = cam_base @ np.linalg.inv(tcp_base)

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
                         pose=tcp_live, cam=cam_live
                         ))

        # T_cam_tcp2 = cam_live @ np.linalg.inv(tcp_live)
        # diff = T_cam_tcp2 @ np.linalg.inv(T_cam_tcp)
        # err = np.linalg.norm(diff[:3, 3])
        # errors.append(err)
        #break

    # print("mean error", np.mean(errors))

    # initialize servo module
    base_action = [*tcp_base[:3, 3], tcp_angles[2], 1]
    demo_dict = make_demo_dict(env, base_state, base_info, base_action)
    control_config = dict(mode="pointcloud-abs", threshold=0.40)

    servo_module = ServoingModule(demo_dict,
                                  control_config=control_config,
                                  plot=True, save_dir=None)
    servo_module.set_env(env)

    from pdb import set_trace
    import open3d as o3d

    pcds = []
    for i in range(len(live)):
        live_state = live[i]["state"]
        live_info = live[i]["info"]
        action, done, servo_info = servo_module.step(live_state, live_info)

        # cam base -> estimate live_cam and live_tcp
        t_camdemo_camlive = servo_info["align_trf"]
        live_cam_est = cam_base @ t_camdemo_camlive
        diff_pos, diff_rot = get_pose_diff(live[i]["cam"], live_cam_est)
        assert(diff_pos < .005)  # 5 mm
        assert(diff_rot < .005)

        live_tcp_est = live_cam_est @ np.linalg.inv(servo_module.T_cam_tcp)
        diff_pos, diff_rot = get_pose_diff(live[i]["pose"], live_tcp_est)
        assert(diff_pos < .005)  # 5 mm
        assert(diff_rot < .005)

        # live_tcp -> cam_base and tcp_base
        cam_base_est = live[i]["pose"] @ servo_module.T_cam_tcp @ np.linalg.inv(t_camdemo_camlive)
        diff_pos, diff_rot = get_pose_diff(cam_base, cam_base_est)
        assert(diff_pos < .005)  # 5 mm
        assert(diff_rot < .005)

        tcp_base_est = cam_base_est @ np.linalg.inv(servo_module.T_cam_tcp)
        diff_pos, diff_rot = get_pose_diff(tcp_base, tcp_base_est)
        assert(diff_pos < .005)  # 5 mm
        assert(diff_rot < .005)

        # using servo module
        tcp_base_est2 = servo_module.abs_to_tcp_world(servo_info, {"tcp_world":live[i]["pose"]})
        diff_pos, diff_rot = get_pose_diff(tcp_base, tcp_base_est2)
        assert(diff_pos < .005)  # 5 mm
        assert(diff_rot < .005)

        plot_bt = True
        if plot_bt:
            env.p.removeAllUserDebugItems()
            env.p.addUserDebugLine([0, 0, 0], live[i]["pose"][:3, 3], lineColorRGB=[0, 1, 0],
                                   lineWidth=2, physicsClientId=0)  # green
            env.p.addUserDebugLine([0, 0, 0], live_tcp_est[:3, 3], lineColorRGB=[0, 0, 1],
                                   lineWidth=2, physicsClientId=0)  # blue
        plot_o3d = False
        if plot_o3d:
            start_pc = servo_module.demo_cam.generate_pointcloud2(live_state, live_info["depth"])
            colors = start_pc[:, 4:7]/255.  # recorded image colors
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(start_pc[:, :3])
            pcd1.colors = o3d.utility.Vector3dVector(colors)
            # pcd1.transform(live[i]["cam"])  # live cam transforms objects to point
            pcd1.transform(live_cam_est)
            pcds.append(pcd1)

        yield diff_pos, diff_rot

    if plot_o3d:
        o3d.visualization.draw_geometries(pcds)


class MoveThenEstimate(unittest.TestCase):
    def test_move_absolute_then_estimate(self, is_sim=True):
        if is_sim:
            env = RobotSimEnv(task="flow_calib", robot="kuka",
                              obs_type=obs_type, renderer=renderer,
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
