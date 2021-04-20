import cv2
import open3d
import numpy as np

import matplotlib
from matplotlib import pyplot as plt

from scipy.spatial.transform import Rotation as R
from flow_control.demo_segment_util import transform_depth
from gym_grasping.robots.grippers import SuctionGripper

matplotlib.use("TkAgg")

T_TCP_CAM = np.array([
   [0.99987185, -0.00306941, -0.01571176, 0.00169436],
   [-0.00515523, 0.86743151, -0.49752989, 0.11860651],
   [0.015156,    0.49754713,  0.86730453, -0.18967231],
   [0., 0., 0., 1.]])


class ClickPolicy:

    def __init__(self, env):
        self.env = env

        self.cam_params = open3d.camera.PinholeCameraParameters()
        self.cam_params.extrinsic = np.linalg.inv(T_TCP_CAM)
        i_info = self.env.camera.get_info()['calibration']
        self.cam_params.intrinsic.set_intrinsics(
            i_info['width'], i_info['height'],
            i_info['fx'], i_info['fy'],
            i_info['ppx'], i_info['ppy']
        )

    def compute_center(self, depth, point):

        flat_depth = transform_depth(
            depth.copy(),
            self.cam_params.extrinsic,
            self.env.camera.calibration
        )

        target_depth = flat_depth[point[::-1]]
        target_mask = np.abs(flat_depth - target_depth) < 0.002

        # Extract mask contours and get the one containing point
        contour = next(
            c for c in cv2.findContours(
                np.uint8(target_mask),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )[0] if cv2.pointPolygonTest(c, point, False) > 0
        )

        # Center of the contour
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        return (cX, cY)

    def select_next(self):
        fig = plt.gcf()
        fig.canvas.mpl_connect('button_press_event', self.click_event)

        plt.imshow(self.env._observation)
        plt.show()

    def update_extrinsics(self):
        tcp_R = np.array(self.env.p.getMatrixFromQuaternion(
            self.env.p.getQuaternionFromEuler(self.env.robot.get_tcp_angles())
        )).reshape((3, 3))
        tcp_T = np.array(self.env.robot.get_tcp_pos())
        tcp_mat = np.vstack([np.hstack([tcp_R, tcp_T[:, None]]), [0, 0, 0, 1]])
        self.cam_params.extrinsic = np.linalg.inv(tcp_mat @ T_TCP_CAM)

    def click_event(self, event):
        if event.xdata is None or event.ydata is None:
            return

        self.update_extrinsics()
        click = (int(event.xdata), int(event.ydata))
        plt.close()

        cX, cY = self.compute_center(self.env._info['depth'].copy(), click)

        # plt.imshow(self.env._observation)
        # plt.plot(cX, cY, 'ro', ms=10.0)
        # plt.show()

        # Create 3D point cloud
        p_cloud = open3d.geometry.PointCloud.create_from_depth_image(
            open3d.geometry.Image(self.env._info['depth'].astype(np.float32)),
            self.cam_params.intrinsic,
            self.cam_params.extrinsic,
            depth_scale=1.0
        ).remove_non_finite_points()

        # Ask for points
        vis = open3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(p_cloud)
        ctr = vis.get_view_control()

        # params = open3d.io.read_pinhole_camera_parameters("coolcamera.json")
        params = ctr.convert_to_pinhole_camera_parameters()
        params.extrinsic = self.cam_params.extrinsic
        ctr.convert_from_pinhole_camera_parameters(params)
        vis.run()
        vis.destroy_window()
        values = vis.get_picked_points()
        vis.close()
        values = np.asarray(p_cloud.points)[values]  # 4, 3
        values = np.hstack([values, np.ones((len(values), 1))]).T

        # Center point to absolute
        z = self.env._info['depth'][cY, cX]
        i = self.cam_params.intrinsic
        x = z * (cX - i.get_principal_point()[0]) / i.get_focal_length()[0]
        y = z * (cY - i.get_principal_point()[1]) / i.get_focal_length()[1]
        p = np.array([[x, y, z, 1]]).T
        center_p = np.linalg.inv(self.cam_params.extrinsic) @ p

        class Waypoint:
            def __init__(self, trn, orn, gripper):
                self.trn = trn
                self.orn = orn
                self.gripper = gripper

        # Object orientation
        d1 = values[:, 0] - values[:, 1]
        d1 = d1 / np.linalg.norm(d1)
        d1 = R.from_euler("xyz", (0, 0, np.arctan2(d1[0], -d1[1]))).as_quat()

        # Target orientation
        d2 = values[:, 2] - values[:, 3]
        d2 = d2 / np.linalg.norm(d2)
        d2 = R.from_euler("xyz", (0, 0, np.arctan2(d2[0], d2[1]))).as_quat()

        self.trajectory = [
            Waypoint(center_p[:3, 0], d1, 'close'),
            Waypoint(center_p[:3, 0] + [0, 0, 0.05], d1, None),
            Waypoint(values[:3, 2] + [0, 0, 0.05], d2, None),
            Waypoint(values[:3, 2] + [0, 0, 0.02], d2, 'open'),
            Waypoint(values[:3, 2] + [0, 0, 0.05], d2, None),
        ]

        self.stage = 0
        self.refined = False

    def policy(self):

        if self.stage == len(self.trajectory):
            return [0, 0, 0, 0, 0]

        self.update_extrinsics()

        goal_dist = np.linalg.norm(
            self.trajectory[self.stage].trn -
            np.array(self.env.robot.get_tcp_pos())
        )

        wp = self.trajectory[self.stage]
        g = wp.gripper

        if not self.refined and goal_dist < 0.01:

            points, _ = cv2.projectPoints(
                wp.trn.reshape(1, 3),
                cv2.Rodrigues(self.cam_params.extrinsic[:3, :3])[0],
                self.cam_params.extrinsic[:3, 3],
                self.cam_params.intrinsic.intrinsic_matrix,
                None
            )

            cX, cY = self.compute_center(
                self.env._info['depth'].copy(),
                (int(points[0, 0, 0]), int(points[0, 0, 1]))
            )

            z = self.env._info['depth'][cY, cX]
            i = self.cam_params.intrinsic
            x = z * (cX - i.get_principal_point()[0]) / i.get_focal_length()[0]
            y = z * (cY - i.get_principal_point()[1]) / i.get_focal_length()[1]
            p = np.array([[x, y, z, 1]]).T
            center_p = np.linalg.inv(self.cam_params.extrinsic) @ p

            print('Change center coordinate')
            wp.trn = center_p[:3, 0]
            self.refined = True

        if g is not None:
            gripper_act = {"open": self.env.robot.gripper.open_action,
                           "close": self.env.robot.gripper.close_action}[g]

            o_gripper_act = {"close": self.env.robot.gripper.open_action,
                             "open": self.env.robot.gripper.close_action}[g]

            is_suction = isinstance(self.env.robot.gripper, SuctionGripper)

            if is_suction:
                if g == "close":
                    gripper_act = self.env.robot.gripper.close_action
                    if self.env.robot.connected:
                        self.stage += 1
                elif g == "open":
                    gripper_act = self.env.robot.gripper.open_action
                    if not self.env.robot.connected:
                        self.stage += 1
            else:  # parallel gripper
                if goal_dist < 2e-3:
                    # countdown is a quick and dirty solution, should be
                    # be something more like if self.env.robot.connected
                    self.countdown -= 1
                    if self.countdown <= 0:
                        self.stage += 1

                else:
                    gripper_act = o_gripper_act
                    self.countdown = 4
            self.prev_gripper_act = gripper_act
        else:
            if goal_dist < 2e-3 and self.stage < len(self.trajectory)-1:
                self.stage += 1

            gripper_act = self.prev_gripper_act

        t = wp.trn.tolist()
        o = R.from_quat(wp.orn).as_euler("xyz")[2]
        return t + [o, gripper_act]
