import cv2
import numpy as np

import matplotlib
from matplotlib import pyplot as plt

from scipy.spatial.transform import Rotation as R
from flow_control.demo.demo_segment_util import transform_depth
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

        self.extrinsic = np.linalg.inv(T_TCP_CAM)
        i_info = self.env.camera.get_info()['calibration']
        self.intrinsic = (
            i_info['width'], i_info['height'],
            i_info['fx'], i_info['fy'],
            i_info['ppx'], i_info['ppy']
        )

        self.clicks = []

    def compute_center(self, depth, point):

        flat_depth = transform_depth(
            depth.copy(),
            self.extrinsic,
            self.env.camera.calibration
        )

        target_depth = flat_depth[point[::-1]]
        target_mask = np.abs(flat_depth - target_depth) < 0.004

        # Extract mask contours and get the one containing point
        contour = min(
            [c for c in cv2.findContours(
                np.uint8(target_mask),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )[0] if cv2.pointPolygonTest(c, point, False) > 0],
            key=lambda c: cv2.contourArea(c)
        )



        # Center of the contour
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        plt.imshow(target_mask * flat_depth)
        plt.plot(cX, cY, 'ro', ms=7.0)
        plt.show()

        return (cX, cY)

    def select_next(self, state, info):
        fig = plt.gcf()
        fig.canvas.mpl_connect('button_press_event', self.click_event)

        self.depth = info['depth'].copy()
        self.img = state.copy()

        plt.imshow(self.img)
        plt.show()

    def update_extrinsics(self):
        tcp_R = R.from_euler("xyz", self.env.robot.get_tcp_angles()).as_matrix()
        #tcp_R = np.array(self.env.p.getMatrixFromQuaternion(
        #    self.env.p.getQuaternionFromEuler(self.env.robot.get_tcp_angles())
        #)).reshape((3, 3))
        tcp_T = np.array(self.env.robot.get_tcp_pos())
        tcp_mat = np.vstack([np.hstack([tcp_R, tcp_T[:, None]]), [0, 0, 0, 1]])
        self.extrinsic = np.linalg.inv(tcp_mat @ T_TCP_CAM)

    def click_event(self, event):
        if event.xdata is None or event.ydata is None:
            return

        plt.plot(int(event.xdata), int(event.ydata), 'ro', ms=7.0)
        plt.draw()
        self.clicks.append((int(event.xdata), int(event.ydata)))

        if len(self.clicks) < 4:
            return

        points = self.clicks
        self.clicks = []

        self.update_extrinsics()
        plt.close()

        points.append(
            self.compute_center(self.depth.copy(), points[0])
        )

        points.append(
            self.compute_center(self.depth.copy(), points[2])
        )

        # plt.imshow(self.img)
        # plt.plot(cX, cY, 'ro', ms=10.0)
        # plt.show()


        for i in range(len(points)):
            # Center point to absolute
            x, y = points[i]
            z = self.depth[y, x]
            x = z * (x - self.intrinsic[4]) / self.intrinsic[2]
            y = z * (y - self.intrinsic[5]) / self.intrinsic[3]
            p = np.array([[x, y, z, 1]]).T
            points[i] = np.linalg.inv(self.extrinsic) @ p
        points = np.hstack(points)

        class Waypoint:
            def __init__(self, trn, orn, gripper):
                self.trn = trn
                self.orn = orn
                self.gripper = gripper

        # Object orientation
        d1 = points[:, 0] - points[:, 1]
        d1 = d1 / np.linalg.norm(d1)
        d1 = R.from_euler("xyz", (0, 0, np.arctan2(d1[0], -d1[1]))).as_quat()

        # Target orientation
        d2 = points[:, 2] - points[:, 3]
        d2 = d2 / np.linalg.norm(d2)
        d2 = R.from_euler("xyz", (0, 0, np.arctan2(d2[0], -d2[1]))).as_quat()

        print(d1)
        print(d2)

        self.trajectory = [
            Waypoint(points[:3, 4], d1, 'close'),
            Waypoint(points[:3, 4] + [0, 0, 0.05], d1, None),
            Waypoint(points[:3, 5] + [0, 0, 0.05], d2, None),
            Waypoint(points[:3, 5] + [0, 0, 0.02], d2, 'open'),
            Waypoint(points[:3, 5] + [0, 0, 0.05], d2, None),
        ]

        self.stage = 0
        self.refined = False

    def policy(self, state, info):

        self.img = state.copy()
        self.depth = info['depth'].copy()

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
                cv2.Rodrigues(self.extrinsic[:3, :3])[0],
                self.extrinsic[:3, 3],
                np.array([[self.intrinsic[2], 0, self.intrinsic[4]],
                          [0, self.intrinsic[3], self.intrinsic[5]],
                          [0, 0, 1]]),
                None
            )

            cX, cY = self.compute_center(
                self.depth.copy(),
                (int(points[0, 0, 0]), int(points[0, 0, 1]))
            )
            # plt.imshow(self.img)
            # plt.plot(cX, cY, 'ro', ms=7.0)
            # plt.show()

            z = self.depth[cY, cX]
            x = z * (cX - self.intrinsic[4]) / self.intrinsic[2]
            y = z * (cY - self.intrinsic[5]) / self.intrinsic[3]
            p = np.array([[x, y, z, 1]]).T
            center_p = np.linalg.inv(self.extrinsic) @ p

            print('Change center coordinate')
            wp.trn = center_p[:3, 0]
            self.refined = True

        if g is not None:
            gripper_act = {"open": self.env.robot.gripper.OPEN_ACTION,
                           "close": self.env.robot.gripper.CLOSE_ACTION}[g]

            o_gripper_act = {"close": self.env.robot.gripper.OPEN_ACTION,
                             "open": self.env.robot.gripper.CLOSE_ACTION}[g]

            is_suction = isinstance(self.env.robot.gripper, SuctionGripper)

            if is_suction:
                if g == "close":
                    gripper_act = self.env.robot.gripper.CLOSE_ACTION
                    if self.env.robot.connected:
                        self.stage += 1
                elif g == "open":
                    gripper_act = self.env.robot.gripper.OPEN_ACTION
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
