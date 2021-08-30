import time

import cv2
#import hydra
import numpy as np

from flow_control.demo.compute_center import compute_center


def file_get_image(fn):
    tmp = np.load(fn)
    return tmp["rgb_gripper"], tmp["depth_gripper"]


def get_point_in_world_frame(cam, robot, T_tcp_cam, clicked_point, depth):
    point_cam_frame = cam.deproject(clicked_point, depth, homogeneous=True)
    if point_cam_frame is None:
        print("No depth measurement at clicked point")
        return None
    T_world_tcp = robot.get_tcp_pose()
    point_world_frame = T_world_tcp @ T_tcp_cam @ point_cam_frame
    return point_world_frame[:3]


def move_home(robot):
    goal_pos = np.array((0.56, 0.0, 0.24))
    goal_orn = np.array((1,  0, 0, 0.))
    robot.move_cart_pos_abs_lin(goal_pos, goal_orn)


actions = dict(
    object=dict(clicks=1, instructions="Click once on the object"),
    nest=dict(clicks=2, instructions="Click left and right of the nest")
)


class WaypointFactory:
    def __init__(self):
        self.index = None
        self.waypoints = [
            "object",
            "nest"
            ]
        self.done_waypoints = []
        self.clicked_points = []

        self.rgb = None
        self.depth = None

    def step(self, rgb, depth):
        # observe new frame
        self.rgb = rgb
        self.depth = depth
        self.step_wp()

    def step_wp(self):
        # step to the next waypoint
        if self.index is None:
            print("-----------Starting Click Policy-------------")
            self.index = 0
        else:
            if self.index == len(self.waypoints) - 1:
                print("-----------Completed Click Policy-------------")
                return

            self.index += 1

        self.clicked_points = []
        action = self.waypoints[self.index]
        print(actions[action]["instructions"])


    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append((x, y))
            action = actions[self.waypoints[self.index]]
            if len(self.clicked_points) == action["clicks"]:
                print("clicked", (x, y),"dispatching")
                self.dispatch_action()
            else:
                print("clicked", (x, y))

    def dispatch_action(self):
        action_name = self.waypoints[self.index]
        action = actions[action_name]

        if action_name == "object":
            assert len(self.clicked_points) == action["clicks"]
            clicked_point = self.clicked_points[0]

            new_center = compute_center(self.rgb, None, clicked_point)

            # show points in image
            self.rgb[clicked_point[1], clicked_point[0]] = (255, 0, 0)
            self.rgb[new_center[1], new_center[0]] = (0, 255, 0)
            cv2.imshow("image", self.rgb[:, :, ::-1])

            point_world = get_point_in_world_frame(cam, robot, T_tcp_cam, clicked_point, depth)
            over_offset = np.array([0, 0, 0.10])
            grip_offset = np.array([0, 0, 0.008])
            self.done_waypoints.append( (point_world + over_offset, orn, 1))
            self.done_waypoints.append(( (point_world + grip_offset, orn, 0))
            print("success: ", action_name, "\n")
            self.step_wp()

        if action_name == "nest":
            assert len(self.clicked_points) == action["clicks"]
            center = np.array(self.clicked_points).mean(axis=0).round().astype(int).tolist()
            length_x = self.clicked_points[1][0] - self.clicked_points[0][0]
            new_center = compute_center(self.rgb, None, center)

            # show points in image
            clicked_point = self.clicked_points[0]
            self.rgb[clicked_point[1], clicked_point[0]] = (255, 0, 0)
            clicked_point = self.clicked_points[1]
            self.rgb[clicked_point[1], clicked_point[0]] = (255, 0, 0)
            self.rgb[new_center[1], new_center[0]] = (0, 255, 0)
            cv2.imshow("image", self.rgb[:, :, ::-1])

            # de-project with different height
            # find average z height of both points
            # find coordinate where ray intersects height

            over_offset = np.array([0, 0, 0.10])
            grip_offset = np.array([0, 0, 0.008])
            relase_offset = np.array([0, 0, -.015])
            # move to above position
            # move to close position

            # pause / manual input

            # move straigt down

            print("success: ", action_name, "\n")
            self.step_wp()






#@hydra.main(config_path="../conf", config_name="panda_calibrate_gripper_cam")
def main():
    node = platform.uname().node
    if node in ('knoppers',):
        cam = hydra.utils.instantiate(cfg.cam)
        robot = hydra.utils.instantiate(cfg.robot)
        T_tcp_cam = cam.get_extrinsic_calibration(robot.name)

        robot.move_to_neutral()
        pos, orn = robot.get_tcp_pos_orn()
        rgb, depth = cam.get_image()
        #move_home(robot)
    else:
        rgb, depth = file_get_image("frame_000000.npz")
        T_tcp_cam = np.eye(4)


    wf = WaypointFactory()

    def callback(event, x, y, flags, param):
        nonlocal wf
        wf.callback(event, x, y, flags, param)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", wf.callback)
    wf.step(rgb, depth)

    prented_click_points = True
    if prented_click_points:
        clicked_points = [(317, 207),
                          (558, 156), (604, 149)]
        for c_pts in clicked_points:
            callback(cv2.EVENT_LBUTTONDOWN, *c_pts, None, None)


    clicked_point = None
    while 1:
        #rgb, depth = cam.get_image()
        cv2.imshow("image", rgb[:, :, ::-1])
        #cv2.imshow("depth", depth)
        cv2.waitKey(1)

        if clicked_point is not None:
            """
            point_world = get_point_in_world_frame(cam, robot, T_tcp_cam, clicked_point, depth)
            if point_world is not None:
                over_offset = np.array([0, 0, 0.10])
                grip_offset = np.array([0, 0, 0.008])
                robot.move_cart_pos_abs_lin(point_world + over_offset, orn)
                robot.move_cart_pos_abs_lin(point_world + grip_offset, orn)
                robot.close_gripper()
                time.sleep(1)
                #robot.move_to_neutral()
                #robot.open_gripper()
                move_home(robot)
            """
            clicked_point = None


if __name__ == "__main__":
    main()
