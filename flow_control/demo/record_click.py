import time
import logging
import platform

from copy import deepcopy

import cv2
import hydra
import numpy as np

from flow_control.demo.compute_center import compute_center

def get_point_in_world_frame(cam, T_tcp_cam, T_world_tcp, depth, clicked_point):
    point_cam_frame = cam.deproject(clicked_point, depth, homogeneous=True)
    if point_cam_frame is None:
        print("No depth measurement at clicked point")
        return None
    point_world_frame = T_world_tcp @ T_tcp_cam @ point_cam_frame
    return point_world_frame[:3]


act_info = dict(
    object=dict(clicks=1, instructions="Click once on the object"),
    nest=dict(clicks=2, instructions="Click left and right of the nest")
)


# these are called actions (always take pictures first)
shape_sorter = [dict(name="move", pos=(0.47, 0.08, 0.26), orn=(1, 0, 0, 0)),
                dict(name="object", width=64, preg_offset=(0, 0, 0.010), grip_offset=(0, 0, 0.010)),
                dict(name="nest", inst_offset=(0, 0, 0.045), release_offset=(0, 0, 0.035))]


vacuum = [dict(name="move", pos=(0.47, 0.08, 0.26), orn=(1, 0, 0, 0)),
          dict(name="object", width=128, preg_offset=(.01, 0, 0.052), grip_offset=(.01, 0, 0.04)),
          dict(name="nest", inst_offset=(0, 0, 0.075))]


# This seems better because it's more explicit
# wp1 = move(fixed, tag="start")
# wp2 = move(tag="object-high")
# wp2 = move(tag="object-low")
# wp2 = move(tag="object-grasp")

# wp3 = move(tag="nest-high")
# wp3 = move(tag="nest-low")
# wp3 = move(tag="nest-inst")

# move1
# pic = record_picture()
# object = detect_object(pic)
# nest = detect_nest(pic)

class WaypointFactory:
    def __init__(self, cam, T_tcp_cam):
        self.index = None
        self.waypoints = shape_sorter

        self.done_waypoints = []
        self.done_waypoint_names = []
        self.clicked_points = []

        self.lock = False  # block updating with new images
        self.done = False

        self.rgb = None
        self.depth = None
        self.pos = None
        self.orn = None
        self.T_world_tcp = None

        self.cam = cam
        self.T_tcp_cam = T_tcp_cam

    def __del__(self):
        cv2.destroyWindow("image")
        cv2.destroyWindow("depth")

    def step(self, rgb, depth, pos, orn, T_world_tcp):
        # update state info (live mode)
        if self.lock:
            return

        cv2.imshow("image", rgb[:, :, ::-1])
        cv2.imshow("depth", depth / depth.max())
        cv2.waitKey(1)

        # observe new frame
        self.rgb = rgb
        self.depth = depth
        self.pos = pos
        self.orn = orn
        self.T_world_tcp = T_world_tcp

    def next_wp(self):
        # step to the next waypoint
        if self.index is None:
            print("-----------Starting Click Policy-------------")
            self.index = 0
        else:
            if self.index == len(self.waypoints) - 1:
                print("-----------Completed Click Policy-------------")
                self.lock = False
                self.done = True
                return
            self.index += 1

        action_name = self.waypoints[self.index]["name"]
        if action_name in act_info:
            print(act_info[action_name]["instructions"])

    def callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.lock:
                self.lock = True
                self.next_wp()

            self.clicked_points.append((x, y))

            for wp in self.waypoints[self.index:]:
                action_name = wp["name"]
                if action_name not in act_info:
                    self.dispatch_action()
                    continue
                elif len(self.clicked_points) == act_info[action_name]["clicks"]:
                    print("clicked", (x, y), "dispatching", action_name)
                    self.dispatch_action()
                else:
                    print("clicked", (x, y), len(self.clicked_points), act_info[action_name]["clicks"])
                break

    def add_waypoint(self, pos, orn, grip, name=""):
        self.done_waypoints.append((tuple(pos), tuple(orn), grip))
        self.done_waypoint_names.append(name)

    def dispatch_action(self):
        action = self.waypoints[self.index]
        name = action["name"]
        if action["name"] == "move":
            self.dispatch_move(action)

        elif action["name"] == "object":
            self.dispatch_object(action)

        elif action["name"] == "nest":
            self.dispatch_nest(action)
            print("success: ", action["name"])

        self.next_wp()

    def dispatch_move(self, action):
        self.add_waypoint(action["pos"], action["orn"], 1)

    def dispatch_object(self, action):
        name = "object"
        clicks = act_info[name]["clicks"] if name in act_info else None
        assert len(self.clicked_points) == clicks
        clicked_point = self.clicked_points[0]

        width = action["width"]
        new_center = compute_center(self.rgb, None, clicked_point, width=width)
        new_center = clicked_point

        # show points in image
        self.rgb[clicked_point[1], clicked_point[0]] = (255, 0, 0)
        self.rgb[new_center[1], new_center[0]] = (0, 255, 0)
        cv2.imshow("image", self.rgb[:, :, ::-1])
        cv2.imshow("depth", self.depth)

        point_world = get_point_in_world_frame(self.cam, self.T_tcp_cam, self.T_world_tcp, self.depth, clicked_point)
        over_offset = np.array([0, 0, 0.10])
        preg_offset = action["preg_offset"]
        grip_offset = action["grip_offset"]
        self.add_waypoint(point_world + over_offset, self.orn, 1, name="grip-over")
        self.add_waypoint(point_world + preg_offset, self.orn, 1, name="grip-before")
        self.add_waypoint(point_world + grip_offset, self.orn, 0, name="grip-after")
        self.clicked_points = []

    def dispatch_nest(self, action):
        name = "nest"
        clicks = act_info[name]["clicks"] if name in act_info else None
        assert len(self.clicked_points) == clicks
        center = np.array(self.clicked_points).mean(axis=0).round().astype(int).tolist()
        new_center = compute_center(self.rgb, None, center)

        # show points in image
        clicked_point = self.clicked_points[0]
        self.rgb[clicked_point[1], clicked_point[0]] = (255, 0, 0)
        clicked_point = self.clicked_points[1]
        self.rgb[clicked_point[1], clicked_point[0]] = (255, 0, 0)
        self.rgb[new_center[1], new_center[0]] = (0, 255, 0)
        cv2.imshow("image", self.rgb[:, :, ::-1])

        # de-project with different height, use average z height of both points
        depth_1 = self.depth[self.clicked_points[0][1], self.clicked_points[0][0]]
        depth_2 = self.depth[self.clicked_points[1][1], self.clicked_points[1][0]]
        # logging.info(f"depths: {depth_1}, {depth_2}")
        if depth_1 == 0 or depth_2 == 0:
            del self.clicked_points[1]
            del self.clicked_points[0]
            print("abort, click again!")

        depth_m = 0.5 * (depth_1 + depth_2)
        point_world = get_point_in_world_frame(self.cam, self.T_tcp_cam, self.T_world_tcp, depth_m, new_center)

        print("point world", point_world)
        over_offset = np.array([0, 0, 0.14])  # above
        inst_offset = action["inst_offset"]
        release_offset = action["release_offset"]

        if len(self.done_waypoints):
            over_wp = list(deepcopy(self.done_waypoints[-1]))
            over_wp[0] += over_offset
        self.add_waypoint(*over_wp, name="nest-height")
        self.add_waypoint(point_world + over_offset, self.orn, 0, name="nest-over")
        self.add_waypoint(point_world + inst_offset, self.orn, 0, name="nest-insertion")
        self.add_waypoint(point_world + release_offset, self.orn, 1, name="nest-release")
        self.clicked_points = []


from robot_io.recorder.simple_recorder import RecEnv

def test_shape_sorter():

    env = RecEnv("/home/argusm/CLUSTER/robot_recordings/flow/sick_vacuum/17-19-19/frame_000000.npz")
    #cam = DemoCam("/home/argusm/CLUSTER/robot_recordings/flow/ssh_demo/orange_trapeze/")
    #cam = DemoCam("/home/argusm/CLUSTER/robot_recordings/flow/sick_vacuum/17-19-19/")
    #robot = DemoRobot()
    cam = env.cam
    robot = env.robot
    action, record_info = None, {"trigger_release": False, "hold_event": False}
    T_tcp_cam = cam.get_extrinsic_calibration()
    wf = WaypointFactory(cam, T_tcp_cam)

    def callback(event, x, y, flags, param):
        nonlocal wf
        wf.callback(event, x, y, flags, param)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", wf.callback)

    clicked_point = None
    pretend_click_points = True
    while 1:
        # Important: Start with cv call to catch clicks that happened,
        # if we don't have those we would update the frame under the click
        cv2.waitKey(1)
        rgb, depth = cam.get_image()
        pos, orn = robot.get_tcp_pos_orn()
        T_world_tcp = robot.get_tcp_pose()
        wf.step(rgb, depth, pos, orn, T_world_tcp)
        if wf.done:
            break
        time.sleep(1)

        if pretend_click_points:
            clicked_points = [(317, 207),
                              (558, 156), (604, 149)]
            clicked_points = [(193, 135),
                              (425, 330), (544, 330)]

            for c_pts in clicked_points:
                callback(cv2.EVENT_LBUTTONDOWN, *c_pts, None, None)

    shape_sorter_wps = [((0.47, 0.08, 0.26), (1, 0, 0, 0), 1),
                        ((0.0, 0.0, 0.1), (1, 0, 0, 0), 1),
                        ((0.0, 0.0, 0.01), (1, 0, 0, 0), 1),
                        ((0.0, 0.0, 0.01), (1, 0, 0, 0), 0),
                        ((0.0, 0.0, 0.15000000000000002), (1, 0, 0, 0), 0),
                        ((0.0, 0.0, 0.14), (1, 0, 0, 0), 0),
                        ((0.0, 0.0, 0.045), (1, 0, 0, 0), 0),
                        ((0.0, 0.0, 0.035), (1, 0, 0, 0), 1)]
    assert wf.done_waypoints == shape_sorter_wps
    print(wf.done_waypoint_names)
    print("test passed.")


def run_live(cfg):
    robot = hydra.utils.instantiate(cfg.robot)
    env = hydra.utils.instantiate(cfg.env, robot=robot)
    cam = env.camera_manager.gripper_cam
    T_tcp_cam = cam.get_extrinsic_calibration(robot.name)
    # don't crop like default panda teleop
    assert cam.resize_resolution == [640, 480]
    robot.move_to_neutral()

    recorder = hydra.utils.instantiate(cfg.recorder)
    recorder.recording = True
    action, record_info = None, {"trigger_release": False, "hold_event": False}

    obs, _, _, _ = env.step(action)
    recorder.step(action, obs, record_info)
    #move_home(robot)

    T_tcp_cam = cam.get_extrinsic_calibration(robot.name)
    wf = WaypointFactory(cam, T_tcp_cam)

    def callback(event, x, y, flags, param):
        nonlocal wf
        wf.callback(event, x, y, flags, param)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", wf.callback)

    clicked_point = None
    while 1:
        # Important: Start with cv call to catch clicks that happened,
        # if we don't have those we would update the frame under the click
        cv2.waitKey(1)
        rgb, depth = cam.get_image()
        pos, orn = robot.get_tcp_pos_orn()
        T_world_tcp = robot.get_tcp_pose()
        wf.step(rgb, depth, pos, orn, T_world_tcp)
        if wf.done:
            break
        time.sleep(1)

    # record views
    obs, _, _, _ = env.step(action)  # record view at neutral position
    recorder.step(action, obs, record_info)

    prev_wp = None
    for i, wp in enumerate(wf.done_waypoints):
        robot.move_cart_pos_abs_lin(wp[0], wp[1])

        # TODO(max): we should not need to do this !!!
        pos, orn = robot.get_tcp_pos_orn()
        #while np.linalg.norm(np.array(wp[0]) - pos) > .02:
        #    logging.error("move failed: re-trying, error=" + str(np.array(wp[0]) - pos) + "pos=" + str(pos))
        #    robot.move_cart_pos_abs_lin(wp[0], wp[1])

        if prev_wp is not None and prev_wp[2] != wp[2]:
            if wp[2] == 0:
                robot.close_gripper()
                time.sleep(1)
            elif wp[2] == 1:
                robot.open_gripper()
                time.sleep(1)
            else:
                raise ValueError

        #res = input("next")
        if env:
            action = dict(motion=(wp[0], wp[1], -1 if wp[2] == 0 else 1), ref='abs')
            obs, _, _, _ = env.step(None)

            # TODO(max): we should not need to do this !!!
            #while np.all(obs['robot_state']['tcp_pos'] == [0, 0, 0]):
            #    print('Invalid state, recomputing step')
            #    time.sleep(0.5)
            #    obs, _, _, _ = env.step(None)

            recorder.step(action, obs, record_info)

        prev_wp = wp

    print("done executing wps!")


"""
TODO List:
    1. include waypoint names in recording
    2. save click info
    3. auto-segment
        a. use scroll wheel color/depth, left click outside?
        b. auto ground-plane detection

    4. save click info
    5. continouus segmentation of demo
"""


@hydra.main(config_path="/home/argusm/lang/robot_io/robot_io/conf", config_name="panda_teleop.yaml")
def main(cfg=None):
    node = platform.uname().node
    if node in ('knoppers',):
        run_live(cfg)
    else:
        test_shape_sorter()

if __name__ == "__main__":
    main()
