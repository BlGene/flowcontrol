import time
import logging
import platform
import pickle

from copy import deepcopy

import cv2
import hydra
import numpy as np

from flow_control.demo.compute_center import compute_center
from flow_control.demo.demo_segment_util import mask_color, mask_center

def get_point_in_world_frame(cam, T_tcp_cam, T_world_tcp, depth, clicked_point):
    point_cam_frame = cam.deproject(clicked_point, depth, homogeneous=True)
    if point_cam_frame is None:
        print("No depth measurement at clicked point")
        return None
    point_world_frame = T_world_tcp @ T_tcp_cam @ point_cam_frame
    return point_world_frame[:3]


# Statefull Operations
# try to keep the same API as https://doc.qt.io/qt-5/qtquick-input-mouseevents.html
class BaseOperation:
    def __init__(self):
        self.wf = None
        self.clicks_req = 0
        self.instructions = None

    def set_wf(self, wf):
        self.wf = wf

    def clicked(self):
        raise NotImplementedError

    def get_name(self):
        return NotImpementedError


class Move(BaseOperation):
    def __init__(self, pos, orn):
        self.name = "move"
        self.clicks_req = 0
        self.instructions = None

        self.pos = pos
        self.orn = orn

    def dispatch(self):
        self.wf.add_waypoint(self.pos, self.orn, 1)


class ObjectSelection(BaseOperation):
    def __init__(self, width=64, preg_offset=(0, 0, 0.010), grip_offset=(0, 0, 0.010)):
        self.name = "object"
        self.clicks_req = 1
        self.instructions = "Click once on the object"
        self.clicked_points = []

        self.width = width
        self.preg_offset = preg_offset
        self.grip_offset = grip_offset

    def clicked(self, click_info):
        event, x, y, flags, param = click_info
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append((x, y))
        else:
            return False, None

        if len(self.clicked_points) == self.clicks_req:
            return True, None
        else:
            print("clicked", (x, y), len(self.clicked_points), clicks)
            return False, None

    def dispatch(self):
        assert len(self.clicked_points) == self.clicks_req
        clicked_point = self.clicked_points[0]

        # segment and show
        clicked_color = self.wf.rgb[clicked_point[1], clicked_point[0]] / 255.
        object_mask = mask_color(self.wf.rgb, clicked_color, .60)
        object_mask = mask_center(object_mask)
        edge = np.gradient(object_mask.astype(float))
        edge = (np.abs(edge[0]) + np.abs(edge[1])) > 0
        self.wf.rgb[edge] = (255, 0, 0)

        # show clicked point
        self.wf.rgb[clicked_point[1], clicked_point[0]] = (255, 0, 0)

        cv2.imshow("image", self.wf.rgb[:, :, ::-1])
        cv2.waitKey(1)

        #new_center = compute_center(self.rgb, None, clicked_point, width=self.width)
        #new_center = clicked_point
        #self.wf.rgb[new_center[1], new_center[0]] = (0, 255, 0)

        point_world = get_point_in_world_frame(self.wf.cam, self.wf.T_tcp_cam, self.wf.T_world_tcp, self.wf.depth, clicked_point)
        over_offset = np.array([0, 0, 0.10])
        self.wf.add_waypoint(point_world + over_offset, self.wf.orn, 1, name="grip-over")
        self.wf.add_waypoint(point_world + self.preg_offset, self.wf.orn, 1, name="grip-before")
        self.wf.add_waypoint(point_world + self.grip_offset, self.wf.orn, 0, name="grip-after")
        self.clicked_points = []


class NestSelection(BaseOperation):
    def __init__(self, inst_offset=(0, 0, 0.05), release_offset=(0, 0, 0.02)):
        self.name = "nest"
        self.clicks_req = 2
        self.instructions = "Click left and right of the nest"
        self.clicked_points = []

        self.inst_offset = inst_offset
        self.release_offset = release_offset

    def clicked(self, click_info):
        event, x, y, flags, param = click_info
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append((x, y))
        else:
            return False, None

        if  len(self.clicked_points) == self.clicks_req:
            return True, None
        else:
            print(f"clicked ({x}, {y}, {len(self.clicked_points)}/{self.clicks_req}")
            return False, None

    def dispatch(self):
        name = "nest"
        assert len(self.clicked_points) == self.clicks_req
        center = np.array(self.clicked_points).mean(axis=0).round().astype(int).tolist()
        new_center = compute_center(self.wf.rgb, None, center)

        # show clicked points
        clicked_point = self.clicked_points[0]
        self.wf.rgb[clicked_point[1], clicked_point[0]] = (255, 0, 0)
        clicked_point = self.clicked_points[1]
        self.wf.rgb[clicked_point[1], clicked_point[0]] = (255, 0, 0)

        # segment and show
        clicked_color = self.wf.rgb[new_center[1], new_center[0]] / 255.
        object_mask = mask_color(self.wf.rgb, clicked_color, .55)
        object_mask = mask_center(object_mask)
        edge = np.gradient(object_mask.astype(float))
        edge = (np.abs(edge[0]) + np.abs(edge[1])) > 0
        self.wf.rgb[edge] = (255, 0, 0)

        self.wf.rgb[new_center[1], new_center[0]] = (0, 255, 0)
        cv2.imshow("image", self.wf.rgb[:, :, ::-1])
        cv2.waitKey(1)

        # de-project with different height, use average z height of both points
        depth_1 = self.wf.depth[self.clicked_points[0][1], self.clicked_points[0][0]]
        depth_2 = self.wf.depth[self.clicked_points[1][1], self.clicked_points[1][0]]
        # logging.info(f"depths: {depth_1}, {depth_2}")
        if depth_1 == 0 or depth_2 == 0:
            del self.clicked_points[1]
            del self.clicked_points[0]
            print("abort, click again!")

        depth_m = 0.5 * (depth_1 + depth_2)
        point_world = get_point_in_world_frame(self.wf.cam, self.wf.T_tcp_cam, self.wf.T_world_tcp, depth_m, new_center)

        print("point world", point_world)
        over_offset = np.array([0, 0, 0.14])  # above
        if len(self.wf.done_waypoints):
            over_wp = list(deepcopy(self.wf.done_waypoints[-1]))
            over_wp[0] += over_offset
        self.wf.add_waypoint(*over_wp, name="nest-height")
        self.wf.add_waypoint(point_world + over_offset, self.wf.orn, 0, name="nest-over")
        self.wf.add_waypoint(point_world + self.inst_offset, self.wf.orn, 0, name="nest-insertion")
        self.wf.add_waypoint(point_world + self.release_offset, self.wf.orn, 1, name="nest-release")
        self.clicked_points = []


# these are called actions (always take pictures first)
shape_sorter = [dict(name="move", pos=(0.47, 0.08, 0.26), orn=(1, 0, 0, 0)),
                dict(name="object", width=64, preg_offset=(0, 0, 0.010), grip_offset=(0, 0, 0.010)),
                dict(name="nest", inst_offset=(0, 0, 0.045), release_offset=(0, 0, 0.035))]


vacuum = [Move(pos=(0.47, 0.08, 0.26), orn=(1, 0, 0, 0)),
          ObjectSelection(width=128, preg_offset=(.01, 0, 0.052), grip_offset=(.01, 0, 0.04)),
          NestSelection(inst_offset=(0, 0, 0.075))]


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
        self.operations = vacuum
        for op in self.operations:
            op.set_wf(self)

        self.done_waypoints = []
        self.done_waypoint_names = []

        self.lock = False  # block updating with new images
        self.done = False

        self.rgb = None
        self.depth = None
        self.pos = None
        self.orn = None
        self.T_world_tcp = None

        self.cam = cam
        self.T_tcp_cam = T_tcp_cam

        self.io_log = []

    def __del__(self):
        #cv2.destroyWindow("depth")
        cv2.destroyWindow("image")

    def step(self, rgb, depth, pos, orn, T_world_tcp):
        # update state info (live mode)
        if self.lock:
            return

        #cv2.imshow("depth", depth / depth.max())
        cv2.imshow("image", rgb[:, :, ::-1])
        cv2.waitKey(1)

        # observe new frame
        self.rgb = rgb
        self.depth = depth
        self.pos = pos
        self.orn = orn
        self.T_world_tcp = T_world_tcp

    # action in this case means image level operation
    def dispatch_op(self):
        operation = self.operations[self.index]
        operation.dispatch()
        self.next_op()

    def next_op(self):
        # step to the next waypoint
        if self.index is None:
            print("-----------Starting Click Policy-------------")
            self.index = 0
        else:
            if self.index == len(self.operations) - 1:
                print("-----------Completed Click Policy-------------")
                self.lock = False
                self.done = True
                return
            self.index += 1

        operation = self.operations[self.index]
        if operation.instructions:
            print(operation.instructions)

    def add_waypoint(self, pos, orn, grip, name=""):
        self.done_waypoints.append((tuple(pos), tuple(orn), grip))
        self.done_waypoint_names.append(name)

    def callback(self, event, x, y, flags, param):
        """
        Interface callback, send event to actions
        """
        self.io_log.append( (event, x, y, flags, param) )

        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.lock:
                self.lock = True
                self.next_op()

            #self.clicked_points.append((x, y))

            for op in self.operations[self.index:]:
                clicks = op.clicks_req
                if clicks == 0:
                    self.dispatch_op()
                    continue
                else:
                    done, click_next = op.clicked((event, x, y, flags, param))
                    if done:
                        print("clicked", (x, y), "dispatching", op.name)
                        self.dispatch_op()
                    assert click_next is None
                break

    def save_io(self, fn):
        """
        Takes filename ending in .npz -> _wfio.pkl

        Arguments:
            fn: str ending in .npz
        """
        io_fn = fn.replace(".npz","_wfio.pkl")
        with open(io_fn, "wb") as fo:
            pickle.dump(self.io_log, fo)

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


    vacuum_wps = [((0.47, 0.08, 0.26), (1, 0, 0, 0), 1), ((0.5621555602802504, 0.12391772919540847, 0.22065518706705803), (0.9996504730101011, 0.014195653999060347, -0.02070310883179757, -0.00829436573336094), 1), ((0.5721555602802504, 0.12391772919540847, 0.172655187067058), (0.9996504730101011, 0.014195653999060347, -0.02070310883179757, -0.00829436573336094), 1), ((0.5721555602802504, 0.12391772919540847, 0.16065518706705803), (0.9996504730101011, 0.014195653999060347, -0.02070310883179757, -0.00829436573336094), 0), ((0.5721555602802504, 0.12391772919540847, 0.30065518706705807), (0.9996504730101011, 0.014195653999060347, -0.02070310883179757, -0.00829436573336094), 0), ((0.39030870922041905, -0.13124625763610404, 0.25444003733583315), (0.9996504730101011, 0.014195653999060347, -0.02070310883179757, -0.00829436573336094), 0), ((0.39030870922041905, -0.13124625763610404, 0.18944003733583314), (0.9996504730101011, 0.014195653999060347, -0.02070310883179757, -0.00829436573336094), 0), ((0.39030870922041905, -0.13124625763610404, 0.13444003733583315), (0.9996504730101011, 0.014195653999060347, -0.02070310883179757, -0.00829436573336094), 1)]
    assert wf.done_waypoints == vacuum_wps
    print(wf.done_waypoint_names)

    wf.save_io(env.file)
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
    1. Include waypoint names in recording
    2. Continouus segmentation of demo
    3. Auto-Segment
        a. use scroll wheel color/depth, left click outside?
        b. auto ground-plane detection
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
