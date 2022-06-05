import pickle
from copy import deepcopy

import cv2
import numpy as np

from flow_control.demo.demo_segment_util import mask_color, mask_center


def get_point_in_world_frame(cam, t_tcp_cam, t_world_tcp, depth, clicked_point):
    """convert a clicked point to world coordinates"""
    point_cam_frame = cam.deproject(clicked_point, depth, homogeneous=True)
    if point_cam_frame is None:
        print("No depth measurement at clicked point")
        return None
    point_world_frame = t_world_tcp @ t_tcp_cam @ point_cam_frame
    return point_world_frame[:3]


# Statefull Operations
# try to keep the same API as https://doc.qt.io/qt-5/qtquick-input-mouseevents.html
class BaseOperation:
    """The base class for operations, the take an image and clicks as input"""
    def __init__(self):
        self.wf = None
        self.clicks_req = 0
        self.instructions = None

    def set_wf(self, wf):
        self.wf = wf

    def clicked(self):
        raise NotImplementedError

    def get_name(self):
        return NotImplementedError


class Move(BaseOperation):
    """perform a move operation"""
    def __init__(self, pos, orn, name="move"):
        self.name = name
        self.clicks_req = 0
        self.instructions = None

        self.pos = pos
        self.orn = orn

    def dispatch(self):
        self.wf.add_waypoint(self.pos, self.orn, 1, name=self.name)


class ObjectSelection(BaseOperation):
    """Select an object by clicking on it."""
    def __init__(self, width=64, over_offset=(0, 0, 0.10),
                 preg_offset=(0, 0, 0.01), grip_offset=(0, 0, 0.01)):
        self.name = "object"
        self.clicks_req = 1
        self.instructions = "Click once on the object"
        self.clicked_points = []

        self.width = width

        self.over_offset = over_offset
        self.preg_offset = preg_offset
        self.grip_offset = grip_offset

    def clicked(self, click_info):
        """collect clicks"""
        event, x_pos, y_pos, flags, param = click_info
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append((x_pos, y_pos))
        return False, None

        if len(self.clicked_points) == self.clicks_req:
            return True, None
        print("clicked", (x_pos, y_pos), len(self.clicked_points), self.clicks_req)
        return False, None

    def dispatch(self):
        """dispatch the operation"""
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

        # new_center = compute_center(self.rgb, None, clicked_point, width=self.width)
        # new_center = clicked_point
        # self.wf.rgb[new_center[1], new_center[0]] = (0, 255, 0)

        point_world = get_point_in_world_frame(self.wf.cam, self.wf.t_tcp_cam,
                                               self.wf.t_world_tcp, self.wf.depth, clicked_point)
        self.wf.add_waypoint(point_world + self.over_offset, self.wf.orn, 1, name="grip-over")
        self.wf.add_waypoint(point_world + self.preg_offset, self.wf.orn, 1, name="grip-close")
        self.wf.add_waypoint(point_world + self.grip_offset, self.wf.orn, 1, name="grip-before")
        self.wf.add_waypoint(point_world + self.grip_offset, self.wf.orn, 0, name="grip-after")
        self.clicked_points = []


class NestSelection(BaseOperation):
    """Select a nest by clicking on edges."""
    def __init__(self, over_offset=(0, 0, 0.14), inst_offset=(0, 0, 0.05),
                 release_offset=(0, 0, 0.02)):
        self.name = "nest"
        self.clicks_req = 2
        self.instructions = "Click left and right of the nest"
        self.clicked_points = []

        self.over_offset = over_offset  # above
        self.inst_offset = inst_offset
        self.release_offset = release_offset

    def clicked(self, click_info):
        """collect clicks for an operation."""
        event, x_pos, y_pos, _, _ = click_info
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append((x_pos, y_pos))
        return False, None

        if len(self.clicked_points) == self.clicks_req:
            return True, None

        print(f"clicked ({x}, {y}) {len(self.clicked_points)}/{self.clicks_req}")
        return False, None

    def dispatch(self):
        """
        dispatch an operation.
        """
        assert len(self.clicked_points) == self.clicks_req
        center = np.array(self.clicked_points).mean(axis=0).round().astype(int).tolist()
        # new_center = compute_center(self.wf.rgb, None, center)
        new_center = center

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
        point_world = get_point_in_world_frame(self.wf.cam, self.wf.t_tcp_cam, self.wf.t_world_tcp, depth_m, new_center)

        print("point world", point_world)
        if len(self.wf.done_waypoints) > 0:
            over_wp = deepcopy(self.wf.done_waypoints[-1])
            over_wp.pos = tuple(over_wp.pos + np.array(self.over_offset))
        self.wf.add_waypoint(over_wp.pos, over_wp.orn, over_wp.grip, name="nest-height")
        self.wf.add_waypoint(point_world + self.over_offset, self.wf.orn, 0, name="nest-over")
        self.wf.add_waypoint(point_world + self.inst_offset, self.wf.orn, 0, name="nest-insertion")
        self.wf.add_waypoint(point_world + self.release_offset, self.wf.orn, 1, name="nest-release")
        self.clicked_points = []

# TODO(max): this duplicates task waypoints
class Waypoint:
    """Waypoint class"""
    def __init__(self, pos, orn, grip, name="waypoint"):
        self.pos = pos
        self.orn = orn
        self.grip = grip
        self.name = name

    def to_action(self):
        """convert a waypoint into an absolute action."""
        grip = -1 if self.grip == 0 else 1
        return dict(motion=(self.pos, self.orn, grip), ref='abs')


class WaypointFactory:
    """
    takes a list of operations, yields waypoints.
    """
    def __init__(self, cam, t_tcp_cam, operations):
        self.index = None
        self.operations = operations
        for opr in self.operations:
            opr.set_wf(self)

        self.done_waypoints = []
        self.lock = False  # block updating with new images
        self.done = False

        self.rgb = None
        self.depth = None
        self.pos = None
        self.orn = None
        self.t_world_tcp = None

        self.cam = cam
        self.t_tcp_cam = t_tcp_cam

        self.io_log = []

    def __del__(self):
        # cv2.destroyWindow("depth")
        cv2.destroyWindow("image")

    def step(self, rgb, depth, pos, orn, t_world_tcp):
        """step the waypoint factory"""
        # update state info (live mode)
        if self.lock:
            return

        # cv2.imshow("depth", depth / depth.max())
        cv2.imshow("image", rgb[:, :, ::-1])
        cv2.waitKey(1)

        # observe new frame
        self.rgb = rgb
        self.depth = depth
        self.pos = pos
        self.orn = orn
        self.t_world_tcp = t_world_tcp

    # action in this case means image level operation
    def displatch_opr(self):
        """dispatch an operation"""
        operation = self.operations[self.index]
        operation.dispatch()
        self.next_op()

    def next_op(self):
        """select the next operation"""
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

    def add_waypoint(self, pos, orn, grip, name=None):
        """add a waypoint"""
        if name is None:
            name = f"waypoint_{len(self.done_waypoints)}"
        wp = Waypoint(pos, orn, grip, name)
        self.done_waypoints.append(wp)

    def callback(self, event, x, y, flags, param):
        """
        Interface callback, send event to actions
        """
        self.io_log.append((event, x, y, flags, param))

        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.lock:
                self.lock = True
                self.next_op()

            # self.clicked_points.append((x, y))

            for opr in self.operations[self.index:]:
                clicks = opr.clicks_req
                if clicks == 0:
                    self.displatch_opr()
                    continue

                done, click_next = opr.clicked((event, x, y, flags, param))
                if done:
                    print("clicked", (x, y), "dispatching", opr.name)
                    self.displatch_opr()
                assert click_next is None
                break

    def save_io(self, rec_fn):
        """
        Takes filename ending in .npz -> _wfio.pkl

        Arguments:
            fn: str ending in .npz
        """
        io_fn = rec_fn.replace(".npz", "_wfio.pkl")
        with open(io_fn, "wb") as f_obj:
            pickle.dump(self.io_log, f_obj)
