"""
This is a stateful module that contains a recording, then
given a  query RGB-D image it outputs the estimated relative
pose. This module also handels incrementing alog the recording.
"""
import logging
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation as R
from flow_control.servoing_demo import ServoingDemo
from flow_control.servoing_live_plot import SubprocPlot
from flow_control.servoing_fitting import solve_transform
from flow_control.rgbd_camera import RGBDCamera

from pdb import set_trace

# TODO(max): below is the hardware calibration of T_tcp_cam,
# 1) this is the hacky visual comparison, it needs to be compared to a marker
#    based calibration.
# 2) it needs to be stored with the recording.
# for calibration make sure that realsense image is rotated 180 degrees (flip_image=True)
# i.e. fingers are in the upper part of the image
T_TCP_CAM = np.array([[9.99801453e-01, -1.81777984e-02, 8.16224931e-03, 2.77370419e-03],
                      [1.99114100e-02, 9.27190979e-01, -3.74059384e-01, 1.31238638e-01],
                      [-7.68387855e-04, 3.74147637e-01, 9.27368835e-01, -2.00077483e-01],
                      [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


# magical gain values for dof, these could come from calibration
DEFAULT_CONF = dict(mode="pointcloud",
                    gain_xy=100,
                    gain_z=50,
                    gain_r=30,
                    threshold=0.20)


class ServoingModule(RGBDCamera):
    """
    This is a stateful module that contains a recording, then
    given a  query RGB-D image it outputs the estimated relative
    pose. This module also handels incrementing alog the recording.
    """

    def __init__(self, recording, episode_num=0, start_index=0,
                 control_config=None, camera_calibration=None,
                 plot=False, save_dir=False):
        # Moved here because this can require caffe
        from flow_control.flow_module_flownet2 import FlowModule
        # from flow_control.flow_module_IRR import FlowModule
        # from flow_control.reg_module_FGR import RegistrationModule

        self.demo = ServoingDemo(recording, episode_num, start_index)

        # get config dict and bake members into class
        if control_config is None:
            config = DEFAULT_CONF
        else:
            config = control_config
        self.config = SimpleNamespace(**config)

        RGBDCamera.__init__(self, camera_calibration)

        self.T_tcp_cam = T_TCP_CAM
        self.null_action = [0, 0, 0, 0, 1]
        size = self.demo.rgb_recording.shape[1:3]
        self.size = size

        # load flow net (needs image size)
        self.flow_module = FlowModule(size=size)
        self.method_name = self.flow_module.method_name
        # self.reg_module = RegistrationModule()
        # self.method_name = "FGR"

        # plotting
        self.cache_flow = None
        self.view_plots = False
        if plot:
            self.view_plots = SubprocPlot(threshold=self.config.threshold,
                                          save_dir=save_dir)
        # vars set in reset
        self.counter = None
        self.action_queue = None

        self.reset()

    def reset(self):
        """
        reset servoing, reset counter and index
        """
        self.counter = 0
        self.action_queue = []
        self.demo.reset()
        if self.view_plots:
            self.view_plots.reset()

    def done(self):
        """
        servoing is done?
        """
        raise NotImplementedError

    def put_action_in_queue(self, cur_robot_state):
        """
        Call once when stepping the demo.

        Returns:
            (implicitly): updated self.action_queue with abs_action as list
        """
        try:
            pre_actions = self.demo.keep_dict[self.demo.frame]["pre"]
        except KeyError:
            return

        self.action_queue.append(pre_actions)

    def get_action_from_queue(self, action, info):
        if len(self.action_queue) == 0:
            return action, info

        info = self.action_queue[0]
        del self.action_queue[0]

        return action, info

    def step(self, live_rgb, live_state, live_depth=None):
        """
        Main loop, this does sequence alignment.

        Usually what frame alignment gives, but sometimes something else.

        Arguments:
            live_rgb: live rgb image
            live_state: live state from robot
            live_depth: live depth image

        Returns:
            action: (x, y, z, r, g)
            done: binary if demo sequence is completed
            info: dict
        """
        align_transform = self.frame_align(live_rgb, live_state, live_depth)
        action, loss = self.trf_to_act_loss(align_transform, live_state)

        # debug output
        loss_str = "loss {:4.4f}".format(loss)
        action_str = " action: " + " ".join(['%4.2f' % a for a in action])
        # loss_str += " demo z {:.4f} live z {:.4f}".format(self.state[2],  live_state[2])
        logging.debug(loss_str + action_str)

        if self.view_plots:
            series_data = (loss, self.demo.frame, live_state[0], live_state[0])
            self.view_plots.step(series_data, live_rgb, self.demo.rgb,
                                 self.cache_flow, self.demo.mask, action=None)

        info = {}
        done = False
        ready = len(self.action_queue) == 0

        if ready and loss < self.config.threshold:
            if self.demo.frame < self.demo.max_frame:
                self.demo.step()
                self.put_action_in_queue(live_state)
                # debug
                step_str = "start: {} / {}".format(self.demo.frame, self.demo.max_frame)
                step_str += " step {} ".format(self.counter)
                logging.debug(step_str )

            elif self.demo.frame == self.demo.max_frame:
                done = True

        if not ready:
            action, info = self.get_action_from_queue(action, info)

        self.counter += 1
        return action, done, info

    def trf_to_act_loss(self, align_transform, live_state):
        """
        Arguments:
            align_transform: transform that aligns demo to live
            live_state: current live robot state

        Returns:
            action: currently (x, y, z, r, g)
            loss: scalar ususally between ~5 and ~0.2
        """
        if self.config.mode not in ("pointcloud", "pointcloud-abs"):
            raise ValueError("unknown mode")

        d_x = align_transform[0, 3]
        d_y = align_transform[1, 3]
        rot_z = R.from_matrix(align_transform[:3, :3]).as_euler('xyz')[2]

        if self.config.mode == "pointcloud":
            move_xy = self.config.gain_xy*d_x, -self.config.gain_xy*d_y
            move_z = -1*self.config.gain_z*(live_state[2] - self.demo.state[2])
            move_rot = -self.config.gain_r*rot_z
            move_g = self.demo.grip_action
            action = [*move_xy, move_z, move_rot, move_g]

        elif self.config.mode == "pointcloud-abs":
            raise NotImplementedError

        loss_xy = np.linalg.norm(move_xy)
        loss_z = np.abs(move_z)/3
        loss_rot = np.abs(move_rot) * 3
        loss = loss_xy + loss_rot + loss_z

        return action, loss

    def frame_align(self, live_rgb, live_state, live_depth):
        """
        get a transformation from a pointcloud.
        """
        # this should probably be (480, 640, 3)
        assert live_rgb.shape == self.demo.rgb.shape

        # 1. compute flow
        flow = self.flow_module.step(self.demo.rgb, live_rgb)
        self.cache_flow = flow

        # 2. compute transformation
        demo_depth = self.demo.depth
        end_points = np.array(np.where(self.demo.mask)).T
        masked_flow = flow[self.demo.mask]
        start_points = end_points + masked_flow[:, ::-1].astype('int')

        if live_depth is None and demo_depth is None:
            live_depth = live_state[2] * np.ones(live_rgb.shape[0:2])
            demo_depth = live_state[2] * np.ones(live_rgb.shape[0:2])
        if live_depth is not None and demo_depth is None:
            demo_depth = live_depth - live_state[2] + self.demo.state[2]

        start_pc = self.generate_pointcloud(live_rgb, live_depth, start_points)
        end_pc = self.generate_pointcloud(self.demo.rgb, demo_depth, end_points)
        mask_pc = np.logical_and(start_pc[:, 2] != 0, end_pc[:, 2] != 0)

        # subsample fitting, maybe evaluate with ransac
        # mask_pc = np.logical_and(mask_pc,
        #                          np.random.random(mask_pc.shape[0]) > .99)
        start_pc = start_pc[mask_pc]
        end_pc = end_pc[mask_pc]

        # 3. transform into TCP coordinates
        # TODO(max): Why can't I transform the estimated transform?
        #            multiplying by a fixed transform should factorize out..
        start_pc[:, 0:4] = (self.T_tcp_cam @ start_pc[:, 0:4].T).T
        end_pc[:, 0:4] = (self.T_tcp_cam @ end_pc[:, 0:4].T).T
        T_tp_t = solve_transform(start_pc[:, 0:4], end_pc[:, 0:4])
        return T_tp_t
