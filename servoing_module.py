"""
This is a stateful module that contains a recording, then
given a  query RGB-D image it outputs the estimated relative
pose. This module also handels incrementing alog the recording.
"""
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
from gym_grasping.flow_control.servoing_demo import ServoingDemo
from gym_grasping.flow_control.servoing_live_plot import SubprocPlot
from gym_grasping.flow_control.servoing_fitting import solve_transform
from gym_grasping.flow_control.rgbd_camera import RGBDCamera

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
        from gym_grasping.flow_control.flow_module_flownet2 import FlowModule
        # from gym_grasping.flow_control.flow_module_IRR import FlowModule
        # from gym_grasping.flow_control.reg_module_FGR import RegistrationModule

        self.demo = ServoingDemo(recording, episode_num, start_index)

        # get config dict and bake members into class
        if control_config is None:
            config = DEFAULT_CONF
        else:
            config = control_config
        for key, val in config.items():
            assert hasattr(self, key) is False or getattr(self, key) is None
            setattr(self, key, val)

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
            self.view_plots = SubprocPlot(threshold=self.threshold,
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
        self.action_queue = {}
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
        # TODO(max): remove gripper_wait_steps replace with gripper vel check
        gripper_wait_steps = 1 # 10 # pause servoing and wait on keyframes

        if self.demo.frame - 1 in self.demo.keyframes:
            self.action_queue["change_gripper"] = gripper_wait_steps

        try:
            pre_action = self.demo.keep_dict[self.demo.frame]["pre"]
        except KeyError:
            return

        if "rel" in pre_action:
            delta = pre_action["rel"]
            world_goal = list(cur_robot_state[:3] + delta[:3])
        elif "abs" in pre_action:
            world_goal = pre_action["abs"][:3]
        else:
            world_goal = None

        if world_goal is not None:
            self.action_queue["abs_action"] = world_goal

    def get_action_from_queue(self, action, info):
        if "change_gripper" in self.action_queue:
            action[0:4] = self.null_action[0:4]
            self.action_queue["change_gripper"] -= 1
            if self.action_queue["change_gripper"] == 0:
                del self.action_queue["change_gripper"]

        if "abs_action" in self.action_queue:
            info["abs_action"] = self.action_queue["abs_action"]
            del self.action_queue["abs_action"]
        return action, info

    def step(self, live_rgb, ee_pos, live_depth=None):
        """
        Step the servoing policy, usually what frame alignment gives, but
        sometimes something else.
        """
        action, loss = self.frame_align(live_rgb, ee_pos, live_depth)
        info = {}

        done = False
        ready = len(self.action_queue) == 0
        # demonstration stepping code, this is basically a step function
        if ready and loss < self.threshold:
            if self.demo.frame < self.demo.max_frame:
                self.demo.step()
                self.put_action_in_queue(ee_pos)

            elif self.demo.frame == self.demo.max_frame:
                done = True

        if not ready:
            action, info = self.get_action_from_queue(action, info)

        self.counter += 1
        return action, done, info


    def frame_align(self, live_rgb, ee_pos, live_depth=None):
        """
        step the servoing policy.

        1. compute transformation
        2. transformation to action
        3. compute loss
        """
        if self.mode in ("pointcloud", "pointcloud-abs"):
            guess = self.get_transform_pc(live_rgb, ee_pos, live_depth)
            rot_z = R.from_matrix(guess[:3, :3]).as_euler('xyz')[2]

            if self.mode == "pointcloud":
                move_xy = self.gain_xy*guess[0, 3], -self.gain_xy*guess[1, 3]
                move_z = self.gain_z*(self.demo.state[2] - ee_pos[2])
                move_rot = -self.gain_r*rot_z
                action = [move_xy[0], move_xy[1], move_z, move_rot,
                          self.demo.grip_action]

            elif self.mode == "pointcloud-abs":
                raise NotImplementedError

            loss_xy = np.linalg.norm(move_xy)
            loss_z = np.abs(move_z)/3
            loss_rot = np.abs(move_rot) * 3
            loss = loss_xy + loss_rot + loss_z
        else:
            raise ValueError("unknown mode")

        # debug output
        loss_str = "loss {:4.4f}".format(loss)
        action_str = " action: " + " ".join(['%4.2f' % a for a in action])
        # loss_str += " demo z {:.4f} live z {:.4f}".format(self.state[2],  ee_pos[2])
        logging.debug(loss_str + action_str)

        if self.view_plots:
            series_data = (loss, self.demo.frame, ee_pos[0], ee_pos[0])
            self.view_plots.step(series_data, live_rgb, self.demo.rgb,
                                 self.cache_flow, self.demo.mask, action=None)
        return action, loss

    def get_transform_pc(self, live_rgb, ee_pos, live_depth):
        """
        get a transformation from a pointcloud.
        """
        # There is possibly a duplicate of this in a notebook, remove that if so

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
            live_depth = ee_pos[2] * np.ones(live_rgb.shape[0:2])
            demo_depth = ee_pos[2] * np.ones(live_rgb.shape[0:2])
        if live_depth is not None and demo_depth is None:
            demo_depth = live_depth - ee_pos[2] + self.demo.state[2]

        start_pc = self.generate_pointcloud(live_rgb, live_depth, start_points)
        end_pc = self.generate_pointcloud(self.demo.rgb, demo_depth, end_points)
        mask_pc = np.logical_and(start_pc[:, 2] != 0, end_pc[:, 2] != 0)

        # subsample fitting, maybe evaluate with ransac
        # mask_pc = np.logical_and(mask_pc,
        #                          np.random.random(mask_pc.shape[0]) > .99)
        start_pc = start_pc[mask_pc]
        end_pc = end_pc[mask_pc]

        # 2. transform into TCP coordinates
        # TODO(max): Why can't I transform the estimated transform?
        #            multiplying by a fixed transform should factorize out..
        start_pc[:, 0:4] = (self.T_tcp_cam @ start_pc[:, 0:4].T).T
        end_pc[:, 0:4] = (self.T_tcp_cam @ end_pc[:, 0:4].T).T
        T_tp_t = solve_transform(start_pc[:, 0:4], end_pc[:, 0:4])
        return T_tp_t
