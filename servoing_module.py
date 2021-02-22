"""
This is a stateful module that contains a recording, then
given a  query RGB-D image it outputs the estimated relative
pose. This module also handels incrementing alog the recording.
"""
import json
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
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

        RGBDCamera.__init__(self, camera_calibration)
        self.T_tcp_cam = T_TCP_CAM
        self.start_index = start_index
        self.null_action = [0, 0, 0, 0, 1]

        self.keyframe_counter_max = 1  # 10 # pause servoing and wait on keyframes

        # set in set_demo (don't move down)
        self.rgb_recording = None
        self.depth_recording = None
        self.mask_recording = None
        self.keep_indexes = None
        self.ee_positions = None
        self.gr_actions = None
        self.keyframes = None

        # set in set_base_frame (call before reset)
        self.base_frame = None
        self.base_image_rgb = None
        self.base_image_depth = None
        self.base_mask = None
        self.base_pos = None
        self.grip_action = None

        if isinstance(recording, str):
            demo_dict = self.load_demo_from_files(recording, episode_num)
            self.set_demo(demo_dict, reset=False)
        else:
            # force to load something because of FlowNet size etc.
            demo_dict = recording
            self.set_demo(demo_dict, reset=False)

        self.max_demo_frame = self.rgb_recording.shape[0] - 1
        size = self.rgb_recording.shape[1:3]
        self.size = size

        # load flow net (needs image size)
        self.flow_module = FlowModule(size=size)
        self.method_name = self.flow_module.method_name
        # self.reg_module = RegistrationModule()
        # self.method_name = "FGR"

        # get config dict and bake members into class
        if control_config is None:
            config = DEFAULT_CONF
        else:
            config = control_config
        for key, val in config.items():
            assert hasattr(self, key) is False or getattr(self, key) is None
            setattr(self, key, val)

        # plotting
        self.cache_flow = None
        self.view_plots = False
        if plot:
            self.view_plots = SubprocPlot(threshold=self.threshold,
                                          save_dir=save_dir)
        # vars set in reset
        self.counter = None
        self.cur_index = None
        self.keyframe_counter = None
        self.action_queue = None

        self.reset()

    @staticmethod
    def load_demo_from_files(recording, episode_num):
        """
        load a demo from files.

        Arguments:
            recording: path to recording containing episode_0.npz
            episode_num: integert to select episode
        """
        ep_num = episode_num
        recording_fn = "{}/episode_{}.npz".format(recording, ep_num)
        keep_dict_fn = "{}/episode_{}.json".format(recording, ep_num)
        mask_recording_fn = "{}/episode_{}_mask.npz".format(recording, ep_num)
        keep_recording_fn = "{}/episode_{}_keep.npz".format(recording, ep_num)

        # load data
        recording_obj = np.load(recording_fn)
        rgb_shape = recording_obj["rgb_unscaled"].shape

        try:
            mask_recording = np.load(mask_recording_fn)["mask"]
        except FileNotFoundError:
            logging.warning(f"Couldn't find {mask_recording_fn}, servoing will fail")
            mask_recording = np.ones(rgb_shape[0:3], dtype=bool)

        try:
            keep_array = np.load(keep_recording_fn)["keep"]
            logging.info("loading saved keep frames.")
        except FileNotFoundError:
            logging.warning(f"Couldn't find {keep_recording_fn}, servoing will take ages")
            keep_array = np.ones(rgb_shape[0])

        try:
            with open(keep_dict_fn) as f_obj:
                keep_dict = json.load(f_obj)
                # undo json mangling
                keep_dict = {int(key): val for key, val in keep_dict.items()}

        except FileNotFoundError:
            keep_dict = {}
        try:
            keyframes = np.load(keep_recording_fn)["key"]
            logging.info("loading saved keyframes.")
        except FileNotFoundError:
            keyframes = []

        return dict(rgb=recording_obj["rgb_unscaled"],
                    depth=recording_obj["depth_imgs"],
                    state=recording_obj["robot_state_full"],
                    actions=recording_obj["actions"],
                    mask=mask_recording,
                    keep=keep_array,
                    key=keyframes,
                    keep_dict=keep_dict)

    # TODO(max): reset used by pose estimation eval, default to false or remove
    def set_demo(self, demo_dict, reset=True):
        """
        set a demo that is given as a dictionary, not file
        """
        self.rgb_recording = demo_dict['rgb']
        self.depth_recording = demo_dict["depth"]
        self.mask_recording = demo_dict["mask"]
        keep_array = demo_dict["keep"]
        state_recording = demo_dict["state"]

        self.keep_indexes = np.where(keep_array)[0]
        self.ee_positions = state_recording[:, :3]

        # self.gr_actions = (state_recording[:, -2] > 0.068).astype('float')
        # self.gr_actions = (state_recording[:, -2] > 0.070).astype('float')
        self.gr_actions = demo_dict["actions"][:, 4].astype('float')

        self.keep_dict = demo_dict["keep_dict"]
        # TODO(max): remove keyframe_counter replace with gripper vel check or
        # with a keep_dict entry
        keyframes = []
        if "key" in demo_dict:
            keyframes = demo_dict["key"]
        if not np.any(keyframes):
            keyframes = set([])
        self.keyframes = keyframes

        if reset:
            self.reset()

    # TODO(max): this is based on a new cur_index value, take this as input
    def set_base_frame(self):
        """
        set a base frame from which to do the servoing
        """
        self.base_frame = self.keep_indexes[np.clip(self.cur_index, 0, len(self.keep_indexes) - 1)]
        assert not self.base_frame > self.max_demo_frame

        self.base_image_rgb = self.rgb_recording[self.base_frame]
        self.base_image_depth = self.depth_recording[self.base_frame]
        self.base_mask = self.mask_recording[self.base_frame]
        self.base_pos = self.ee_positions[self.base_frame]
        self.grip_action = float(self.gr_actions[self.base_frame])

        step_str = "start: {} / {}".format(self.base_frame, self.max_demo_frame)
        step_str += " step {} ".format(self.counter)
        logging.debug(step_str + str(self.action_queue))

    def set_keep_action(self, cur_state):
        """
        Call once when stepping the demo.

        Returns:
            (implicitly): updated self.action_queue with abs_action as list
        """
        # check if the current base_frame is a keyframe, in that case se
        # the keyframe_counter so that the next few steps remain stable
        if self.base_frame - 1 in self.keyframes:
            self.action_queue["change_gripper"] = self.keyframe_counter_max

        try:
            pre_action = self.keep_dict[self.base_frame]["pre"]
        except KeyError:
            return

        if "rel" in pre_action:
            # print("We now want to do a relative motion")
            delta = pre_action["rel"]
            world_goal = list(cur_state[:3] + delta[:3])
        elif "abs" in pre_action:
            world_goal = pre_action["abs"][:3]
        else:
            world_goal = None

        if world_goal is not None:
            self.action_queue["abs_action"] = world_goal

    def reset(self):
        """
        reset servoing, reset counter and index
        """
        self.cur_index = self.start_index
        self.counter = 0
        self.keyframe_counter = 0
        self.action_queue = {}

        self.set_base_frame()

        if self.view_plots:
            self.view_plots.reset()

    def done(self):
        """
        servoing is done?
        """
        raise NotImplementedError

    def get_transform_pc(self, live_rgb, ee_pos, live_depth):
        """
        get a transformation from a pointcloud.
        """
        # There is possibly a duplicate of this in a notebook, remove that if so

        # this should probably be (480, 640, 3)
        assert live_rgb.shape == self.base_image_rgb.shape

        # 1. compute flow
        flow = self.flow_module.step(self.base_image_rgb, live_rgb)
        self.cache_flow = flow

        # 2. compute transformation
        demo_rgb = self.base_image_rgb
        demo_depth = self.base_image_depth
        end_points = np.array(np.where(self.base_mask)).T
        masked_flow = flow[self.base_mask]
        start_points = end_points + masked_flow[:, ::-1].astype('int')

        if live_depth is None and demo_depth is None:
            live_depth = ee_pos[2] * np.ones(live_rgb.shape[0:2])
            demo_depth = ee_pos[2] * np.ones(live_rgb.shape[0:2])
        if live_depth is not None and demo_depth is None:
            demo_depth = live_depth - ee_pos[2] + self.base_pos[2]

        start_pc = self.generate_pointcloud(live_rgb, live_depth, start_points)
        end_pc = self.generate_pointcloud(demo_rgb, demo_depth, end_points)
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

    def step(self, live_rgb, ee_pos, live_depth=None):
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
                move_z = self.gain_z*(self.base_pos[2] - ee_pos[2])
                move_rot = -self.gain_r*rot_z
                action = [move_xy[0], move_xy[1], move_z, move_rot,
                          self.grip_action]

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
        # loss_str += " demo z {:.4f} live z {:.4f}".format(self.base_pos[2],  ee_pos[2])
        logging.debug(loss_str + action_str)

        if self.view_plots:
            series_data = (loss, self.base_frame, ee_pos[0], ee_pos[0])
            self.view_plots.step(series_data, live_rgb, self.base_image_rgb,
                                 self.cache_flow, self.base_mask, action=None)

        # demonstration stepping code
        done = False
        if "change_gripper" in self.action_queue:
            action[0:4] = self.null_action[0:4]
            self.action_queue["change_gripper"] -= 1
            if self.action_queue["change_gripper"] == 0:
                del self.action_queue["change_gripper"]

        elif loss < self.threshold:
            if self.base_frame < self.max_demo_frame:
                # TODO(max): this is basically a step function
                self.cur_index += 1
                self.set_base_frame()
                self.set_keep_action(ee_pos)

            elif self.base_frame == self.max_demo_frame:
                done = True

        info = {}  # return for abs action
        if "abs_action" in self.action_queue:
            info["abs_action"] = self.action_queue["abs_action"]
            del self.action_queue["abs_action"]

        self.counter += 1

        return action, guess, done, info
