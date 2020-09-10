"""
This is a stateful module that contains a recording, then
given a  query RGB-D image it outputs the estimated relative
pose. This module also handels incrementing alog the recording.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R
from gym_grasping.flow_control.servoing_live_plot import ViewPlots
from gym_grasping.flow_control.servoing_fitting import solve_transform
from gym_grasping.flow_control.rgbd_camera import RGBDCamera


class ServoingModule(RGBDCamera):
    """
    This is a stateful module that contains a recording, then
    given a  query RGB-D image it outputs the estimated relative
    pose. This module also handels incrementing alog the recording.
    """

    def __init__(self, recording, episode_num=0, start_index=0,
                 control_config=None, camera_calibration=None,
                 plot=False, opencv_input=False):
        # Moved here because this can require caffe
        from gym_grasping.flow_control.flow_module_flownet2 import FlowModule
        # from gym_grasping.flow_control.flow_module_IRR import FlowModule
        # from gym_grasping.flow_control.reg_module_FGR import RegistrationModule
        RGBDCamera.__init__(self, camera_calibration)
        self.mode = None
        self.step_log = None
        self.frame = None
        self.start_index = start_index
        if isinstance(recording, str):
            # load files, format according to lukas' recorder.
            demo_dict = self.load_demo_from_files(recording, episode_num)
            self.set_demo(demo_dict, reset=False)
        else:
            # force to load something because of FlowNet size etc.
            demo_dict = recording
            self.set_demo(demo_dict, reset=False)

        # function variables
        self.cur_index = start_index
        self.opencv_input = opencv_input
        self.null_action = [0, 0, 0, 0, 1]

        self.max_demo_frame = self.rgb_recording.shape[0] - 1
        size = self.rgb_recording.shape[1:3]
        self.size = size

        # load flow net (needs image size)
        self.flow_module = FlowModule(size=size)
        self.method_name = self.flow_module.method_name
        # self.reg_module = RegistrationModule()
        # self.method_name = "FGR"

        # default config dictionary
        def_config = dict(mode="pointcloud",
                          gain_xy=100,
                          gain_z=50,
                          gain_r=30,
                          threshold=0.20)

        if control_config is None:
            config = def_config
        else:
            config = control_config

        # bake members into class
        for key, val in config.items():
            assert hasattr(self, key) is False or getattr(self, key) is None
            setattr(self, key, val)

        # ignore keyframes for now
        if np.any(self.keyframes):
            self.keyframe_counter_max = 10
        else:
            self.keyframes = set([])
        self.keyframe_counter = 0

        if plot:
            self.view_plots = ViewPlots(threshold=self.threshold)
        else:
            self.view_plots = False
        self.key_pressed = False

        # select frame
        self.counter = 0
        # declare variables
        self.base_frame = None
        self.base_image_rgb = None
        self.base_image_depth = None
        self.base_mask = None
        self.base_pos = None
        self.grip_state = None

        self.reset()

    @staticmethod
    def load_demo_from_files(recording, episode_num):
        """
        load a demo from files.
        """
        ep_num = episode_num
        recording_fn = "{}/episode_{}.npz".format(recording, ep_num)
        mask_recording_fn = "{}/episode_{}_mask.npz".format(recording, ep_num)
        keep_recording_fn = "{}/episode_{}_keep.npz".format(recording, ep_num)
        # load data
        recording_obj = np.load(recording_fn)
        rgb_recording = recording_obj["rgb_unscaled"]
        depth_recording = recording_obj["depth_imgs"]
        state_recording = recording_obj["robot_state_full"]
        print(state_recording[:, -2])
        # ee_positions = state_recording[:, :3]
        # gr_positions = (state_recording[:, -2] > 0.066).astype('float')
        # gr_positions = (recording_obj["actions"][:, -1] + 1) / 2.0
        try:
            mask_recording = np.load(mask_recording_fn)["mask"]
        except FileNotFoundError:
            mask_recording = np.ones(rgb_recording.shape[0:3]).astype(bool)
        try:
            keep_array = np.load(keep_recording_fn)["keep"]
            print("INFO: loading saved keep frames.")
        except FileNotFoundError:
            keep_array = np.ones(rgb_recording.shape[0])
        # note the keyframe option is not being used.
        try:
            keyframes = np.load(keep_recording_fn)["key"]
            print("INFO: loading saved keyframes.")
        except FileNotFoundError:
            keyframes = []

        return dict(rgb=rgb_recording,
                    depth=depth_recording,
                    state=state_recording,
                    mask=mask_recording,
                    keep=keep_array,
                    key=keyframes)

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
        ee_positions = state_recording[:, :3]
        gr_positions = (state_recording[:, -2] > 0.068).astype('float')
        self.ee_positions = ee_positions
        self.gr_positions = gr_positions

        if "key" in demo_dict:
            keyframes = demo_dict["key"]
        else:
            keyframes = []
        self.keyframes = keyframes

        if reset:
            self.reset()

    def set_base_frame(self):
        """
        set a base frame from which to do the servoing
        """
        # check if the current base_frame is a keyframe, in that case se
        # the keyframe_counter so that the next few steps remain stable
        if self.base_frame in self.keyframes:
            self.keyframe_counter = self.keyframe_counter_max
        self.base_frame = self.keep_indexes[np.clip(self.cur_index, 0, len(self.keep_indexes) - 1)]
        self.base_image_rgb = self.rgb_recording[self.base_frame]
        self.base_image_depth = self.depth_recording[self.base_frame]
        self.base_mask = self.mask_recording[self.base_frame]
        self.base_pos = self.ee_positions[self.base_frame]
        self.grip_state = float(self.gr_positions[self.base_frame])

    def reset(self):
        """
        reset servoing, reset counter and index
        """
        self.counter = 0
        self.cur_index = self.start_index
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
        # this should probably be (480, 640, 3)
        assert live_rgb.shape == self.base_image_rgb.shape
        # 1. compute flow
        flow = self.flow_module.step(self.base_image_rgb, live_rgb)
        # 2. compute transformation
        # for compatibility with notebook.
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
        # mask_pc = np.logical_and(mask_pc,
        #                          np.random.random(mask_pc.shape[0]) > .99)
        start_pc = start_pc[mask_pc]
        end_pc = end_pc[mask_pc]
        # transform into TCP coordinates
        # T_tcp_cam = np.array([
        #     [0.99987185, -0.00306941, -0.01571176, 0.00169436],
        #     [-0.00515523, 0.86743151, -0.49752989, 0.11860651],
        #     [0.015156, 0.49754713, 0.86730453, -0.18967231],
        #     [0., 0., 0., 1.]])
        # for calibration make sure that realsense image is rotated 180 degrees (flip_image=True)
        # fingers are in the upper part of the image
        T_tcp_cam = np.array([[9.99801453e-01, -1.81777984e-02,  8.16224931e-03, 2.77370419e-03],
                              [1.99114100e-02,  9.27190979e-01, -3.74059384e-01, 1.31238638e-01],
                              [-7.68387855e-04,  3.74147637e-01,  9.27368835e-01, -2.00077483e-01],
                              [0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

        start_pc[:, 0:4] = (T_tcp_cam @ start_pc[:, 0:4].T).T
        end_pc[:, 0:4] = (T_tcp_cam @ end_pc[:, 0:4].T).T
        T_tp_t = solve_transform(start_pc[:, 0:4], end_pc[:, 0:4])

        guess = T_tp_t

        self.cache_flow = flow
        return guess

    def get_transform_flat(self, live_rgb, ee_pos):
        """
        get a transfromation from 2D, non pointcloud, data.
        """
        flow = self.flow_module.step(self.base_image_rgb, live_rgb)
        # select foreground
        points = np.array(np.where(self.base_mask)).astype(np.float32)
        size = self.size
        points = (points - ((np.array(size)-1)/2)[:, np.newaxis]).T  # [px]
        points[:, 1] *= -1  # explain this
        observations = points+flow[self.base_mask]  # units [px]
        points = np.pad(points, ((0, 0), (0, 2)), mode="constant")
        observations = np.pad(observations, ((0, 0), (0, 2)),
                              mode="constant")
        guess = solve_transform(points, observations)
        return guess

    def step(self, live_rgb, ee_pos, live_depth=None):
        """
        step the servoing policy.

        1. compute transformation
        2. transformation to action
        3. compute loss
        """
        if self.mode in ("pointcloud", "pointcloud-abs"):
            guess = self.get_transform_pc(live_rgb, ee_pos, live_depth)
            rot_z = R.from_dcm(guess[:3, :3]).as_euler('xyz')[2]
            # magical gain values for dof, these could come from calibration
            # change names
            if self.mode == "pointcloud":
                move_xy = self.gain_xy*guess[0, 3], -1*self.gain_xy*guess[1, 3]
                move_z = self.gain_z*(self.base_pos[2] - ee_pos[2])
                move_rot = -self.gain_r*rot_z
                action = [move_xy[0], move_xy[1], move_z, move_rot,
                          self.grip_state]

            elif self.mode == "pointcloud-abs":
                move_xy = self.gain_xy*guess[0, 3], -self.gain_xy*guess[1, 3]
                move_z = self.gain_z*(self.base_pos[2] - ee_pos[2])
                move_rot = self.gain_r*rot_z
                action = [move_xy[0], move_xy[1], move_z, move_rot,
                          self.grip_state]

            loss_xy = np.linalg.norm(move_xy)
            loss_z = np.abs(move_z)/3
            loss_rot = np.abs(move_rot) * 3
            loss = loss_xy + loss_rot + loss_z

        elif self.mode == "flat":
            raise NotImplementedError
#            guess = self.get_transform_flat(live_rgb, ee_pos)
#            rot_z = R.from_dcm(guess[:3, :3]).as_euler('xyz')[2]  # units [r]
#            pos_diff = self.base_pos - ee_pos
#            # gain values for control, these could come form calibration
#            move_xy = (-self.gain_xy*guess[0, 3]/size[0],
#                       self.gain_xy*guess[1, 3]/size[1])
#            move_z = self.gain_z * pos_diff[2]
#            move_rot = -self.gain_r*rot_z
#            action = [move_xy[0], move_xy[1], move_z, move_rot,
#                      self.grip_state]
#            loss_xy = np.linalg.norm(move_xy)
#            loss_z = np.abs(move_z)/3
#            loss_rot = np.abs(move_rot)
#            loss = loss_xy + loss_rot + loss_z
        else:
            raise ValueError("unknown mode")

        # output actions in TCP frame
        self.frame = "TCP"
        if not np.all(np.isfinite(action)):
            print("bad action")
            action = self.null_action

        print("loss", loss, "demo z", self.base_pos[2], "live z", ee_pos[2], "action:", action)
        if self.view_plots:
            series_data = (loss, self.base_frame, ee_pos[0], ee_pos[0])
            self.view_plots.step(series_data, live_rgb, self.base_image_rgb, self.cache_flow, self.base_mask,
                                 [0, 0, 0, 0, 1])

        self.step_log = dict(base_frame=self.base_frame,
                             loss=loss,
                             action=action)

        # demonstration stepping code
        done = False
        if self.keyframe_counter > 0:
            # action[0:2] = [0,0]  # zero x,y
            # action[3] = 0  # zero angle
            action[0:4] = self.null_action[0:4]
            self.keyframe_counter -= 1
        elif loss < self.threshold or self.key_pressed:
            if self.base_frame < self.max_demo_frame:
                # this is basically a step function
                self.cur_index += 1
                self.set_base_frame()
                self.key_pressed = False
                print("demonstration: ", self.base_frame, "/",
                      self.max_demo_frame)
            elif self.base_frame == self.max_demo_frame:
                done = True

        self.counter += 1
        if self.opencv_input:
            return action, guess, self.mode, done
        return action, guess, done
