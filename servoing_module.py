import numpy as np

from scipy.spatial.transform import Rotation as R
import getpass

from gym_grasping.flow_control.servoing_fitting import solve_transform
from gym_grasping.flow_control.flow_module import FlowModule
from gym_grasping.flow_control.live_plot import ViewPlots

from pdb import set_trace


class ServoingModule:
    def __init__(self, recording, episode_num = 0, base_index=0, threshold=.35,
                 camera_calibration = None,
                 plot=False, opencv_input=False):
        username = getpass. getuser()
        if username == "argusm":
            folder_format = "MAX"
        else:
            import cv2
            folder_format = "LUKAS"

        # load files
        if folder_format == "MAX":
            flow_recording_fn = "./{}/episode_1_img.npz".format(recording)
            mask_recording_fn = "./{}/episode_1_mask.npz".format(recording)
            state_recording_fn = "./{}/episode_1.npz".format(recording)
            rgb_recording = np.load(flow_recording_fn)["img"]
            mask_recording = np.load(mask_recording_fn)["mask"]
            state_recording = np.load(state_recording_fn)
            ee_positions = state_recording["ee_positions"]
            gr_positions = state_recording["gripper_states"]
            episode_len = ee_positions.shape[0]
            keep_array = np.ones(episode_len)
            depth_recording = [None,]*episode_len
        else:
            flow_recording_fn = "{}/episode_{}.npz".format(recording, episode_num)
            mask_recording_fn = "{}/episode_{}_mask.npz".format(recording, episode_num)
            keep_recording_fn = "{}/episode_{}_keep.npz".format(recording, episode_num)

            rgb_recording = np.load(flow_recording_fn)["rgb_unscaled"]
            mask_recording = np.load(mask_recording_fn)["mask"]
            state_recording_fn = "{}/episode_{}.npz".format(recording, episode_num)
            state_recording = np.load(state_recording_fn)["robot_state_full"]
            depth_recording = np.load(state_recording_fn)["depth_imgs"]
            ee_positions = state_recording[:, :3]
            # gr_positions = (state_recording[:, -2] > 0.04).astype('float')
            gr_positions = (np.load(state_recording_fn)["actions"][:, -1] + 1) / 2.0
            keep_array = np.load(keep_recording_fn)["keep"]

        keep_indexes = np.where(keep_array)[0]
        self.keep_indexes = keep_indexes
        self.base_index = base_index
        base_frame = keep_indexes[self.base_index]

        self.rgb_recording = rgb_recording
        self.depth_recording = depth_recording
        self.mask_recording = mask_recording
        self.ee_positions = ee_positions
        self.gr_positions = gr_positions

        #self.select_keyframes()
        #self.keyframe_counter_max = 20
        #self.keyframe_counter = self.keyframe_counter_max
        self.keyframes = set([])

        # select frame
        self.set_base_frame(base_frame)
        self.threshold = threshold
        self.max_demo_frame = rgb_recording.shape[0] - 1
        size = rgb_recording.shape[1:3]
        self.size = size
        self.camera_calibration = camera_calibration

        # load flow net (needs image size)
        print("Image shape from recording", size)
        self.flow_module = FlowModule(size=size)

        self.counter = 0
        if plot:
            self.view_plots = ViewPlots(threshold=threshold)
        else:
            self.view_plots = False

        self.opencv_input = opencv_input
        self.key_pressed = False
        self.mode = "auto"

    def set_base_frame(self, base_frame):
        self.base_frame = base_frame
        self.base_image_rgb = self.rgb_recording[base_frame]
        self.base_image_depth = self.depth_recording[base_frame]
        self.base_mask = self.mask_recording[base_frame]
        self.base_pos = self.ee_positions[base_frame]
        self.grip_state = self.gr_positions[base_frame]

    def select_keyframes(self, off=False):
        """
        This selects a set of keyframes for which we want to iterate for longer.
        For now these should be the frames before the gripper is actuated.
        """
        self.keyframes = np.where(np.diff(self.gr_positions))[0]

    def reset(self):
        self.set_base_frame(self.base_index)

    def generate_pointcloud(self, rgb_image, depth_image, masked_points):
        assert(self.camera_calibration)
        assert(self.camera_calibration["width"] == rgb_image.shape[0])
        assert(self.camera_calibration["height"] == rgb_image.shape[1])

        C_X = self.camera_calibration["ppx"]
        C_Y = self.camera_calibration["ppy"]
        FOC_X = self.camera_calibration["fx"]
        FOC_Y = self.camera_calibration["fy"]

        pointcloud = []
        l = len(masked_points)
        u, v = masked_points[:,0], masked_points[:,1]
        Z = depth_image[u, v]
        color_new = rgb_image[u, v]
        X = (v - C_X) * Z / FOC_X
        Y = (u - C_Y) * Z / FOC_Y
        pointcloud = np.stack((X, Y, Z, np.ones(l),
                               color_new[:,0], color_new[:,1], color_new[:,2]),
                              axis=1)
        return pointcloud

    def step(self, live_rgb, ee_pos, live_depth=None):
        # 1. compute flow
        # 2. compute transformation
        # 3. transformation to control
        assert(live_rgb.shape == self.base_image_rgb.shape)

        # Control computation
        flow = self.flow_module.step(self.base_image_rgb, live_rgb)

        mode = "flat"
        if mode == "pointcloud":
            # for compatibility with notebook.
            demo_rgb = self.base_image_rgb
            demo_depth = self.base_image_depth

            end_points = np.array(np.where(self.base_mask)).T
            masked_flow = flow[self.base_mask]
            start_points = end_points + masked_flow[:, ::-1].astype('int')

            T_tcp_cam = np.array([
                [0.99987185, -0.00306941, -0.01571176, 0.00169436],
                [-0.00515523, 0.86743151, -0.49752989, 0.11860651],
                [0.015156, 0.49754713, 0.86730453, -0.18967231],
                [0., 0., 0., 1.]])

            if live_depth is None and demo_depth is None:
                live_depth = ee_pos[2] * np.ones(live_rgb.shape[0:2])
                demo_depth = ee_pos[2] * np.ones(live_rgb.shape[0:2])

            if live_depth is not None and demo_depth is None:
                demo_depth = live_depth - ee_pos[2] + self.base_pos[2]

            start_pc = self.generate_pointcloud(live_rgb, live_depth, start_points)
            end_pc = self.generate_pointcloud(demo_rgb, demo_depth, end_points)

            mask_pc = np.logical_and(start_pc[:, 2] != 0, end_pc[:, 2] != 0)
            #mask_pc = np.logical_and(mask_pc, np.random.random(mask_pc.shape[0]) > .95)

            start_pc = start_pc[mask_pc]
            end_pc = end_pc[mask_pc]

            # transform into TCP coordinates
            start_pc[:, 0:4] = (T_tcp_cam @ start_pc[:, 0:4].T).T
            end_pc[:, 0:4] = (T_tcp_cam @ end_pc[:, 0:4].T).T
            T_tp_t = solve_transform(start_pc[:, 0:4], end_pc[:, 0:4])
            # --- end copy from notebook ---
            guess = T_tp_t
            rot_z = R.from_dcm(guess[:3,:3]).as_euler('xyz')[2]
            # magical gain values for control, these could come from calibration

            # change names
            gain_xy = 20 # units [action/ norm-coords to -1,1]
            gain_z = 30  # units [action/m]
            gain_r = 10   # units [action/r]

            move_xy = gain_xy*guess[0,3], -1*gain_xy*guess[1,3]
            move_z = gain_z*(self.base_pos - ee_pos)[2]
            move_rot = gain_r*rot_z
            action = [move_xy[0], move_xy[1], move_z, move_rot, self.grip_state]

            loss_xy = np.linalg.norm(move_xy)
            loss_z = np.abs(move_z)/3
            loss_rot = np.abs(move_rot)
            loss = loss_xy + loss_rot + loss_z


        elif mode == "flat":
            size = self.size

            #select foreground
            points = np.array(np.where(self.base_mask)).astype(np.float32)
            points = (points - ((np.array(size)-1)/2)[:, np.newaxis]).T  # [px]
            points[:,1]*=-1  # explain this
            observations = points+flow[self.base_mask]  # units [px]
            points = np.pad(points, ((0, 0), (0, 2)), mode="constant")
            observations = np.pad(observations, ((0, 0), (0, 2)), mode="constant")
            guess = solve_transform(points, observations)
            rot_z = R.from_dcm(guess[:3,:3]).as_euler('xyz')[2]  # units [r]
            pos_diff = self.base_pos - ee_pos

            # gain values for control, these could come form calibration
            gain_xy = 50 # units [action/ norm-coords to -1,1]
            gain_z = 30  # units [action/m]
            gain_r = 7   # units [action/r]
            move_xy = -gain_xy*guess[0,3]/size[0], gain_xy*guess[1,3]/size[1]
            move_z = gain_z * pos_diff[2]
            move_rot = gain_r*rot_z
            action = [move_xy[0], move_xy[1], move_z, move_rot, self.grip_state]

            loss_xy = np.linalg.norm(move_xy)
            loss_z = np.abs(move_z)/3
            loss_rot = np.abs(move_rot)
            loss = loss_xy + loss_rot + loss_z

        else:
            raise ValueError("unknown mode")

        # Outputs of this block: used at beginning of loop
        #   1. mean_flow
        #   2. mean_rot

        if not np.all(np.isfinite(action)):
            print("bad action")
            null_action = [0,0,0,0,1]
            action = null_action

        self.step_log = dict(base_frame=self.base_frame,
                             loss=loss,
                             action=action)

        # plotting code
        if self.view_plots:
            # show flow
            flow_img = self.flow_module.computeImg(flow, dynamic_range=False)
            # show segmentatione edge
            if self.counter % 5 == 0:
                edge  = np.gradient(self.base_mask.astype(float))
                edge = (np.abs(edge[0])+ np.abs(edge[1])) > 0
                flow_img[edge] = (255, 0, 0)

            action_str = " ".join(['{: .2f}'.format(a) for a in action])
            print("loss = {:.4f} {}".format(loss, action_str))
            # show loss, frame number
            self.view_plots.step(loss, self.base_frame, self.base_pos[2], ee_pos[2])
            self.view_plots.low_1_h.set_data(live_rgb)
            self.view_plots.low_2_h.set_data(self.base_image_rgb)
            self.view_plots.low_3_h.set_data(flow_img)
            plot_fn = f'./video/{self.counter:03}.png'
            #plt.savefig(plot_fn, bbox_inches=0)

            if self.opencv_input:
                # depricated, causes error
                cv2.imshow('window', np.zeros((100,100)))
                k = cv2.waitKey(10) % 256
                if k == ord('d'):
                    self.key_pressed = True
                    self.base_frame += 1
                    print(self.base_frame)
                elif k == ord('a'):
                    self.key_pressed = True
                    self.base_frame -= 1
                    print(self.base_frame)
                elif k == ord('c'):
                    if self.mode == "manual":
                        self.mode = "auto"
                    else:
                        self.mode = "manual"
            self.base_frame = np.clip(self.base_frame, 0, 300)

        # demonstration stepping code
        if loss < self.threshold and self.base_frame < self.max_demo_frame or self.key_pressed:
            if self.base_index in self.keyframes:
                if self.keyframe_counter > 0:
                    self.keyframe_counter -= 1
                else:
                    self.base_index += 1
                    self.set_base_frame(self.base_frame)
                    self.keyframe_counter = self.keyframe_counter_max
            else:
                # this is basically a step function
                if not self.key_pressed:
                    self.base_index += 1
                self.base_frame = self.keep_indexes[np.clip(self.base_index,0,len(self.keep_indexes)-1)]
                self.key_pressed = False
                self.set_base_frame(self.base_frame)
                print("demonstration: ", self.base_frame, "/", self.max_demo_frame)

        self.counter += 1
        if self.opencv_input:
            return action, self.mode
        else:
            return action


