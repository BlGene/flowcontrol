import numpy as np

from scipy.spatial.transform import Rotation as R
import getpass

from gym_grasping.flow_control.servoing_fitting import solve_transform
from gym_grasping.flow_control.flow_module import FlowModule
from gym_grasping.flow_control.live_plot import ViewPlots

from pdb import set_trace

import time
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' %
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed

class ServoingModule:
    def __init__(self, recording, episode_num = 0, start_index=0,
                 control_config=None, camera_calibration = None,
                 plot=False, opencv_input=False):
        # load files
        username = getpass. getuser()
        if username == "argusm":
            folder_format = "MAX"
        else:
            import cv2
            folder_format = "LUKAS"

        # load files
        if folder_format == "MAX":
            state_recording_fn = "./{}/episode_{}.npz".format(recording, episode_num)
            flow_recording_fn = "./{}/episode_{}_img.npz".format(recording, episode_num)
            mask_recording_fn = "./{}/episode_{}_mask.npz".format(recording, episode_num)
            keep_recording_fn = "{}/episode_{}_keep.npz".format(recording, episode_num)
            state_recording = np.load(state_recording_fn)
            rgb_recording = np.load(flow_recording_fn)["rgb"]
            mask_recording = np.load(mask_recording_fn)["mask"]
            ee_positions = state_recording["ee_positions"]
            gr_positions = state_recording["gripper_states"]
            episode_len = ee_positions.shape[0]
            depth_recording = [None, ]*episode_len
        else:
            flow_recording_fn = "{}/episode_{}.npz".format(recording, episode_num)
            mask_recording_fn = "{}/episode_{}_mask.npz".format(recording, episode_num)
            keep_recording_fn = "{}/episode_{}_keep.npz".format(recording, episode_num)
            rgb_recording = np.load(flow_recording_fn)["rgb_unscaled"]
            try:
                mask_recording = np.load(mask_recording_fn)["mask"]
            except FileNotFoundError:
                mask_recording = np.ones(rgb_recording.shape[0:3]).astype('bool')
            state_recording_fn = "{}/episode_{}.npz".format(recording, episode_num)
            state_recording = np.load(state_recording_fn)["robot_state_full"]
            depth_recording = np.load(state_recording_fn)["depth_imgs"]
            ee_positions = state_recording[:, :3]
            gr_positions = (state_recording[:, -2] > 0.066).astype('float')
            #gr_positions = (np.load(state_recording_fn)["actions"][:, -1] + 1) / 2.0

        # function variables
        self.start_index = start_index
        self.camera_calibration = camera_calibration
        self.opencv_input = opencv_input

        try:
            keep_array = np.load(keep_recording_fn)["keep"]
            print("INFO: loading saved keep frames.")
        except FileNotFoundError:
            keep_array = np.ones(rgb_recording.shape[0])

        try:
            self.keyframes = np.load(keep_recording_fn)["key"]
            print("INFO: loading saved keyframes.")
        except FileNotFoundError:
            self.keyframes = []

        keep_indexes = np.where(keep_array)[0]
        self.keep_indexes = keep_indexes
        self.cur_index = start_index

        self.rgb_recording = rgb_recording
        self.depth_recording = depth_recording
        self.mask_recording = mask_recording
        self.ee_positions = ee_positions
        self.gr_positions = gr_positions
        self.null_action = [0, 0, 0, 0, 1]

        self.max_demo_frame = rgb_recording.shape[0] - 1
        size = rgb_recording.shape[1:3]
        self.size = size

        # load flow net (needs image size)
        print("Image shape from recording", size)
        self.flow_module = FlowModule(size=size)

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
        for k,v in control_config.items():
            assert(hasattr(self, k) == False)
            self.__setattr__(k,v)

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

    def set_base_frame(self):
        # check if the current base_frame is a keyframe, in that case se the keyframe_counter
        # so that the next few steps remain stable
        if self.base_frame in self.keyframes:
            self.keyframe_counter = self.keyframe_counter_max
        self.base_frame = self.keep_indexes[np.clip(self.cur_index, 0, len(self.keep_indexes) - 1)]
        self.base_image_rgb = self.rgb_recording[self.base_frame]
        self.base_image_depth = self.depth_recording[self.base_frame]
        self.base_mask = self.mask_recording[self.base_frame]
        self.base_pos = self.ee_positions[self.base_frame]
        self.grip_state = float(self.gr_positions[self.base_frame])



    def reset(self):
        self.counter = 0
        self.cur_index = self.start_index
        self.set_base_frame()
        if self.view_plots:
            self.view_plots.reset()

    def generate_pointcloud(self, rgb_image, depth_image, masked_points):
        assert(self.camera_calibration)
        assert(self.camera_calibration["width"] == rgb_image.shape[1])
        assert(self.camera_calibration["height"] == rgb_image.shape[0])

        C_X = self.camera_calibration["ppx"]
        C_Y = self.camera_calibration["ppy"]
        FOC_X = self.camera_calibration["fx"]
        FOC_Y = self.camera_calibration["fy"]

        l = len(masked_points)
        u, v = masked_points[:,0], masked_points[:,1]
        #mask_u = np.where(np.logical_or(u < 0, u >= rgb_image.shape[1]))[0]
        #mask_v = np.where(np.logical_or(v < 0, v >= rgb_image.shape[1]))[0]
        # we save positions outside of bounds to set z values to 0
        mask_u = np.logical_or(u < 0, u >= rgb_image.shape[0])
        mask_v = np.logical_or(v < 0, v >= rgb_image.shape[1])
        mask_uv = np.logical_not(np.logical_or(mask_u, mask_v))
        u = np.clip(u, 0, rgb_image.shape[0] - 1)
        v = np.clip(v, 0, rgb_image.shape[1] - 1)

        Z = depth_image[u, v] * mask_uv

        color_new = rgb_image[u, v]
        X = (v - C_X) * Z / FOC_X
        Y = (u - C_Y) * Z / FOC_Y
        pointcloud = np.stack((X, Y, Z, np.ones(l),
                               color_new[:,0], color_new[:,1], color_new[:,2]),
                              axis=1)
        return pointcloud


    def done(self):
        pass
        #if counter >= self.

    def step(self, live_rgb, ee_pos, live_depth=None):
        # 1. compute flow
        # 2. compute transformation
        # 3. transformation to dof
        assert(live_rgb.shape == self.base_image_rgb.shape)

        # Control computation
        flow = self.flow_module.step(self.base_image_rgb, live_rgb)

        if self.mode in ("pointcloud", "pointcloud-abs"):
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
            #mask_pc = np.logical_and(mask_pc, np.random.random(mask_pc.shape[0]) > .99)

            start_pc = start_pc[mask_pc]
            end_pc = end_pc[mask_pc]

            #T_tcp_cam = np.eye(4)
            # transform into TCP coordinates
            start_pc[:, 0:4] = (T_tcp_cam @ start_pc[:, 0:4].T).T
            end_pc[:, 0:4] = (T_tcp_cam @ end_pc[:, 0:4].T).T
            T_tp_t = solve_transform(start_pc[:, 0:4], end_pc[:, 0:4])
            # --- end copy from notebook ---
            guess = T_tp_t
            rot_z = R.from_dcm(guess[:3,:3]).as_euler('xyz')[2]
            # magical gain values for dof, these could come from calibration


            # change names
            if self.mode == "pointcloud":
                move_xy = self.gain_xy*guess[0,3], -1*self.gain_xy*guess[1,3]
                move_z = self.gain_z*(self.base_pos[2] - ee_pos[2])
                move_rot = -self.gain_r*rot_z
                action = [move_xy[0], move_xy[1], move_z, move_rot, self.grip_state]

            elif self.mode == "pointcloud-abs":
                move_xy = self.gain_xy * guess[0, 3], -1 * self.gain_xy * guess[1, 3]
                move_z = self.gain_z * (self.base_pos[2] - ee_pos[2])
                move_rot = -self.gain_r * rot_z

                T_EE = np.eye(4)
                T_EE[:3, :3] = R.from_euler('xyz', ee_pos[3:6]).as_dcm()
                T_EE[:3, 3] = ee_pos[:3]
                T_EE_new = T_EE @ np.linalg.inv(guess)

                xyz_abs = T_EE_new[:3, 3]
                rot_z_abs = R.from_dcm(T_EE_new[:3, :3]).as_euler('xyz')[2]
                action = [xyz_abs[0], xyz_abs[1], xyz_abs[2], rot_z_abs, self.grip_state]

            loss_xy = np.linalg.norm(move_xy)
            loss_z = np.abs(move_z)/3
            loss_rot = np.abs(move_rot) * 3
            loss = loss_xy + loss_rot + loss_z

        elif self.mode == "flat":
            #select foreground
            points = np.array(np.where(self.base_mask)).astype(np.float32)
            size = self.size
            points = (points - ((np.array(size)-1)/2)[:, np.newaxis]).T  # [px]
            points[:, 1]*=-1  # explain this
            observations = points+flow[self.base_mask]  # units [px]
            points = np.pad(points, ((0, 0), (0, 2)), mode="constant")
            observations = np.pad(observations, ((0, 0), (0, 2)), mode="constant")
            guess = solve_transform(points, observations)
            rot_z = R.from_dcm(guess[:3, :3]).as_euler('xyz')[2]  # units [r]
            pos_diff = self.base_pos - ee_pos
            

           # gain values for control, these could come form calibration
            move_xy = -self.gain_xy*guess[0,3]/size[0], self.gain_xy*guess[1,3]/size[1]
            move_z = self.gain_z * pos_diff[2]
            move_rot = -self.gain_r*rot_z
            action = [move_xy[0], move_xy[1], move_z, move_rot, self.grip_state]

            loss_xy = np.linalg.norm(move_xy)
            loss_z = np.abs(move_z)/3
            loss_rot = np.abs(move_rot)
            loss = loss_xy + loss_rot + loss_z
        else:
            raise ValueError("unknown mode")

        # output actions in TCP frame
        self.frame = "TCP"

        if not np.all(np.isfinite(action)):
            print("bad action")
            action = self.null_action

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

            action_str = " ".join(['{: .4f}'.format(a) for a in action])
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
                    self.cur_index += 1
                    print(self.cur_index)
                elif k == ord('a'):
                    self.key_pressed = True
                    self.cur_index -= 1
                    print(self.cur_index)
                elif k == ord('c'):
                    if self.mode == "manual":
                        self.mode = "auto"
                    else:
                        self.mode = "manual"
            self.base_frame = np.clip(self.base_frame, 0, 300)

        # demonstration stepping code
        done = False
        if self.keyframe_counter > 0:
            #action[0:2] = [0,0]  # zero x,y
            #action[3] = 0  # zero angle
            action[0:4] = self.null_action[0:4]
            self.keyframe_counter -= 1

        elif loss < self.threshold or self.key_pressed:
            if self.base_frame < self.max_demo_frame:
                # this is basically a step function
                self.cur_index += 1
                self.set_base_frame()
                self.key_pressed = False
                print("demonstration: ", self.base_frame, "/", self.max_demo_frame)

            elif self.base_frame == self.max_demo_frame:
                done = True

        self.counter += 1

        if self.opencv_input:
            return action, guess, self.mode, done
        else:
            return action, guess, done
