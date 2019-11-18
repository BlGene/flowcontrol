import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

from gym_grasping.flow_control.servoing_fitting import solve_transform
from gym_grasping.flow_control.flow_module import FlowModule
from gym_grasping.flow_control.live_plot import ViewPlots

import matplotlib.pyplot as plt

# C_X = 317.2261962890625
# C_Y = 245.13381958007812
C_X = 315.20367431640625
C_Y = 245.70614624023438
# FOC_X = 474.3828125
# FOC_Y = 474.3828430175781
FOC_X = 617.8902587890625
FOC_Y = 617.8903198242188

class ServoingModule:
    def __init__(self, recording, episode_num = 0, base_index=0, threshold=.35, plot=False, opencv_input=False):
        # load files

        folder_format = "MAX"

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
            size = rgb_recording.shape[1:3]
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
            size = rgb_recording.shape[1:3][::-1]

            keep_array = np.load(keep_recording_fn)["keep"]
            keep_indexes = np.where(keep_array)[0]
            self.keep_indexes = keep_indexes
            self.base_index = base_index
            base_frame = keep_indexes[self.base_index]




        self.size = size

        self.rgb_recording = rgb_recording
        self.depth_recording = depth_recording
        self.mask_recording = mask_recording
        self.ee_positions = ee_positions
        self.gr_positions = gr_positions

        size = flow_recording.shape[1:3]
        self.size = size

        self.forward_flow = False  # default is backward(demo->obs)
        self.flow_mask = True  # default is masking
        self.fitting_control = True  # use least squares fit of FG points

        # load flow net (needs image size)
        print("Image shape from recording", size)
        self.flow_module = FlowModule(size=size)

        # select frame
        self.base_frame = base_frame
        self.set_base_frame(self.base_frame)
        self.max_demo_frame = rgb_recording.shape[0] - 1

        self.threshold = threshold
        # depricated
        self.mask_erosion = False
        #num_mask = 32 # the number of pixels to retain after erosion
        #if self.mask_erosion:
        #    from scipy import ndimage as ndi
        #    self.base_mask = ndi.distance_transform_edt(base_mask)
        #    mask_thr = np.sort(base_mask.flatten())[-num_mask]
        #    self.base_mask = base_mask > mask_thr

        self.counter = 0
        if plot:
            self.view_plots = ViewPlots(threshold=threshold)

            # def onclick(event):
            #     print(event.key)
            #
            # cid = self.view_plots.fig.canvas.mpl_connect('key_press_event', onclick)
        else:
            self.view_plots = False
        self.opencv_input = opencv_input
        self.key_pressed = False
        self.mode = "auto"

    def generate_pointcloud(self, rgb_image, depth_image, masked_points):
        pointcloud = []
        for u, v in masked_points:
            try:
                Z = depth_image[u, v] * 0.000125
                color = rgb_image[u, v]
            except IndexError:
                Z = 0
                color = 0, 0, 0
            X = (v - C_X) * Z / FOC_X
            Y = (u - C_Y) * Z / FOC_Y
            pointcloud.append([X, Y, Z, 1, *color])
        pointcloud = np.array(pointcloud)
        return pointcloud

    def step(self, live_rgb, ee_pos, live_depth=None):
        # 1. compute flow
        # 2. compute transformation
        # 3. transformation to control

        assert(live_rgb.shape == self.base_image_rgb.shape)

        # Control computation
        if self.forward_flow:
            flow = self.flow_module.step(live_rgb, self.base_image_rgb)
        else:
            flow = self.flow_module.step(self.base_image_rgb, live_rgb)


        # Do the masked reference computation here.
        # because I need to use the FG mask
        assert(self.forward_flow == False)

        # for compatibility with notebook.
        start_image = live_rgb
        start_depth = live_depth
        end_image = self.base_image_rgb
        end_depth = self.base_image_depth

        end_points = np.array(np.where(self.base_mask)).T
        masked_flow = flow[self.base_mask]
        start_points = end_points + masked_flow[:, ::-1].astype('int')

        T_tcp_cam = np.array([
            [0.99987185, -0.00306941, -0.01571176, 0.00169436],
            [-0.00515523, 0.86743151, -0.49752989, 0.11860651],
            [0.015156, 0.49754713, 0.86730453, -0.18967231],
            [0., 0., 0., 1.]])

        K = np.array([[617.89, 0, 315.2, 0],
                      [0, 617.89, 245.7, 0],
                      [0, 0, 1, 0]])

        def project(K, X):
            x = K @ X
            return x[0:2] / x[2]

        start_pc = self.generate_pointcloud(start_image, start_depth, start_points)
        end_pc = self.generate_pointcloud(end_image, end_depth, end_points)

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
        r = R.from_dcm(guess[:3,:3])
        xyz = r.as_euler('xyz')
        rot_z = xyz[2]
        # magical gain values for control, these could come from calibration

        if not np.all(np.isfinite(guess)):
            print("bad trf guess")
            null_action = [0,0,0,0,1]
            return null_action, self.mode

        # change names
        t_scaling = 10
        r_scaling = 10

        mean_flow = guess[0,3]*t_scaling, guess[1,3]*t_scaling
        mean_rot = -1*rot_z*r_scaling

        # Outputs of this block: used at beginning of loop
        #   1. mean_flow
        #   2. mean_rot

        pos_diff = self.base_pos - ee_pos
        loss_z = np.abs(pos_diff[2])*10
        loss_pos = np.linalg.norm(mean_flow)
        loss_rot = np.abs(mean_rot) / 1.5
        loss = loss_pos + loss_rot + loss_z

        z = pos_diff[2] * 10 * 3
        action = [mean_flow[0], mean_flow[1], z, mean_rot, self.grip_state]


        # plotting code
        if self.view_plots:
            # show flow
            flow_img = self.flow_module.computeImg(flow, dynamic_range=False)
            # show segmentatione edge
            if self.counter % 5 == 0:
                edge  = np.gradient(self.base_mask.astype(float))
                edge = (np.abs(edge[0])+ np.abs(edge[1])) > 0
                flow_img[edge] = (255, 0, 0)

            print("loss =", loss, action)
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

        print("threshold", self.threshold)
        if loss < self.threshold and self.base_frame < self.max_demo_frame or self.key_pressed:
            # this is basically a step function
            if not self.key_pressed:
                self.base_index += 1
            self.base_frame = self.keep_indexes[np.clip(self.base_index,0,len(self.keep_indexes)-1)]
            self.key_pressed = False
            self.set_base_frame(self.base_frame)
            print("demonstration: ", self.base_frame, "/", self.max_demo_frame)

        self.counter += 1
        return action

    def set_base_frame(self, base_frame):
        self.base_image_rgb = self.rgb_recording[base_frame]
        self.base_image_depth = self.depth_recording[base_frame]
        self.base_mask = self.mask_recording[base_frame]
        self.base_pos = self.ee_positions[base_frame]
        self.grip_state = self.gr_positions[base_frame]

        #
        #if self.mask_erosion:
        #self.base_mask = ndi.distance_transform_edt(self.base_mask)
        #mask_thr = np.sort(base_mask.flatten())[-num_mask]
        #self.base_mask = base_mask > mask_thr

