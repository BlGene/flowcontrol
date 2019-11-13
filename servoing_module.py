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
    def __init__(self, recording, episode_num = 0, base_frame=0, threshold=.35, plot=False):
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
            self.base_index = 0
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
        self.key_pressed = False
        self.mode = "manual"

    def generate_pointcloud(self, rgb_image, depth_image, masked_points):
        pointcloud = []
        for u, v in masked_points:
            color = rgb_image[u, v]
            Z = depth_image[u, v] * 0.000125
            # if Z == 0: continue
            X = (u - C_X) * Z / FOC_X
            Y = (v - C_Y) * Z / FOC_Y
            pointcloud.append([X, Y, Z, *color])
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

        # x = np.linspace(-1, 1, self.size[0])
        # y = np.linspace(-1, 1, self.size[1])
        # xv, yv = np.meshgrid(x, y)
        #
        # field =  np.stack((yv, -xv), axis=2)
        # field[:,:,0] *= (84-1)/2*self.size[0]/256
        # field[:,:,1] *= (84-1)/2*self.size[1]/256
        # points = field

        # observations = points + flow
        #select foreground
        # points = points[self.base_mask]

        # array shape: height x width
        points = np.array(np.where(self.base_mask)).T
        # print(self.base_mask.shape)
        # observations = observations[self.base_mask]
        masked_flow = flow[self.base_mask]
        observations = points + masked_flow[:,::-1].astype('int')

        if live_depth is not None:
            base_pointcloud = self.generate_pointcloud(self.base_image_rgb, self.base_image_depth, points)
            live_pointcloud = self.generate_pointcloud(live_rgb, live_depth, observations)
            guess = solve_transform(base_pointcloud, live_pointcloud)
        else:
            points = np.pad(points.astype('float'), ((0, 0), (0, 2)), mode="constant")
            observations = np.pad(observations.astype('float'), ((0, 0), (0, 2)), mode="constant")
            guess = solve_transform(points, observations)
        # from mpl_toolkits.mplot3d import Axes3D
        # import random
        #
        # fig = plt.figure()
        # ax = Axes3D(fig)
        #
        # ax.set_xlim3d(0, 0.3)
        # ax.set_ylim3d(0, 0.3)
        # ax.set_zlim3d(0, 0.3)
        # ax.scatter(np.reshape(live_pointcloud[:,0], -1), np.reshape(live_pointcloud[:,1], -1),np.reshape(live_pointcloud[:,2], -1), c=live_pointcloud[:,3:6]/255)
        # ax.scatter(0,0,0)
        # plt.show()

        # observations -= np.array([240, 320])
        # points -= np.array([240, 320])

        # scaling_factor = (84-1)/2*1/256
        # observations *= scaling_factor
        # points *= scaling_factor
        #
        # fig = plt.figure()
        # ax = fig.add_subplot(111)  # , projection='3d')
        # ax.scatter(points[:, 1], points[:, 0])
        # ax.scatter(observations[:, 1], observations[:, 0])
        #
        # ax.scatter([0], [0])
        # ax.set_xlim([int(-320 * scaling_factor),int(320 * scaling_factor)])
        # ax.set_ylim([int(240* scaling_factor),int(-240* scaling_factor)])
        # ax.set_aspect('equal')
        # plt.show()
        # print(points)
        # print(points.mean(axis=0))


        r = R.from_dcm(guess[:3,:3])
        xyz = r.as_euler('xyz')
        rot_z = xyz[2]
        # magical gain values for control, these could come from calibration
        mean_flow = guess[0,3]/23, guess[1,3]/23
        mean_rot = -7*rot_z

        if not self.forward_flow:
            # then reverse direction of actions
            mean_flow = np.array(mean_flow) * -1
            mean_rot = mean_rot * -1
        # Outputs of this block: used at beginning of loop
        #   1. mean_flow
        #   2. mean_rot

        pos_diff = self.base_pos - ee_pos
        loss_z = np.abs(pos_diff[2])*10
        loss_pos = np.linalg.norm(mean_flow)
        loss_rot = np.abs(mean_rot) / 1.5
        loss = loss_pos + loss_rot + loss_z

        z = pos_diff[2] * 10 * 3
        action = [mean_flow[0], -mean_flow[1], z, mean_rot, self.grip_state]


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

            # depricated, causes error
            #img = np.concatenate((live_rgb[:,:,::-1], base_image_rgb[:,:,::-1], flow_img[:,:,::-1]),axis=1)
            # cv2.imshow('window', np.zeros((100,100)))
            # k = cv2.waitKey(10) % 256
            # if k == ord('d'):
            #     self.key_pressed = True
            #     self.base_frame += 1
            #     print(self.base_frame)
            # elif k == ord('a'):
            #     self.key_pressed = True
            #     self.base_frame -= 1
            #     print(self.base_frame)
            # elif k == ord('c'):
            #     if self.mode == "manual":
            #         self.mode = "auto"
            #     else:
            #         self.mode = "manual"
            self.base_frame = np.clip(self.base_frame, 0, 300)
        # demonstration stepping code
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

