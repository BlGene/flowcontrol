import numpy as np

from scipy.spatial.transform import Rotation as R

from gym_grasping.flow_control.servoing_fitting import solve_transform
from gym_grasping.flow_control.flow_module import FlowModule
from gym_grasping.flow_control.live_plot import ViewPlots

class ServoingModule:
    def __init__(self, recording, base_frame=8, threshold=.35, plot=False):
        # load files
        flow_recording_fn = "./{}/episode_1_img.npz".format(recording)
        mask_recording_fn = "./{}/episode_1_mask.npz".format(recording)
        state_recording_fn = "./{}/episode_1.npz".format(recording)
        flow_recording = np.load(flow_recording_fn)["img"]
        mask_recording = np.load(mask_recording_fn)["mask"]
        state_recording = np.load(state_recording_fn)
        ee_positions = state_recording["ee_positions"]
        gr_positions = state_recording["gripper_states"]

        self.flow_recording = flow_recording
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
        self.base_image = flow_recording[base_frame]
        self.base_mask = mask_recording[base_frame]
        self.base_pos = ee_positions[base_frame]
        self.grip_state = gr_positions[base_frame]
        self.max_demo_frame = flow_recording.shape[0] - 1

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


    def step(self, state, ee_pos):
        assert(state.shape == self.base_image.shape)

        # Control computation
        if self.forward_flow:
            flow = self.flow_module.step(state, self.base_image)
        else:
            flow = self.flow_module.step(self.base_image, state)
        if self.flow_mask:
            assert self.forward_flow == False
            flow_tmp = flow.copy()
            flow_tmp[np.logical_not(self.base_mask)] = 0

        # Do the masked reference computation here.
        if self.fitting_control:
            # because I need to use the FG mask
            assert(self.forward_flow == False)

            x = np.linspace(-1, 1, self.size[0])
            y = np.linspace(-1, 1, self.size[1])
            xv,yv = np.meshgrid(x, y)
            # rotate
            field =  np.stack((yv, -xv), axis=2)
            field[:,:,0] *= (84-1)/2*self.size[0]/256
            field[:,:,1] *= (84-1)/2*self.size[1]/256
            points = field

            observations = points + flow
            #select foreground
            points = points[self.base_mask]
            observations = observations[self.base_mask]
            points = np.pad(points, ((0, 0), (0, 2)), mode="constant")
            observations = np.pad(observations, ((0, 0), (0, 2)), mode="constant")
            guess = solve_transform(points, observations)
            r = R.from_dcm(guess[:3,:3])
            xyz = r.as_euler('xyz')
            rot_z = xyz[2]
            # magical gain values for control, these could come from calibration
            mean_flow = guess[0,3]/23, guess[1,3]/23
            mean_rot = -7*rot_z
        else:
            # previously I was using FG points for translation
            # radial field trick for rotation
            mean_flow = np.mean(flow_tmp[:,:,0], axis=(0,1)), np.mean(flow_tmp[:,:,1], axis=(0,1))
            # compute rotation
            rot_field = np.sum(flow*flow_module.inv_field, axis=2)
            mean_rot = np.mean(rot_field, axis=(0, 1))
        if not self.forward_flow:
            # then reverse direction of actions
            mean_flow = np.array(mean_flow) * -1
            mean_rot = mean_rot * -1
        # Outputs of this block: used at beginning of loop
        #   1. mean_flow
        #   2. mean_rot

        pos_diff = self.base_pos - ee_pos
        loss = np.linalg.norm(mean_flow) + np.abs(mean_rot) + np.abs(pos_diff[2])*10

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
            self.view_plots.low_1_h.set_data(state)
            self.view_plots.low_2_h.set_data(self.base_image)
            self.view_plots.low_3_h.set_data(flow_img)
            plot_fn = f'./video/{self.counter:03}.png'
            #plt.savefig(plot_fn, bbox_inches=0)

            # depricated, causes error
            #img = np.concatenate((state[:,:,::-1], base_image[:,:,::-1], flow_img[:,:,::-1]),axis=1)
            #cv2.imshow('window', cv2.resize(img, (300*3,300)))
            #cv2.waitKey(1)

        # demonstration stepping code
        if loss < self.threshold and self.base_frame < self.max_demo_frame:
            # this is basically a step function
            self.base_frame += 1
            self.base_image = self.flow_recording[self.base_frame]
            self.base_mask = self.mask_recording[self.base_frame]
            self.base_pos = self.ee_positions[self.base_frame]
            self.grip_state = self.gr_positions[self.base_frame]
            if self.mask_erosion:
                self.base_mask = ndi.distance_transform_edt(base_mask)
                mask_thr = np.sort(base_mask.flatten())[-num_mask]
                self.base_mask = base_mask > mask_thr
            print("demonstration: ", self.base_frame, "/", self.max_demo_frame)

        self.counter += 1
        return action
