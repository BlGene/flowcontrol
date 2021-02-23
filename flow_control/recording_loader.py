"""
A class to load the data saved by a recording, behaves like env.
"""
import os
from PIL import Image
import numpy as np

from gym_grasping.flow_control.rgbd_camera import RGBDCamera


class RecordingLoader(RGBDCamera):
    """
    Class to load the data saved by a recording.

    At the moment this can come from two sources:
        1) demonstation episodeds using demo_episode_recorder.py
        2) views from random poses using random_view_recoreder.py

    default scaling for: color is [0, 255]
                         depth is [m]
                         seg is [0, 1]

    method naming similar to RealSense
    """

    def __init__(self, path, name=None, seg_path=None):
        self.index = None

        # load camera calibration
        cam_cal_fn = os.path.join(path, "calib.json")
        if os.path.isfile(cam_cal_fn):
            super().__init__(cam_cal_fn)

        if name is not None:
            path = os.path.join(path, name)

        # load RGB-D images
        obj_files = sorted(os.listdir(path))

        rgb_files = []
        depth_files = []
        # TODO(max) this code is very brittle to missing files
        for filename in obj_files:
            if filename.startswith("rgb_") and filename.endswith(".png"):
                file_path = os.path.join(path, filename)
                image = Image.open(file_path)
                rgb_files.append(image)
            elif filename.startswith("depth_") and filename.endswith(".png"):
                file_path = os.path.join(path, filename)
                image = Image.open(file_path)
                depth_files.append(image)
        self.rgb_files = rgb_files
        self.depth_files = depth_files
        self.depth_scaling = 0.000125
        print("images loaded.")
        # Load foreground masks
        if seg_path is None:
            seg_path = path
        mask_fn = os.path.join(seg_path, "mask.npz")
        masks = np.load(mask_fn)["masks"]
        self.masks = masks

    def get_index(self):
        '''get current index'''
        return self.index

    def set_index(self, index):
        '''set indexed frame'''
        self.index = index

    def get_color_frame(self, index=None):
        '''get color frame'''
        if index is None:
            index = self.index
        rgb = self.rgb_files[index]
        return np.asarray(rgb)

    def get_depth_frame(self, index=None):
        '''get depth frame'''
        if index is None:
            index = self.index
        depth = self.depth_files[index]
        return depth

    def get_seg_frame(self, index=None):
        '''get segmentation frame'''
        seg = self.masks[index]
        return seg

    # as (rgb, d, s)
    def get_RGBDS_frame(self, index=None):
        '''get RGB-D-Segmentation frame'''
        if index is None:
            index = self.index
        rgb = self.rgb_files[index]
        depth = self.depth_files[index]
        seg = self.masks[index]
        return (np.asarray(rgb), np.asarray(depth)*self.depth_scaling,
                np.asarray(seg))

    def get_pointcloud(self, index=None, masked=False):
        '''get a pointcloud, in camera coords?'''
        if index is None:
            index = self.index
        rgb_image, depth_image, mask = self.get_RGBDS_frame(index)
        if masked:
            masked_points = np.array(np.where(mask)).T
            pcd = self.generate_pointcloud(rgb_image, depth_image,
                                           masked_points)
        else:
            pcd = self.generate_pointcloud2(rgb_image, depth_image)
        return pcd

# class MaskedRecordingLoader:
#    """
#    Masked in this case means time steps are masked, not pixels

#    Make a masked version of the Loaded, so that:
#        1) pre-defined sub-sampling can be done.
#        2) single frames can be excluded.
#    """
#    def __init__(self):
#        raise NotImplementedError
