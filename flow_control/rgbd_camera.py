"""
Simple RGB-D camera, that allows de projection to a point cloud
"""
import os
import json
import logging
import numpy as np


class RGBDCamera:
    """
    Simple RGB-D camera, that allows de projection to a point cloud
    """

    def __init__(self, info_dict):
        calibration = info_dict["calibration"]

        if "ppx" in calibration:
            calibration["cx"] = calibration["ppx"]
            calibration["cy"] = calibration["ppy"]
            del calibration["ppx"]
            del calibration["ppy"]

        if isinstance(calibration, str) and os.path.isfile(calibration):
            with open(calibration) as json_file:
                f_as_dict = json.load(json_file)
            k_matrix = f_as_dict["K"]
            f_x = k_matrix[0][0]
            f_y = k_matrix[1][1]
            cx = k_matrix[0][2]
            cy = k_matrix[1][2]
            calibration = dict(cx=cx, cy=cy, fx=f_x, fy=f_y)
            old_calibration = dict(cx=315.20367431640625,
                                   cy=245.70614624023438,
                                   fx=617.8902587890625, fy=617.8903198242188)
            if calibration != old_calibration:
                print("double check this")
                raise NotImplementedError

        self.calibration = calibration  # intrinsic
        try:
            self.flip_horizontal = info_dict["flip_horizontal"]
            self.T_tcp_cam = info_dict["T_tcp_cam"]  # extrinsic
        except KeyError:
            logging.warning("Old demonstration, setting flip_horizontal=True.")
            self.flip_horizontal = True

    def generate_pointcloud(self, rgb_image, depth_image, masked_points):
        """
        Generate a pointcloud by de-projecting points using camera calibration.

        Input:
            rgb_image: numpy array of dim [h, w, 3]
            depth_image: numpy array of dim [h, w]
            masked_poits: np array of dim [N, 2], img coords of point to keep

        Output:
            pointcloud: np array of dim [N x 7], 7 being (x,y,z,1,r,g,b)
            where 0 depth (z) indicates no data
        """
        if "width" in self.calibration:
            assert self.calibration["width"] == depth_image.shape[1]
        if "height" in self.calibration:
            assert self.calibration["height"] == depth_image.shape[0]
        assert masked_points.shape[1] == 2

        if 'rgb' in self.calibration:
            raise NotImplementedError  # old format

        c_x = self.calibration["cx"]
        c_y = self.calibration["cy"]
        foc_x = self.calibration["fx"]
        foc_y = self.calibration["fy"]

        num_points = len(masked_points)
        u_crd, v_crd = masked_points[:, 0], masked_points[:, 1]
        # save positions that map to outside of bounds, so that they can be
        # set to 0
        mask_u = np.logical_or(u_crd < 0, u_crd >= rgb_image.shape[0])
        mask_v = np.logical_or(v_crd < 0, v_crd >= rgb_image.shape[1])
        mask_uv = np.logical_not(np.logical_or(mask_u, mask_v))
        # temporarily clip out of bounds values so that we can use numpy
        # indexing
        u_clip = np.clip(u_crd, 0, rgb_image.shape[0] - 1)
        v_clip = np.clip(v_crd, 0, rgb_image.shape[1] - 1)
        # now set these values to 0 depth
        z_crd = depth_image[u_clip, v_clip] * mask_uv
        x_crd = (v_clip - c_x) * z_crd / foc_x
        y_crd = (u_clip - c_y) * z_crd / foc_y
        color_new = rgb_image[u_clip, v_clip]
        pointcloud = np.stack((x_crd, y_crd, z_crd, np.ones(num_points),
                               color_new[:, 0], color_new[:, 1],
                               color_new[:, 2]),
                              axis=1)
        return pointcloud

    def generate_pointcloud2(self, rgb_image, depth_image, as_image=False):
        """
        Generate a pointcloud by de projecting points using camera calibration.
        This version does not do masking.

        Input:
            rgb_image: numpy array of dim [h, w, 3]
            depth_image: numpy array of dim [h, w]

        Output:
            pointcloud: np array of dim [N x 7], 7 being (x,y,z,1,r,g,b)
            where 0 depth (z) indicates no data
        """
        if "width" in self.calibration:
            assert self.calibration["width"] == depth_image.shape[1]
        if "height" in self.calibration:
            assert self.calibration["height"] == depth_image.shape[0]

        c_x = self.calibration["cx"]
        c_y = self.calibration["cy"]
        foc_x = self.calibration["fx"]
        foc_y = self.calibration["fy"]

        rows, cols = depth_image.shape
        c_crd, r_crd = np.meshgrid(np.arange(cols), np.arange(rows),
                                   sparse=True)

        z_crd = depth_image
        x_crd = z_crd * (c_crd - c_x) / foc_x
        y_crd = z_crd * (r_crd - c_y) / foc_y
        ones = np.ones(z_crd.shape)
        pointcloud = np.stack((x_crd, y_crd, z_crd, ones,
                               rgb_image[:, :, 0],
                               rgb_image[:, :, 1],
                               rgb_image[:, :, 2]), axis=2)
        if as_image:
            return pointcloud
        else:
            return pointcloud.reshape(-1, pointcloud.shape[-1])
