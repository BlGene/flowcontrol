import numpy as np
import matplotlib.pyplot as plt

from skimage import measure
from scipy import ndimage


def mask_color(image, i, color_choice, threshold):

    if color_choice == "bw":  # this is for the wheel task
        tmp = np.linalg.norm(image/255, axis=2) / 3**.5
        mask = tmp > threshold
    elif color_choice == "keep_black":
        tmp = np.linalg.norm(image/255, axis=2)
        mask = tmp < threshold
    else:
        color_choice = np.array(color_choice)
        tmp = np.linalg.norm(image * color_choice, axis=2) / np.linalg.norm(image, axis=2)
        mask = tmp > threshold

    return mask


def erode_mask(mask, close=2, erode=4):
    # TODO(max): I think there is a function that does both
    mask = ndimage.binary_closing(mask, iterations=close)
    mask = ndimage.morphology.binary_erosion(mask, iterations=erode)
    return mask


def label_mask(mask, i):
    # this computes the connected components
    labels, num = measure.label(mask, background=0, return_num=True)
    label_id, label_count = np.unique(labels, return_counts=True)
    # find the biggest component here.
    # np.argsort(label_count)
    # print(label_count)
    mask = (labels == 0)
    return mask


def mask_center(mask):
    """
    Then take the one closest to the center
    Remove blobs with area beow .1 of max area

    Arguments:
        mask: input mask (False is background, True is foreground)
    Returns:
        closest_mask: mask with only blob closest to center
    """
    blobs_labels, num = measure.label(mask, background=0, return_num=True)

    if num == 1:
        return mask  # we can skip this

    regions = measure.regionprops(blobs_labels)
    center_dists = {}
    areas = {}
    for reg in regions:
        center_dist = np.array(mask.shape)/2-reg["centroid"]
        center_dist = np.linalg.norm(center_dist)
        center_dists[reg["label"]] = center_dist
        areas[reg["label"]] = reg["area"]
        # print(reg["label"], reg["centroid"], reg["area"])

    # remove blobs with area beow 0.1 of max area
    area_t = max(areas.values()) * 0.1
    for label, area in areas.items():
        if area < area_t:
            del center_dists[label]

    closest = min(center_dists, key=center_dists.get)
    closest_mask = (blobs_labels == closest)
    return closest_mask


def transform_depth(depth_image, transformation, calibration):
    """
    Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centred at cx, cy with field of view fx, fy.

    depth is a 2-D ndarray with shape (rows, cols) containing
    depths from 1 to 254 inclusive. The result is a 3-D array with
    shape (rows, cols, 3). Pixels with invalid depth in the input have
    NaN for the z-coordinate in the result.
    """
    assert(calibration)
    assert(calibration["width"] == depth_image.shape[1])
    assert(calibration["height"] == depth_image.shape[0])

    C_X = calibration["ppx"]
    C_Y = calibration["ppy"]
    F_X = calibration["fx"]
    F_Y = calibration["fy"]

    rows, cols = depth_image.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)

    z = depth_image
    x = z * (c - C_X) / F_X
    y = z * (r - C_Y) / F_Y
    o = np.ones(z.shape)

    tmp = np.stack((x, y, z, o), axis=2)
    tmp2 = tmp @ transformation
    tmp3 = tmp2[:, :, :3] / tmp2[:, :, 3, np.newaxis]
    return tmp3[:, :, 2]


if __name__ == "__main__":
    mask = np.load("mask_test.npz")["arr_0"]
    closest_mask = mask_center(mask)
    fix, ax = plt.subplots(1, 2)
    ax[0].imshow(mask, cmap='gray')
    ax[1].imshow(closest_mask, cmap='nipy_spectral')
    plt.show()
