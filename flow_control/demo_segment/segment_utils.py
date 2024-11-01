import numpy as np
import matplotlib.pyplot as plt

from skimage import measure
from scipy import ndimage

from robot_io.envs.playback_env import PlaybackEnv


def mask_color(image, color_choice, threshold):
    """
    Arguments:
        image (w, h, 3) uint8
        color_choice (r, g, b) [0, 1] float
        threshold [0, 1] float
    """
    if isinstance(color_choice, str) and color_choice == "bw":  # this is for the wheel task
        tmp = np.linalg.norm(image / 255, axis=2) / 3**.5
        mask = tmp > threshold
    elif isinstance(color_choice, str) and color_choice == "keep_black":
        tmp = np.linalg.norm(image / 255, axis=2)
        mask = tmp < threshold
    else:
        color_choice = np.array(color_choice)
        tmp = np.linalg.norm(image * color_choice, axis=2) / np.linalg.norm(image, axis=2)
        mask = tmp > threshold
    return mask


def erode_mask(mask, close=2, erode=4):
    """
    Perform erosion of the mask.
    """
    # TODO(max): I think there is a function that does both
    mask = ndimage.binary_closing(mask, iterations=close)
    mask = ndimage.morphology.binary_erosion(mask, iterations=erode)
    return mask


def label_mask(mask):
    """
    Get the biggest connected component in the mask.
    """
    # this computes the connected components
    labels, num = measure.label(mask, background=0, return_num=True)
    # find the biggest component here.
    # label_id, label_count = np.unique(labels, return_counts=True)
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
        center_dist = np.array(mask.shape) / 2 - reg["centroid"]
        center_dist = np.linalg.norm(center_dist)
        center_dists[reg["label"]] = center_dist
        areas[reg["label"]] = reg["area"]

    if len(areas) == 0:
        return np.zeros(mask.shape)

    # remove blobs with area below 0.1 of max area
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
    assert calibration
    assert calibration["width"] == depth_image.shape[1]
    assert calibration["height"] == depth_image.shape[0]

    c_x = calibration["ppx"]
    c_y = calibration["ppy"]
    f_x = calibration["fx"]
    f_y = calibration["fy"]

    rows, cols = depth_image.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)

    z = depth_image
    x = z * (c - c_x) / f_x
    y = z * (r - c_y) / f_y
    o = np.ones(z.shape)

    tmp = np.stack((x, y, z, o), axis=2)
    tmp2 = tmp @ transformation
    tmp3 = tmp2[:, :, :3] / tmp2[:, :, 3, np.newaxis]
    return tmp3[:, :, 2]


def segment_plane():
    """
    Segment a plane in the recording.
    """
    import open3d as o3d

    rec = PlaybackEnv("/home/argusm/CLUSTER/robot_recordings/flow/sick_vacuum/17-19-19/").to_list()
    image, depth = rec[0].cam.get_image()
    print("done loading")

    pc = rec[0].cam.generate_pointcloud2(image, depth)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(pc[:, 4:7] / 255.)

    print("start segment")
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.003,
                                             ransac_n=3,
                                             num_iterations=100)
    [pl_a, pl_b, pl_c, pl_d] = plane_model
    print(f"Plane equation: {pl_a:.2f}x + {pl_b:.2f}y + {pl_c:.2f}z + {pl_d:.2f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


def plot_mask():
    initial_mask = np.load("mask_test.npz")["arr_0"]
    closest_mask = mask_center(initial_mask)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(initial_mask, cmap='gray')
    ax[1].imshow(closest_mask, cmap='nipy_spectral')
    plt.show()


if __name__ == "__main__":
    plot_mask()
