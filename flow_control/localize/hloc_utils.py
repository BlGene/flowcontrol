"""
Interface functions for using hloc
"""
import json
import pdb
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import open3d as o3d

from hloc.utils.io import get_keypoints, get_matches

from flow_control.demo.playback_env_servo import PlaybackEnvServo
from flow_control.servoing.fitting import solve_transform, eval_fit
from flow_control.servoing.fitting_ransac import Ransac


def export_image(root_dir, episode_name, frame_index, image_path):
    playback = PlaybackEnvServo(root_dir / episode_name, load=[frame_index])
    image_arr = playback[frame_index].cam.get_image()[0]
    Image.fromarray(image_arr).save(image_path)


def export_images_by_parts(root_dir, parts_fn, mapping_dir, export=True):
    Path(mapping_dir).mkdir(parents=True, exist_ok=True)
    mapping_rel = 'mapping'

    with open(parts_fn) as f_objs:
        demo_parts = json.load(f_objs)


    export_list = []
    ref_parts = defaultdict(list)
    for episode_name in sorted(demo_parts):
        #for part in demo_parts[demo]:
        for part in demo_parts[episode_name]:
            # export first image only
            frame_index = demo_parts[episode_name][part][0]
            ref_parts[part].append(f'{mapping_rel}/{episode_name}_{frame_index}.jpeg')
            export_list.append((episode_name, frame_index))
    if not export:
        return dict(ref_parts)


    for episode_name, frame_index in tqdm(export_list):
        image_path = Path(mapping_dir)/ f"{episode_name}_{frame_index}.jpeg"
        export_image(root_dir, episode_name, frame_index, image_path)

    return dict(ref_parts)


def get_playback(root_dir, reference):
    episode_dir, frame_num_str = reference.replace("mapping/", "").replace(".jpeg", "").split("_")
    frame_num = int(frame_num_str)
    print(root_dir, episode_dir)
    npz_fn = root_dir / episode_dir / "frame_{0:06d}.npz".format(frame_num)
    print(npz_fn)
    pb = PlaybackEnvServo(root_dir / episode_dir, load=[frame_num])
    return pb, frame_num


def get_playback_keep(root_dir, reference):
    episode_dir, frame_num_str = reference.replace("mapping/", "").replace(".jpeg", "").split("_")
    frame_num = int(frame_num_str)
    servo_keep_fn = root_dir / episode_dir / "servo_keep.json"
    with open(servo_keep_fn, 'r') as f_obj:
        servo_keep = json.load(f_obj)

    gripper_close_idx = -1
    for key, val in servo_keep.items():
        if val['name'] == 'gripper_close':
            gripper_close_idx = int(key)

    pb = PlaybackEnvServo(root_dir / episode_dir, load='keep')
    return pb, frame_num, gripper_close_idx

def get_segmentation(root_dir, reference):
    pb, frame_index = get_playback(root_dir, reference)
    return pb.get_fg_mask(frame_index)


def write_keypoints(path: Path, name: str, p: np.ndarray, uncertainty=None) -> None:
    with h5py.File(str(path), 'a', libver='latest') as hfile:
        if name in hfile:
            del hfile[name]
        hfile_name = hfile.create_group(name)
        hfile_name["keypoints"] = p
        if uncertainty is not None:
            dset = hfile[name]["keypoints"]
            dset.attrs["uncertainty"] = uncertainty


def save_features_seg(root_dir, features_seg_path, features_path, references):
    for name in tqdm(references):
        # get keypoints for whole image
        kps, noise = get_keypoints(features_path, name, return_uncertainty=True)

        # apply segmentation
        seg = get_segmentation(root_dir, name)
        in_seg = np.where(seg[kps[:, 1].astype(int), kps[:, 0].astype(int)])[0]
        kps_seg = kps[in_seg]

        # write keypoints
        write_keypoints(features_seg_path, name, kps_seg, uncertainty=noise)

        #check results
        kps_seg2, noise2 = get_keypoints(features_seg_path, name, return_uncertainty=True)
        assert np.all(kps_seg == kps_seg2)
        assert noise == noise2


def get_a_in_b(A, B):
    aset = [tuple(x) for x in A]
    bset = [tuple(x) for x in B]
    a_indexes = []
    for i, a in enumerate(aset):
        if a in bset:
            a_indexes.append(i)
    return np.array(a_indexes)

def get_a_in_b_fast(A, B):
    nrows, ncols = A.shape
    dtype={'names':['f{}'.format(i) for i in range(ncols)], 'formats':ncols * [A.dtype]}
    C, A_ind, B_ind = np.intersect1d(A.view(dtype), B.view(dtype), return_indices=True)
    return A_ind

def kp_seg_filter(kps_d_match, name_d, features_seg_path):
    """
    filter keypoints that appear in segmentation, by default using h5 cache.
    """
    kps_d_f = get_keypoints(features_seg_path, name_d)
    return get_a_in_b_fast(kps_d_match, kps_d_f)

def kp_seg_filter_pb(kps_d_match, name_d):
    # filter keypoints by fg segmentation
    seg_d = get_segmentation(name_d)
    in_seg = np.where(seg_d[kps_d_match[:, 1].astype(int), kps_d_match[:, 0].astype(int)])[0]
    return in_seg


def get_pointcloud(name, kps=None, cam=None, root_dir=None):
    """return pointcloud for a frame, if kps are given for those, otherwise for all points"""
    if cam is None:
        pb, frame_index = get_playback(root_dir, name)
        cam = pb[frame_index].cam

    rgb_image, depth_image = cam.get_image()

    if kps is None:
        points = cam.compute_pointcloud(depth_image, rgb_image, homogeneous=True)
    else:
        points = []
        for kp in kps.astype(int):
            point = cam.deproject(kp, depth_image, homogeneous=True)
            points.append(point)
    return points


def get_segmented_pointcloud(name, kps=None, cam=None, root_dir=None, bbox=None, is_live=False, trf=None):
    """return pointcloud for a frame, if kps are given for those, otherwise for all points"""
    pb, frame_num, gripper_close_idx = get_playback_keep(root_dir, name)

    if cam is None:
        cam = pb[frame_num].cam

    ps, bbox_ext = [178, 425], 0.02

    if is_live:
        # bbox_pts = np.asarray(bbox.get_box_points())
        #
        # bbox_pts = np.hstack((bbox_pts, np.ones((8, 1))))
        # bbox_trf = (trf @ bbox_pts.T).T
        # pdb.set_trace()
        #
        # bbox = bbox_trf[:, 0:3]
        # bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(bbox))

        bbox_oriented = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(bbox)
        # pdb.set_trace()
        bbox = bbox_oriented.transform(trf)
        # bbox = bbox_oriented.rotate(trf[0:3, 0:3])
        # bbox = bbox.translate(trf[0:3, 3])
    else:
        bbox = get_bbox(pb, gripper_close_idx, ps, bbox_ext)

    rgb_image, depth_image = cam.get_image()

    if kps is None:
        points = cam.compute_pointcloud(depth_image, rgb_image, homogeneous=True)
    else:
        points = []
        for kp in kps.astype(int):
            point = cam.deproject(kp, depth_image, homogeneous=True)
            points.append(point)

    # Transform to world
    t_tcp_cam = pb.cam.get_extrinsic_calibration()
    t_tcp_robot = pb.robot.get_tcp_pose()
    trf = t_tcp_robot @ t_tcp_cam

    # figure out pixel assignments
    far_val = 10
    u_crd, v_crd = np.where(np.logical_and(depth_image > 0, depth_image < far_val))

    pointcloud_idx = np.zeros(depth_image.shape, dtype=int)
    # what we are looking for f(x,y) -> pointcloud idx
    pointcloud_idx[u_crd, v_crd] = np.arange(len(points))

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points[:, :3])
    if points.shape[1] == 7:
        pc.colors = o3d.utility.Vector3dVector(points[:, 4:7])

    pc.transform(trf)

    inliers = bbox.get_point_indices_within_bounding_box(pc.points)

    return points[inliers, :], bbox


def get_bbox(playback, frame_num, ps, bbox_ext):
    # this is a pixel point between the grippers (where the object is located)
    t_tcp_robot = playback[frame_num].robot.get_tcp_pose()
    t_tcp_cam = playback[frame_num].cam.get_extrinsic_calibration()
    trf = t_tcp_robot @ t_tcp_cam

    cam = playback[frame_num].cam
    img, depth = cam.get_image()
    pointcloud = cam.compute_pointcloud(depth, img)  # in camera coords

    # figure out pixel assignments
    far_val = 10
    u_crd, v_crd = np.where(np.logical_and(depth > 0, depth < far_val))

    pointcloud_idx = np.zeros(depth.shape, dtype=int)

    # what we are looking for f(x,y) -> pointcloud idx
    pointcloud_idx[u_crd, v_crd] = np.arange(len(pointcloud))
    pt_idx = pointcloud_idx[ps[0], ps[1]]

    point_before = pointcloud[pt_idx]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pointcloud[:, :3])
    if pointcloud.shape[1] == 6:
        pc.colors = o3d.utility.Vector3dVector(pointcloud[:, 3:6])

    pc.transform(trf)

    # create bounding box for cropping
    print("setting bbox.")
    point_after = pc.select_by_index([pt_idx]).points[0]
    min_bound = point_after - bbox_ext
    max_bound = point_after + bbox_ext
    print(type(point_after))

    # crop around selected shape
    print(pointcloud[pt_idx])
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)

    return bbox


# bbox_auto = get_object_point(first_open)

def align_pointclouds(root_dir, matches_path, features_path, features_seg_path,  name_q, name_d, query_cam=None):
    # collect all matches
    matches, scores = get_matches(matches_path, name_q, name_d)

    # get keypoints, filter by matches
    kps_q, noise_q = get_keypoints(features_path, name_q, return_uncertainty=True)
    kps_d, noise_d = get_keypoints(features_path, name_d, return_uncertainty=True)
    kps_q_match = kps_q[matches[:, 0]]
    kps_d_match = kps_d[matches[:, 1]]
    if len(kps_q_match) < 4:
        return None

    kps_d_f, _ = get_keypoints(features_seg_path, name_d, return_uncertainty=True)
    in_seg = kp_seg_filter(kps_d_match, name_d, features_seg_path)
    kps_q_seg = kps_q_match[in_seg]
    kps_d_seg = kps_d_match[in_seg]

    # print("num matches:", len(in_seg))
    if len(in_seg) < 4:
        return None

    # filter by valid depth values
    pc_q = get_pointcloud(name_q, kps_q_seg.astype(int), query_cam)
    pc_d = get_pointcloud(name_d, kps_d_seg.astype(int), root_dir=root_dir)
    pc_q_mask = [x is None for x in pc_q]
    pc_d_mask = [x is None for x in pc_d]
    pc_mask = np.logical_or(pc_d_mask, pc_q_mask)
    pc_q = np.array(pc_q, dtype=object)[np.logical_not(pc_mask)]
    pc_d = np.array(pc_d, dtype=object)[np.logical_not(pc_mask)]
    pc_q = np.array(list(pc_q), dtype=float)
    pc_d = np.array(list(pc_d), dtype=float)
    if len(pc_q) < 4:
        return None

    # get alignment
    thresh = .002  # 2mm
    num_pts_needed = 3
    ransac = Ransac(pc_q, pc_d, solve_transform, eval_fit, thresh, num_pts_needed, outlier_ratio=0.4)
    fit_q_pos, trf_est = ransac.run()

    num_inliers = np.sum(fit_q_pos < thresh)
    num_candidates = len(fit_q_pos)
    in_score = np.mean(fit_q_pos[fit_q_pos < thresh])
    # print(f"inliers: {num_inliers}/{num_candidates} {num_inliers/num_candidates:.2f} in-score: {in_score:.6f}")

    return dict(trf_est=trf_est, num_inliers=num_inliers, num_candidates=num_candidates,
                in_score=in_score, kps_q=kps_q_seg, kps_d=kps_d_seg)

