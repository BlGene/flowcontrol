"""
Util functions to create an automatic segmentation mask
"""
import cv2
import numpy as np
from tqdm import tqdm

from flow_control.flow.module_raft import FlowModule


def segment(video: np.ndarray, keyframe1: int, keyframe2: int):
    """
    Creates a segmentation mask of the video

    During 0 <= t <= keyframe1: Focuses on an object grasped at keyframe1
    During keyframe1 <= t <= keyframe2: Focuses on the object where the gripper
        opens at keyframe2

    video: [t, h, w, c] dtype=uint8
    keyframe1: int Frame when the object is grasped
    keyftame2: int Frame when the object is dropped
    """

    flow_module = FlowModule()

    mask_gripper = refine_mask(static_mask(flow_module, video, 0), video[0])
    mask_grasped = refine_mask(
        static_mask(flow_module, video, keyframe1 + 10),
        video[keyframe1 + 10]
    )

    final_masks = np.zeros(video.shape[:3], bool)

    # Move forward the gripper mask until its closed
    curr_frame = keyframe1 - 10 - 1
    for next_frame in tqdm(range(keyframe1 - 10, keyframe1 + 10, 1)):
        # Warp mask following the flow
        mask_gripper = flow_module.warp_mask(
            mask_gripper,
            video[curr_frame],
            video[next_frame]
        )

        # Refine mask with grabcut
        mask_gripper = refine_mask(mask_gripper, video[next_frame])

        curr_frame = next_frame

    # Difference between the grasped mask and the propagated gripper
    mask_object = refine_mask(
        mask_grasped & (mask_grasped ^ mask_gripper),
        video[curr_frame]
    )
    mask_object_init = mask_object.copy()

    final_masks[curr_frame] = mask_object[..., 0]

    # Move backwards the object mask until beginning of the video
    flow_module.flow_prev = None
    for next_frame in tqdm(range(curr_frame - 1, -1, -1)):
        # Warp mask following the flow
        mask_object = flow_module.warp_mask(
            mask_object, video[curr_frame],
            video[next_frame]
        )

        # Refine mask with grabcut
        mask_object = refine_mask(
            mask_object & (mask_object ^ mask_gripper),
            video[next_frame]
        )

        final_masks[next_frame] = mask_object[..., 0]
        curr_frame = next_frame

    # Mask target object
    mask_target = center_mask(mask_object_init, video[keyframe2 - 10], 100)

    mask_target = mask_target & (mask_gripper ^ mask_target)

    mask_target = refine_mask(mask_target, video[keyframe2 - 10])
    mask_target_init = mask_target.copy()

    mask_clean = mask_target & (mask_target ^ mask_grasped)
    mask_clean = mask_clean & (mask_clean ^ mask_object_init)
    final_masks[keyframe2 - 9:] = mask_clean[..., 0]

    # Move backwards the target mask until object is grasped
    curr_frame = keyframe2 - 10
    flow_module.flow_prev = None
    for next_frame in tqdm(range(curr_frame - 1, keyframe1 + 9, -1)):
        # Warp mask following the flow
        mask_target = flow_module.warp_mask(
            mask_target, video[curr_frame],
            video[next_frame]
        )

        # Refine mask with grabcut
        mask_target = refine_mask(
            mask_target, video[next_frame],
            dilation_kernel=31,
            erosion_kernel=15
        )

        mask_clean = mask_target & (mask_target ^ mask_grasped)
        mask_clean = mask_clean & (mask_clean ^ mask_object_init)
        mask_clean = cv2.erode(np.uint8(mask_clean), np.ones((5, 5), np.uint8))
        mask_clean = mask_clean[..., None]
        mask_clean = cv2.dilate(mask_clean, np.ones((5, 5), np.uint8))
        mask_clean = mask_clean[..., None].astype(bool)

        final_masks[next_frame] = mask_clean[..., 0]
        curr_frame = next_frame

    # Move forwards the target mask until the end of the video
    mask_target = mask_target_init.copy()
    curr_frame = keyframe2 - 10
    flow_module.flow_prev = None
    for next_frame in tqdm(range(curr_frame + 1, video.shape[0])):
        # Warp mask following the flow
        mask_target = flow_module.warp_mask(
            mask_target,
            video[curr_frame],
            video[next_frame]
        )

        # Refine mask with grabcut
        mask_target = refine_mask(mask_target, video[next_frame])

        final_masks[next_frame] = mask_target[..., 0]
        curr_frame = next_frame

    return final_masks


def center_mask(mask, img, radius=50):
    """ Tries to create a mask of an object located on the center of the image
    """
    mask = np.uint8(mask)

    prob_fg = np.zeros_like(mask)
    sure_fg = cv2.erode(mask, np.ones((31, 31), np.uint8))[..., None]

    prob_fg[cv2.dilate(mask, np.ones((31, 31), np.uint8))[..., None] > 0] = 3
    prob_fg = cv2.circle(
        prob_fg, (img.shape[1] // 2, img.shape[0] // 2),
        radius, 3, -1
    )

    # Probably background
    mask = np.full_like(mask, 2)
    # Probable foreground
    mask[prob_fg > 0] = 3
    # Sure foreground
    mask[sure_fg > 0] = 1

    # Run grab cut to refine the mask
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)

    mask, _, _ = cv2.grabCut(img, mask, None, bgd, fgd,
                             3, cv2.GC_INIT_WITH_MASK)
    mask = np.uint8(np.where((mask == 2) | (mask == 0), 0, 1))
    return mask


def refine_mask(mask: np.ndarray,
                img: np.ndarray,
                erosion_kernel: int = 31,
                dilation_kernel: int = 15) -> np.ndarray:
    """ Refines a mask based on its corresponding image

    mask: [H, W, 1] np.uint8
    img: [H, W, 3] np.uint8
    """
    mask = np.uint8(mask)

    # Erosion and dilate parameters
    erosion_kernel = np.ones((erosion_kernel, erosion_kernel), np.uint8)
    dilation_kernel = np.ones((dilation_kernel, dilation_kernel), np.uint8)

    # Dilated mask is probable foreground
    prob_fg = cv2.dilate(mask, dilation_kernel)[..., None]
    # Eroded mask is sure foreground
    sure_fg = cv2.erode(mask, erosion_kernel)[..., None]

    # Sure background
    mask = np.full_like(mask, 0)
    # Probable foreground
    mask[prob_fg > 0] = 3
    # Sure foreground
    mask[sure_fg > 0] = 1

    # Run grab cut to refine the mask
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    mask, _, _ = cv2.grabCut(img, mask, None, bgd, fgd,
                             3, cv2.GC_INIT_WITH_MASK)
    mask = np.uint8(np.where((mask == 2) | (mask == 0), 0, 1))

    return mask > 0


def static_mask(flow_module,
                video: np.ndarray,
                start: int,
                threshold: int = 0.5,
                max_steps: int = 50,
                step_size: int = 1) -> np.ndarray:
    """ Creates a mask of static pixels
    """
    max_range = min(start + max_steps * step_size, video.shape[0] - 1)
    for i in range(start + step_size, max_range, step_size):
        flow = flow_module.step(video[start], video[i])
        mask = (np.linalg.norm(flow, axis=-1, keepdims=True) < threshold)
        if np.abs(flow.mean()) > threshold and mask.sum() > 0.025 * mask.size:
            print('Static mask completed')
            return mask

    raise ValueError("Not suitable initial flow found")
