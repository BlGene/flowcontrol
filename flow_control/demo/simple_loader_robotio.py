import json
from glob import glob
from pathlib import Path
from typing import Union, List
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class Action:
    """
    Class for sending cartesian actions to the robot.
    Due to differences in the implementation of the robot controllers, not all action types are available for every
    robot.

    Args:
        target_pos: The target position, either absolute or relative.
        target_orn: The target orientation, either absolute or relative, as quaternion or euler angles.
        gripper_action: Binary gripper action. 1 -> open, -1 -> close.
        ref: "abs" (absolute w.r.t base frame) | "rel" (relative w.r.t. tcp frame) | "rel_world" (relative w.r.t. base frame)
        path: "ptp" (motion linear in joint space) | "lin" (motion linear in cartesian space)
        blocking: If True, block until the action is executed.
        impedance: If True, use impedance control (compliant robot). Typically less precise, but safer.
    """
    target_pos: Union[tuple, np.ndarray]
    target_orn: Union[tuple, np.ndarray]
    gripper_action: int
    ref: str = "abs"  # "abs" | "rel" | "rel_world"
    path: str = "ptp"  # "ptp" | "lin"

    blocking: bool = False
    impedance: bool = False

class SimpleLoader:
    def __init__(self, demo_dir) -> None:
        self.demo_dir = Path(demo_dir)
        
    # Some helper functions to load the data from numpy format
    def get_image(self, frame_index) -> np.ndarray:
        """returns the image for a given frame
        Returns:
            rgb_gripper: numpy array (w,h, 3) in range (0, 255)
        """
        arr = np.load(self.demo_dir / f"frame_{frame_index:06d}.npz")
        return arr["rgb_gripper"]

    def get_depth(self, frame_index) -> np.ndarray:
        """Returns
            depth: numpy array size (w, h) units of meters
        """
        arr = np.load(self.demo_dir / f"frame_{frame_index:06d}.npz")
        return self.depth_img_from_uint16(arr["depth_gripper"])

    def get_reward(self) -> float:
        """return the final reward of the episode"""
        frame_names = sorted(glob(str(self.demo_dir/ "frame_*.npz")))
        rew = np.load(frame_names[-1])["rew"].item()
        return rew

    def get_len(self) -> int:
        """return the lenght of the episode"""
        frame_names = sorted(glob(str(self.demo_dir/ "frame_*.npz")))
        return len(frame_names)

    @staticmethod
    def pos_orn_to_matrix(pos, orn) -> np.ndarray:
        """return 4x4 pose and orientation matrix"""
        mat = np.eye(4)
        #if isinstance(orn, np.quaternion):
        #    orn = np_quat_to_scipy_quat(orn)
        #    mat[:3, :3] = R.from_quat(orn).as_matrix()
        if len(orn) == 4:
            mat[:3, :3] = R.from_quat(orn).as_matrix()
        elif len(orn) == 3:
            mat[:3, :3] = R.from_euler('xyz', orn).as_matrix()
        mat[:3, 3] = pos
        return mat

    @staticmethod
    def depth_img_from_uint16(depth_img, max_depth=4) -> np.ndarray:
        depth_img[np.isnan(depth_img)] = 0
        return depth_img.astype('float') / (2 ** 16 - 1) * max_depth

    @staticmethod
    def depth_img_to_uint16(depth_img, max_depth=4) -> np.ndarray:
        depth_img = np.clip(depth_img, 0, max_depth)
        return (depth_img / max_depth * (2 ** 16 - 1)).astype('uint16')

    def get_tcp_pose(self, frame_index) -> np.ndarray:
        """return the tcp position for a given frame as a 4x4 matrix"""
        arr = np.load(self.demo_dir / f"frame_{frame_index:06d}.npz", allow_pickle=True)
        state = arr["robot_state"].item()
        return self.pos_orn_to_matrix(state["tcp_pos"], state["tcp_orn"])

    def get_gripper_width(self, frame_index) -> float:
        arr = np.load(self.demo_dir / f"frame_{frame_index:06d}.npz", allow_pickle=True)
        return arr["robot_state"].item()["gripper_opening_width"]

    def get_extr_cal(self) -> np.ndarray:
        """return extrinsic calibration
        Returns:
            extr: 4x4 matrix with units of meters
        """
        camera_info = np.load(Path(self.demo_dir) / "camera_info.npz", allow_pickle=True)
        extr = camera_info["gripper_extrinsic_calibration"]
        return extr

    def get_intr_cal(self) -> dict:
        """return intrinsic calibraiton:
        Returns:
            intr: dict with entries: width, height, cx, cy, fx, fy in pixels
        """
        camera_info = np.load(Path(self.demo_dir) / "camera_info.npz", allow_pickle=True)
        intr = camera_info["gripper_intrinsics"].item()
        return intr

    @staticmethod
    def act2inst(dict_action, path=None, blocking=None) -> Action:
        """
        This turns a dictionary action into an Action class action
        """
        if dict_action is None:
            return None
        if not "path" in dict_action:
            dict_action["path"] = None
        if not "blocking" in dict_action:
            dict_action["blocking"] = None

        act_inst = Action(target_pos=dict_action["motion"][0],
                        target_orn=dict_action["motion"][1],
                        gripper_action=dict_action["motion"][2],
                        ref=dict_action["ref"],
                        path=path if path is not None else dict_action["path"],
                        blocking=blocking if blocking is not None else dict_action["blocking"])
        return act_inst

    def get_action(self, frame_num) -> Action:
        arr = np.load(self.demo_dir / f"frame_{frame_num:06d}.npz", allow_pickle=True)
        dict_action  = arr["action"].item()
        action = self.act2inst(dict_action)
        return action

    def get_info(self, frame_index) -> dict:
        """returns the info for a given frame"""
        arr = np.load(self.demo_dir / f"frame_{frame_index:06d}.npz", allow_pickle=True)
        return arr["info"].item()

    def get_keyframes(self) -> List[int]:
        """Returns:
            list of frame numbers  of keyframes
        """
        keep_dict_fn = Path(self.demo_dir) / "servo_keep.json"
        with open(keep_dict_fn) as f_obj:
            keep_dict_from_file = json.load(f_obj)
            # undo json mangling
            keep_dict_e = {int(key): val for key, val in keep_dict_from_file.items()}
        return list(keep_dict_e.keys())

    def get_segmentation(self, frame_num) -> np.ndarray:
        """return a forground calibration
        Returns:
            mask: numpy array (w, h) of type boolean
        """
        mask_recording_fn = Path(self.demo_dir) / "servo_mask.npz"
        mask_fo = np.load(mask_recording_fn, allow_pickle=True)
        fg_masks = dict([(i, mask==mask_fo["fg"][i]) for i, mask in enumerate(mask_fo["mask"])])
        mask = fg_masks[frame_num]
        return mask
