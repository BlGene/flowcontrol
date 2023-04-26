from pathlib import Path
from typing import Union
from dataclasses import dataclass
from glob import glob

import numpy as np
from PIL import Image

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
    def __init__(self, demo_dir, run) -> None:
        self.demo_dir = Path(demo_dir)
        self.run = run
        
    # Some helper functions to load the data from numpy format
    def get_image(self, frame_index) -> np.ndarray:
        """returns the image for a given frame
        Returns:
            rgb_gripper: numpy array (w,h, 3) in range (0, 255)
        """
        img_path = self.demo_dir / "{0}/rgb/{1:05d}.png".format(self.run, frame_index)
        return np.asarray(Image.open(img_path))
    
    def get_len(self) -> int:
        """return the lenght of the episode"""
        images_path = (self.demo_dir / f"{self.run}/rgb/*.png")
        frame_names = sorted(glob(str(images_path)))
        return len(frame_names)
