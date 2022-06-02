import numpy as np
from scipy.spatial.transform import Rotation as R

#
# Duplicates of robot_io.utils.utils, clean up?
#

def pos_orn_to_matrix(pos, orn):
    """
    Arguments:
        position (x,y,z)
        orientation (q_x, q_y, q_z, w)
    Returns:
        mat: 4x4 homogeneous transformation
    """
    assert len(pos) == 3
    assert len(orn) == 4
    mat = np.eye(4)
    mat[:3, 3] = pos
    mat[:3, :3] = R.from_quat(orn).as_matrix()
    return mat

def matrix_to_pos_orn(mat):
    """
    Arguments:
        mat: 4x4 homogeneous transformation
    Returns:
        tuple(position: np.array of shape (3,), orientation: np.array of shape (4,) -> quaternion xyzw)
    """
    orn = R.from_matrix(mat[:3, :3]).as_quat()
    pos = mat[:3, 3]
    return pos, orn
