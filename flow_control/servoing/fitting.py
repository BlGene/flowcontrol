"""
Find transformation s.t. (R|t) @ p == q
"""
import logging
import numpy as np
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R


def solve_transform(points_p, points_q):
    """
    Find transformation s.t. (R|t) @ p == q

    Arguments:
        points_p: (n, d) non-homogeneous
        points_q: (n, d) non-homogeneous
    Returns:
        align_transform: (d+1, d+1) homogeneous transformation
    """
    if len(points_p) == 0:
        logging.warning(f"solve_transform failed: empty set.")
        return np.eye(4)

    points_p = points_p[:, :3]
    points_q = points_q[:, :3]
    assert points_p.shape[1] == 3
    assert points_q.shape[1] == 3

    # compute mean translation
    p_mean = points_p.mean(axis=0)
    o_mean = points_q.mean(axis=0)

    # whiten
    points_x = points_p - p_mean
    points_y = points_q - o_mean

    s_matrix = (points_x.T @ points_y)

    u, _, v_T = np.linalg.svd(s_matrix)
    v = v_T.T
    det = np.linalg.det(v @ u.T)

    d_num = s_matrix.shape[0]
    Idt = np.eye(d_num)
    Idt[-1, -1] = det

    rot = (v @ Idt @ u.T)
    trans = o_mean - rot @ p_mean

    res = np.eye(d_num + 1)
    res[:d_num, :d_num] = rot
    res[:d_num, d_num] = trans

    if np.any(np.isnan(res)):
        logging.warning(f"solve_transform failed, contains NaN values.")
        return np.eye(4)

    assert np.linalg.det(res) > 0, "res = {}".format(str(res))
    return res


def eval_fit(trf_estm, start_ptc, end_ptc):
    start_m = (trf_estm @ start_ptc[:, 0:4].T).T
    fit_qe = np.linalg.norm(start_m[:, :3] - end_ptc[:, :3], axis=1)
    return fit_qe


def test_solve():
    """
    Notes:

    Robust Loss Lecture:
    https://www.cs.bgu.ac.il/~mcv172/wiki.files/Lec5.pdf

    Jon Barron's Paper:
    https://arxiv.org/pdf/1701.03077.pdf

    Thomas:
    # 1. try with L1 norm. weighting, with 1/e error. (convex)
    # 2. Ransac sampling to avoid local optima
    """

    success_rate = []
    guess_1 = []
    guess_2 = []
    for _ in tqdm(range(2000)):
        # generate a random transformation
        # rot = R.random(random_state=1234)
        rot = R.random()
        transform = np.eye(4)
        transform[:3, :3] = rot.as_matrix()
        transform[:3, 3] = [0.8, 0.5, 0.2]

        # generate a set of points
        # np.random.seed(1234)
        points = np.random.rand(25, 3)
        points = np.pad(points, ((0, 0), (0, 1)), mode="constant",
                        constant_values=1)

        # eval transform
        points2 = (transform @ points.T).T
        guess = solve_transform(points[:, :3], points2[:, :3])
        l_2 = np.linalg.norm(transform - guess)
        assert l_2 < 1e-5

        # print("l2 normal", l2)

        # reverse order of first 5 points
        points2 = (transform @ points.T).T
        points2[0:6] = points2[0:6][::-1]  # this should be an even number
        guess = solve_transform(points[:, :3], points2[:, :3])
        points2_guess = (guess @ points.T).T
        l2_g1 = np.linalg.norm(transform - guess)
        # print("l2 guess1:", l2_g1)

        # guess again
        error_threshold = 1e-4
        error = np.linalg.norm(points2 - points2_guess, axis=1)
        keep = error < error_threshold
        if np.sum(keep) < 6:
            keep = np.argsort(error)[:-6]

        points = points[keep]
        points2 = points2[keep]
        guess = solve_transform(points[:, :3], points2[:, :3])
        l2_g2 = np.linalg.norm(transform - guess)

        # print("l2 guess2:", l2_g2)
        success_rate.append(l2_g2 < l2_g1)
        guess_1.append(l2_g1)
        guess_2.append(l2_g2)

    print("g2 < g1", np.mean(success_rate))
    print("g1 mean", np.mean(guess_1))
    print("g2 mean", np.mean(guess_2))


if __name__ == "__main__":
    test_solve()
