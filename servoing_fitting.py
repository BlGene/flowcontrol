import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from pdb import set_trace


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' %
                  (method.__name__, (te - ts) * 1000))
        return result

    return timed


def solve_transform(p, q):
    # Find transformation s.t. (R|t) @ p == q

    # compute mean translation
    p_mean = p.mean(axis=0)
    o_mean = q.mean(axis=0)

    # whiten
    x = p - p_mean
    y = q - o_mean

    S = (x.T @ y)
    # assert S.shape == (4,4)

    d = S.shape[0]
    u, s, vh = np.linalg.svd(S)
    det = np.linalg.det(u @ vh)
    I = np.eye(d)
    I[-1, -1] = det

    R = (vh.T @ I @ u.T)
    t = o_mean - R @ p_mean

    R[:d-1, d-1] = t[:d-1]
    guess = R
    return guess


"""
Notes:

Robust Loss Lecture:
https://www.cs.bgu.ac.il/~mcv172/wiki.files/Lec5.pdf

Jon Barron's Paper:
https://arxiv.org/pdf/1701.03077.pdf
"""
if __name__ == "__main__":
    from scipy.spatial.transform import Rotation as R
    from tqdm import tqdm

    sr = []
    g1 = []
    g2 = []
    for i in tqdm(range(2000)):
        # generate a random transformation
        #rot = R.random(random_state=1234)
        rot = R.random()
        transform = np.eye(4)
        transform[:3, :3] = rot.as_matrix()
        transform[:3, 3] = [0.8, 0.5, 0.2]


        # generate a set of points
        #np.random.seed(1234)
        points = np.random.rand(25, 3)
        points = np.pad(points, ((0, 0), (0, 1)), mode="constant",
                        constant_values=1)

        # eval transform
        points2 =  (transform @ points.T).T
        guess = solve_transform(points, points2)
        l2 = np.linalg.norm(transform-guess)
        assert(l2 < 1e-5)


        #print("l2 normal", l2)

        # reverse order of first 5 points
        points2 =  (transform @ points.T).T
        points2[0:6] = points2[0:6][::-1]  # this should be an even number
        guess = solve_transform(points, points2)
        points2_guess = (guess @ points.T).T
        l2_g1 = np.linalg.norm(transform-guess)
        #print("l2 guess1:", l2_g1)


        # guess again
        error_threshold = 1e-4
        error = np.linalg.norm(points2-points2_guess, axis=1)
        keep = error < error_threshold
        if np.sum(keep) < 6:
            keep = np.argsort(error)[:-6]

        points = points[keep]
        points2 = points2[keep]
        guess = solve_transform(points, points2)
        l2_g2 = np.linalg.norm(transform-guess)


        #print("l2 guess2:", l2_g2)
        sr.append(l2_g2 < l2_g1)
        g1.append(l2_g1)
        g2.append(l2_g2)

    print("g2 < g1", np.mean(sr))
    print("g1 mean", np.mean(g1))
    print("g2 mean", np.mean(g2))
