import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import time
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


def test_transform(points, transform, plot=False):
    points = np.pad(points, ((0, 0), (0, 1)),
                    mode="constant", constant_values=1)

    # first transform points
    observations = (transform @ points.T).T

    guess = solve_transform(points, observations)

    print(transform)
    print(guess)
    print(np.linalg.norm(transform-guess))
    print()

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2])
        ax.scatter(observations[:, 0], observations[:, 1], observations[:, 2])
        plt.show()


def test_90_rot():
    points = np.array([[.5, -1, 0, 0],
                       [0.5, -2, 0, 0],
                       [-0.5, -2, 0, 0],
                       [-0.5, -1, 0, 0]])

    r = R.from_euler('z', 90, degrees=True)
    rot_z = np.eye(4)
    rot_z[:3, :3] = r.as_dcm()

    print(rot_z.shape)
    print(points.shape)

    observations = (rot_z @ points.T).T

    print(observations.shape)

    observations[:, 0] += 1
    print(observations)
    guess = solve_transform(points, observations)
    print(guess)

    euler = R.from_dcm(guess[:3, :3]).as_euler(seq="xyz", degrees=True)

    print(euler)


if __name__ == "__main__":
    test_transform(triangle, trn_x)
    test_transform(points, rot_z)

    Q = trn_x @ rot_z @ np.linalg.inv(trn_x)
    test_transform(points, Q)

    points = np.pad(points, ((0, 0), (0, 1)), mode="constant",
                    constant_values=1)
    source = (trn_x @ points.T).T
    target = (trn_x @ rot_z @ points.T).T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(source[:, 0], source[:, 1], source[:, 2])
    ax.scatter(target[:, 0], target[:, 1], target[:, 2])
    ax.scatter([0], [0], [0])
    plt.show()
