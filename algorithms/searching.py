import numpy as np

from utils import maths


def compute_search_action(p, c, i, zor, zoo, zoa, zoi):
    dist = np.sqrt(np.sum((p[:, :2] - p[i, :2]) ** 2, axis=1))
    i_r = np.logical_and(0.0 < dist, dist <= zor)
    if np.any(i_r):
        d = -np.sum(maths.normalize_matrix_rows(p[i_r, :2] - p[i, :2]), axis=0)
    else:
        i_ca = np.logical_and(c, dist <= zoi)
        if np.any(i_ca):
            d = np.sum(maths.normalize_matrix_rows(p[i_ca, :2] - p[i, :2]), axis=0)
        else:
            b = 0
            d = np.zeros(2)
            i_o = np.logical_and(zor < dist, dist <= zor + zoo)
            if np.any(i_o):
                b += 1
                d[0] = np.cos(p[i, 2]) + np.sum(np.cos(p[i_o, 2]), axis=0)
                d[1] = np.sin(p[i, 2]) + np.sum(np.sin(p[i_o, 2]), axis=0)
            i_a = np.logical_and(zor + zoo < dist, dist <= zor + zoo + zoa)
            if np.any(i_a):
                b += 1
                d += np.sum(maths.normalize_matrix_rows(p[i_a, :2] - p[i, :2]), axis=0)
            if b > 0:
                d /= b
    if np.any(d):
        d_theta = np.arctan2(d[1], d[0]) % (2 * np.pi)
    else:
        d_theta = 0.
    return d_theta, 0
