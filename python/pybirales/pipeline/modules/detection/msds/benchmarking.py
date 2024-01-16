import numpy as np
from line_profiler import LineProfiler
from numba import njit

from profiler import Profiler
from util import __ir


@njit("float64[:, :](float64[:,:, :])")
def compute_dm(X):
    min_r = 0.001
    l = X.shape[0]
    m = X.shape[1]

    # dm = np.full((405, (m * (m - 1)) // 2), 10000, dtype=np.float)

    dm = np.zeros((l, (m * (m - 1)) // 2), dtype=np.float) + 10000

    for row in range(l):
        k = 0
        for i in range(0, m - 1):
            for j in range(i + 1, m):
                diff = X[row, i] - X[row, j]

                # perfectly horizontal and distance is large  (> 1 px)
                if diff[0] == 0 and abs(diff[1]) > 1:
                    dm[row, k] = 10000
                # vertical
                elif diff[1] == 0 and abs(diff[0]) > 1:
                    dm[row, k] = 10000
                else:
                    diff += 1e-6
                    if min_r <= (diff[0] / diff[1]) <= 0:
                        dm[row, k] = (diff[0] ** 2 + diff[1] ** 2) ** 0.5
                k = k + 1
    return dm


# @njit("float64[:, :](float64[:,:, :])")
def compute_dm_2(X):
    min_r = 0.001
    l, m, n = X.shape

    dm = np.zeros((l, (m * (m - 1)) // 2), dtype=float) + 10000

    # Compute pairwise differences
    diff = X[:, np.newaxis] - X[:, :, np.newaxis]
    diff[:, :, 0] += 1e-6  # Add small epsilon to avoid division by zero

    # Compute Euclidean distance for non-horizontal and non-vertical differences
    mask = (diff[:, :, 1] != 0) & (np.abs(diff[:, :, 0] / diff[:, :, 1]) >= min_r)
    dm[:, mask] = np.linalg.norm(diff[:, :, :], axis=-1)[mask]

    return dm


# @cuda.jit("(float64[:,:, :], int64, int64,float64[:, :])")
# def compute_cp(X, l, m, dm):
#     min_r = 0.001
#
#     for row in range(l):
#         k = 0
#         for i in range(0, m - 1):
#             for j in range(i + 1, m):
#                 diff = X[row, i] - X[row, j]
#
#                 # perfectly horizontal and distance is large  (> 1 px)
#                 if diff[0] == 0 and abs(diff[1]) > 1:
#                     dm[row, k] = 10000
#                 # vertical
#                 elif diff[1] == 0 and abs(diff[0]) > 1:
#                     dm[row, k] = 10000
#                 else:
#                     diff += 1e-6
#                     if min_r <= (diff[0] / diff[1]) <= 0:
#                         dm[row, k] = (diff[0] ** 2 + diff[1] ** 2) ** 0.5
#                 k = k + 1


def ir2(data, min_n=10, i=None):
    if len(data) < min_n and len(np.unique(data[:, 0])) != 1:
        b = data[:, :2] - np.mean(data[:, :2], axis=0)
        coords = np.flip(b.T, axis=0)
        eigen_values, eigen_vectors = np.linalg.eig(np.cov(coords))

        sort_indices = np.argsort(eigen_values)[::-1]

        p_v1 = eigen_vectors[sort_indices[0]]

        # primary eigenvector is perfectly horizontal or perfectly vertical
        if p_v1[0] == 0 or p_v1[1] == 0:
            return 1

        # Gradient of 1st eigenvector
        m1 = -1 * p_v1[1] / p_v1[0]

        return np.abs(eigen_values[sort_indices[1]] / eigen_values[sort_indices[0]]) * np.sign(m1)

    return 0.0


def test_dm():
    X = np.load('fixtures/leave_data.npy')

    profiler = Profiler()

    profiler.start()
    a = compute_dm(X)
    profiler.add_timing('compute_dm')

    profiler.reset()
    b = compute_dm_2(X)
    profiler.add_timing('compute_dm - optimised')

    # print((a == b).all())

    # profiler.reset()
    # l = X.shape[0]
    # m = X.shape[1]
    # dm = cp.zeros((l, (m * (m - 1)) // 2), dtype=float) + 10000
    # compute_cp(X, l, m, dm)
    # profiler.add_timing('compute_dm - cupy')

    profiler.show()


def test_ir():
    C = np.load('fixtures/ir_data.npy')

    profiler = Profiler()

    profiler.start()
    a = __ir(C)
    profiler.add_timing('ir.org')

    profiler.start()
    a = ir2(C)
    profiler.add_timing('ir2')

    lp = LineProfiler()
    lp_wrapper = lp(ir2)
    lp_wrapper(C)
    lp.print_stats()

    profiler.show()


if __name__ == '__main__':
    # test_dm()

    test_ir()
