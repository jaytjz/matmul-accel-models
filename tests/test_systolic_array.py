# [AI] This whole file was written by AI.

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "DenseMatDenseMat"))
from SystolicArray import SystolicArray


def run_matmul(A, B, verbose=False):
    """Feed A from the left and B from the top with skew; return (C, array)."""
    M, K = A.shape
    _, N = B.shape
    s = SystolicArray(M, N)
    steps = K + M + N - 2  # last MAC lands at cycle (K-1)+(M-1)+(N-1)
    for t in range(steps):
        row_in = [int(A[r, t - r]) if 0 <= t - r < K else 0 for r in range(M)]
        col_in = [int(B[t - c, c]) if 0 <= t - c < K else 0 for c in range(N)]
        s.step(row_in, col_in)
        if verbose:
            print(s, end="\n\n")
    return np.array(s.get_acc()), s


def test_2x2():
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    C, s = run_matmul(A, B, True)
    assert (C == A @ B).all(), f"\n{s}\nexpected\n{A @ B}"


def test_3x3():
    A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
    C, s = run_matmul(A, B)
    assert (C == A @ B).all(), f"\n{s}\nexpected\n{A @ B}"


if __name__ == "__main__":
    test_2x2()
    test_3x3()
    print("PASS")
