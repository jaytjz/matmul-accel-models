import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "SparseMatDenseMat"))
from SpMM import CSR, PE, CSRWalker, StripMajorMem, SpMM, Word

def test_strip_major_mem():
    # B (2 x 5) with 2 lanes: 3 strips, the last one is 1 real column + 1 padding
    B = [[1.0, 2.0, 3.0, 4.0,  5.0],
         [6.0, 7.0, 8.0, 9.0, 10.0]]
    mem = StripMajorMem(B, lanes=2)

    assert mem.strips == 3
    assert mem.mem == [[1, 2], [6, 7],               # strip 0: cols 0-1
                       [3, 4], [8, 9],               # strip 1: cols 2-3
                       [5, 0], [10, 0]]              # strip 2: col 4 + padding
    assert [mem.base(s) for s in range(3)] == [0, 2, 4]

    # row k of strip s is at base(s) + k, holding B[k][s*lanes : (s+1)*lanes]
    assert mem[mem.base(1) + 0] == [3, 4]
    assert mem[mem.base(2) + 1] == [10, 0]


def test_csr_walker():
    # A (3 x 3), with an empty middle row
    #   [[0, 2, 0],
    #    [0, 0, 0],
    #    [1, 0, 3]]
    A = CSR(row_ptr=[0, 1, 1, 3], col_idx=[1, 0, 2], val=[2.0, 1.0, 3.0])
    walker = CSRWalker(A)

    expected = [Word(last=True,  col=1, val=2.0),    # row 0: single nonzero, so it's also the last
                Word(last=True,  col=0, val=0.0),    # row 1: empty -> nop that still closes the row
                Word(last=False, col=0, val=1.0),    # row 2: first nonzero
                Word(last=True,  col=2, val=3.0)]    # row 2: last nonzero

    words = []
    while not walker.done:
        words.append(walker.step())
    assert words == expected                         # one word per nonzero + one per empty row

    # reset replays the same stream (done once per strip of B)
    walker.reset()
    assert [walker.step() for _ in expected] == expected
    assert walker.done


def test_pe():
    # strip 1 of B = [[1, 2, 3, 4], [5, 6, 7, 8]] holds cols 2-3: [[3, 4], [7, 8]]
    mem = StripMajorMem([[1.0, 2.0, 3.0, 4.0],
                         [5.0, 6.0, 7.0, 8.0]], lanes=2)
    pe = PE(mem, lanes=2)
    pe.base = mem.base(1)

    # C row = 2 * B[0][2:4] + 1 * B[1][2:4]
    assert pe.step(Word(last=False, col=0, val=2.0)) is None     # accumulate, nothing out yet
    assert pe.accs == [6, 8]
    assert pe.step(Word(last=True, col=1, val=1.0)) == [13, 16]  # last word emits the row
    assert pe.accs == [0, 0]                                     # and clears the accumulators

    # a nop (empty row of A) emits a zero row
    assert pe.step(Word(last=True, col=0, val=0.0)) == [0, 0]


def test_example(verbose=False):
    # A (3 x 3), with an empty middle row
    #   [[2, 0, 1],
    #    [0, 0, 0],
    #    [0, 3, 0]]
    A = CSR(row_ptr=[0, 2, 2, 3], col_idx=[0, 2, 1], val=[2.0, 1.0, 3.0])

    B = [[1.0,  2.0,  3.0,  4.0],                # 3 x 4
         [5.0,  6.0,  7.0,  8.0],
         [9.0, 10.0, 11.0, 12.0]]

    m = SpMM(A, B, lanes=2)

    # # B is stored as 2 strips of 2 columns, address = s * B_rows + k
    assert m.B_mem.mem == [[1, 2], [5, 6], [9, 10],      # strip 0: cols 0-1
                       [3, 4], [7, 8], [11, 12]]     # strip 1: cols 2-3

    C = m.run(verbose)

    assert C == [[11, 14, 17, 20],
                 [0, 0, 0, 0],
                 [15, 18, 21, 24]]
    # 3 nonzeros + 1 nop for the empty row, replayed for each of 2 strips
    assert m.cycles == 8


def test_padded_strip():
    # B has 3 columns, so with 2 lanes the last strip is zero-padded
    A = CSR(row_ptr=[0, 1, 2], col_idx=[1, 0], val=[2.0, 1.0])   # [[0, 2], [1, 0]]
    B = [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]

    m = SpMM(A, B, lanes=2)

    assert m.B_mem.mem == [[1, 2], [4, 5],               # strip 0: cols 0-1
                       [3, 0], [6, 0]]               # strip 1: col 2 + padding
    assert m.run() == [[8, 10, 12],
                       [1, 2, 3]]
    assert m.cycles == 4


def to_csr(dense):
    row_ptr, col_idx, val = [0], [], []
    for row in dense:
        for k, x in enumerate(row):
            if x != 0:
                col_idx.append(k)
                val.append(float(x))
        row_ptr.append(len(val))
    return CSR(row_ptr, col_idx, val)


def test_big_B(verbose=False):
    # B (7 x 31) with 8 lanes: 4 strips, the last one has 7 real columns + 1 padding
    rng = np.random.default_rng(0)
    B = rng.integers(-9, 10, size=(7, 31)).astype(float)
    A = rng.integers(-9, 10, size=(5, 7)) * (rng.random((5, 7)) < 0.3)   # ~30% dense
    A[3] = 0                                                             # one empty row

    m = SpMM(to_csr(A), B.tolist(), lanes=16)
    C = m.run(verbose)

    assert np.array_equal(C, A @ B)


if __name__ == "__main__":
    test_strip_major_mem()
    test_csr_walker()
    test_pe()
    test_example()
    test_padded_strip()
    test_big_B()
    print("PASS")
