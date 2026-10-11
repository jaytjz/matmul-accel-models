import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "SparseMatSparseMat"))
from SpMSpM import CSR, CSRWriter, Elem, FiberReader, Merger, RowLoader, ScaleAccumulate, SpMSpM

def dense_to_csr(M: list[list[float]]) -> CSR:
    row_ptr, col_idx, val = [0], [], []
    for row in M:
        for j, x in enumerate(row):
            if x != 0:
                col_idx.append(j)
                val.append(x)
        row_ptr.append(len(col_idx))
    return CSR(row_ptr, col_idx, val)

def test_row_loader(verbose=False):
    A = [[0, 0,  0, 0],   # empty first row
         [4, 0, -2, 0],   # ordinary row
         [0, 0,  0, 0],   # empty middle row
         [1, 5,  0, 7],   # exactly num_lanes nonzeros
         [0, 0,  0, 9],   # nonzero in the last column
         [0, 0,  0, 0]]   # empty last row
    A_csr = dense_to_csr(A)
    loader = RowLoader(A_csr, num_lanes=3)
    if verbose:
        print(f"A as CSR: {A_csr}")

    for i, row in enumerate(A):
        assert not loader.done, f"done too early, before row {i}"
        expected = [(a_col, a_rc) for a_col, a_rc in enumerate(row) if a_rc != 0]
        got = loader.load_row()
        if verbose:
            print(f"row {i}: {row} -> {got}")
        assert got == expected, f"row {i}"
    assert loader.done

    with pytest.raises(AssertionError) as e:
        loader.load_row()  # past the last row
    if verbose:
        print(f"past last row raised: {e.value}")

    with pytest.raises(AssertionError) as e:
        RowLoader(dense_to_csr([[1, 2, 3, 4]]), num_lanes=3).load_row()  # more nonzeros than lanes
    if verbose:
        print(f"too many nonzeros raised: {e.value}")

def test_fiber_reader(verbose=False):
    B = [[0, 0,  0, 0],   # empty first row
         [4, 0, -2, 0],   # ordinary row
         [0, 0,  0, 0],   # empty middle row
         [1, 5,  0, 7],   # several nonzeros
         [0, 0,  0, 9],   # nonzero in the last column
         [0, 0,  0, 0]]   # empty last row
    B_csr = dense_to_csr(B)
    reader = FiberReader(lane=2, B=B_csr)
    if verbose:
        print(f"B as CSR: {B_csr}")
        print(f"before any load, head: {reader.head()}")
    assert reader.head().eos, "unloaded lane should offer nothing"

    for k, row in enumerate(B):  # one reader reused for every row, as a lane is
        reader.load(B_csr.row_ptr[k], B_csr.row_ptr[k + 1])
        if verbose:
            print(f"row {k}: {row}, load({reader.ptr}, {reader.end})")
        expected = [(b_col, b_rc, 2) for b_col, b_rc in enumerate(row) if b_rc != 0]
        got = []
        while not (e := reader.head()).eos:
            assert reader.head() == e, "head() must not consume"
            got.append((e.col, e.val, e.lane))
            if verbose:
                print(f"  head: {e}, pop")
            reader.pop()
        if verbose:
            print(f"  head: {reader.head()}")
        assert got == expected, f"row {k}"
        assert reader.head().eos, f"row {k}: should stay at EOS"
        with pytest.raises(AssertionError) as err:
            reader.pop()  # pop after EOS
        if verbose:
            print(f"  pop after EOS raised: {err.value}")

    with pytest.raises(AssertionError) as err:
        reader.load(0, len(B_csr.col_idx) + 1)  # range past the end of B
    if verbose:
        print(f"range past end of B raised: {err.value}")

def test_merger(verbose=False):
    B = [[0, 2, 0, 3, 0],   # row 0
         [1, 0, 0, 4, 0],   # row 1: shares column 3 with row 0
         [0, 0, 0, 0, 0],   # row 2: empty
         [0, 5, 6, 7, 8],   # row 3: overlaps rows 0 and 1, reaches the last column
         [9, 0, 0, 0, 0]]   # row 4
    B_csr = dense_to_csr(B)
    K = 4
    readers = [FiberReader(lane=q, B=B_csr) for q in range(K)]
    merger = Merger(readers)
    if verbose:
        print(f"B as CSR: {B_csr}")

    # Each case is the rows of B a row of A would put on lanes 0, 1, ...; the rest stay empty
    cases = [[0, 1, 3],      # shared columns across three lanes, one lane unused
             [2, 4],         # an empty row of B on a lane
             [3],            # single lane
             [],             # empty row of A: EOS straight away
             [4, 2, 1, 0]]   # all lanes in use, out of order
    for rows in cases:  # same merger reused, as it is for every row of A
        for q, reader in enumerate(readers):
            if q < len(rows):
                reader.load(B_csr.row_ptr[rows[q]], B_csr.row_ptr[rows[q] + 1])
            else:
                reader.load(0, 0)  # unused lane
        if verbose:
            print(f"rows {rows}:")
            for q, reader in enumerate(readers):
                label = f"B row {rows[q]} = {B[rows[q]]}" if q < len(rows) else "unused"
                print(f"  lane {q}: load({reader.ptr}, {reader.end}), {label}")

        # every nonzero of every loaded row, in column order, equal columns lowest lane first
        expected = sorted((b_col, q, b_rc) for q, k in enumerate(rows)
                          for b_col, b_rc in enumerate(B[k]) if b_rc != 0)
        got = []
        while not (e := merger.head()).eos:
            assert merger.head() == e, "head() must not consume"
            got.append((e.col, e.lane, e.val))
            if verbose:
                lane_heads = ", ".join("EOS" if h.eos else f"{h.col}" for h in (r.head() for r in readers))
                print(f"  lane head cols [{lane_heads}] -> head: {e}, pop")
            merger.pop()
        if verbose:
            print(f"  head: {merger.head()}")
        assert got == expected, f"rows {rows}"
        assert merger.done
        merger.pop()  # popping EOS is a no op
        if verbose:
            print(f"  pop on EOS, head still: {merger.head()}")
        assert merger.head().eos

def test_scale_accumulate(verbose=False):
    EOS = Elem(col=0, val=0.0, lane=0, eos=True)
    sa = ScaleAccumulate(num_lanes=4)

    # Row with scales a_rc = 2, -1, 3 on lanes 0, 1, 2 (lane 3 unused).
    # Input is what the merger would send: sorted by col, ties lowest lane first.
    sa.load([2, -1, 3])
    if verbose:
        print(f"load([2, -1, 3]), scale table: {sa.scale}")
    stream = [Elem(0, 5, 1),   # -5 starts the acc
              Elem(1, 2, 0),   #  4 new col: flush (0, -5)
              Elem(1, 1, 2),   #  3 same col: acc 4 + 3 = 7
              Elem(3, 3, 0),   #  6 new col: flush (1, 7)
              Elem(3, 6, 1),   # -6 same col: acc 6 - 6 = 0
              Elem(4, 1, 2),   #  3 new col: flush the explicit zero (3, 0)
              EOS]             # flush (4, 3), row done
    expected = [(None, False),
                ((0, -5), False),
                (None, False),
                ((1, 7), False),
                (None, False),
                ((3, 0), False),
                ((4, 3), True)]
    def step(e):
        got = sa.step(e)
        if verbose:
            elem = "EOS" if e.eos else f"col {e.col}, val {e.val}, lane {e.lane} (x{sa.scale[e.lane]})"
            acc = f"(col {sa.acc_col}, val {sa.acc_val})" if sa.acc_valid else "empty"
            print(f"  step {elem} -> out {got[0]}, row_done {got[1]}, acc {acc}")
        return got

    got = [step(e) for e in stream]
    if verbose:
        print(f"  mults {sa.mults}, adds {sa.adds}")
    assert got == expected  # values and the cycle each comes out
    assert (sa.mults, sa.adds) == (6, 2)

    sa.load([])                                      # empty row of A
    if verbose:
        print("load([])")
    assert step(EOS) == (None, True)

    sa.load([4])                                     # nothing left over from earlier rows
    if verbose:
        print("load([4])")
    assert step(Elem(2, 0.5, 0)) == (None, False)
    assert step(EOS) == ((2, 2.0), True)

    with pytest.raises(AssertionError) as err:
        sa.step(Elem(0, 1, 0))                       # step after EOS without load
    if verbose:
        print(f"step after EOS without load raised: {err.value}")

    sa.load([1, 1])
    if verbose:
        print("load([1, 1])")
    step(Elem(3, 1, 0))
    with pytest.raises(AssertionError) as err:
        sa.step(Elem(2, 1, 1))                       # unsorted input
    if verbose:
        print(f"unsorted input raised: {err.value}")
    with pytest.raises(AssertionError) as err:
        sa.load([1])                                 # load before the row finished
    if verbose:
        print(f"load before row finished raised: {err.value}")

def test_csr_writer(verbose=False):
    w = CSRWriter()
    # (out, row_done) per cycle, exactly as ScaleAccumulate.step() returns it
    stream = [(None, True),        # row 0: empty, EOS straight away
              (None, False),       # row 1: acc filling, nothing out yet
              ((0, -5), False),
              (None, False),
              ((1, 7), False),
              ((3, 0), False),     # explicit zero is kept
              ((4, 3), True),      # last element and row end in the same cycle
              ((2, 2.0), True),    # row 2: single element, col 2 < row 1's last col 4
              (None, True)]        # row 3: empty last row
    for out, row_done in stream:
        w.step(out, row_done)
        if verbose:
            print(f"step({out}, {row_done}) -> row_ptr {w.row_ptr}, col_idx {w.col_idx}, val {w.val}")

    assert w.num_rows == 4
    C = w.result()
    if verbose:
        print(f"result: {C}")
    assert C == CSR(row_ptr=[0, 0, 4, 5, 5],
                    col_idx=[0, 1, 3, 4, 2],
                    val=[-5, 7, 0, 3, 2.0])

    w.step((5, 1), True)
    if verbose:
        print(f"after another step, writer row_ptr {w.row_ptr}, earlier result row_ptr {C.row_ptr}")
    assert C.row_ptr == [0, 0, 4, 5, 5], "result() must be a copy"

    w = CSRWriter()
    w.step((3, 1), False)
    with pytest.raises(AssertionError) as err:
        w.step((3, 2), False)      # duplicate column: accumulator failed to sum
    if verbose:
        print(f"duplicate column raised: {err.value}")
    w = CSRWriter()
    w.step((3, 1), False)
    with pytest.raises(AssertionError) as err:
        w.step((1, 1), False)      # column out of order within a row
    if verbose:
        print(f"column out of order raised: {err.value}")

def csr_to_dense(C: CSR, num_cols: int) -> list[list[float]]:
    M = [[0] * num_cols for _ in range(len(C.row_ptr) - 1)]
    for i in range(len(C.row_ptr) - 1):
        for t in range(C.row_ptr[i], C.row_ptr[i + 1]):
            M[i][C.col_idx[t]] = C.val[t]
    return M


@pytest.mark.parametrize("seed", range(50))
def test_spmspm(seed, verbose=False):
    rng = random.Random(seed)
    K = rng.randint(1, 4)
    m, n, p = rng.randint(1, 8), rng.randint(1, 8), rng.randint(1, 8)
    # small values so products often cancel to an explicit zero
    A = [[rng.choice([0, 0, rng.randint(-2, 2)]) for _ in range(n)] for _ in range(m)]
    B = [[rng.choice([0, 0, rng.randint(-2, 2)]) for _ in range(p)] for _ in range(n)]
    for row in A:  # keep every row of A within the lane count
        for j in [j for j, x in enumerate(row) if x][K:]:
            row[j] = 0
    A[0] = [0] * n   # always an empty row of A
    B[-1] = [0] * p  # always an empty row of B

    s = SpMSpM(dense_to_csr(A), dense_to_csr(B), num_lanes=K)
    if verbose:
        print(f"seed {seed}: K = {K} lanes, A is {m}x{n}, B is {n}x{p}")
        print("A =")
        for row in A:
            print(f"  {row}")
        print("B =")
        for row in B:
            print(f"  {row}")
    C = s.run(verbose)

    expected = [[sum(A[i][k] * B[k][j] for k in range(n)) for j in range(p)] for i in range(m)]
    if verbose:
        print(f"C as CSR: {C}")
        print("C =")
        for row in csr_to_dense(C, p):
            print(f"  {row}")
        print("expected =")
        for row in expected:
            print(f"  {row}")
    assert csr_to_dense(C, p) == expected
    for i in range(m):  # valid CSR: strictly ascending columns in every row
        cols = C.col_idx[C.row_ptr[i]:C.row_ptr[i + 1]]
        assert cols == sorted(set(cols)), f"row {i}"
    partial_products = sum(sum(1 for x in B[k] if x) for i in range(m) for k in range(n) if A[i][k])
    if verbose:
        print(f"cycles {s.cycles} = {partial_products} partial products + {m} EOS")
    assert s.cycles == partial_products + m  # one per partial product, one EOS per row
        
if __name__ == "__main__":
    test_row_loader()
    print()
    test_fiber_reader()
    print()
    test_merger()
    print()
    test_scale_accumulate(verbose=True)
    print()
    test_csr_writer()
    print()
    test_spmspm(9, verbose=True)
