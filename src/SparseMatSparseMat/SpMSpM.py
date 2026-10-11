from dataclasses import dataclass

@dataclass
class CSR:
    row_ptr: list[int]
    col_idx: list[int]
    val: list[float]

@dataclass
class Elem:
    col: int      # column j of B (and so of C). The merger sorts on this
    val: float    # b_kj, not yet scaled
    lane: int     # which FiberReader it came from, used to look up a_ik
    eos: bool = False  # end of stream: col and val are ignored

class RowLoader:
    """Hands out one row of A at a time as (a_col, a_rc) pairs."""
    def __init__(self, A: CSR, num_lanes: int):
        self.A = A
        self.num_lanes = num_lanes
        self.num_rows = len(A.row_ptr) - 1
        self.row = 0

    @property
    def done(self) -> bool:
        return self.row == self.num_rows

    def load_row(self) -> list[tuple[int, float]]:
        assert not self.done, "load_row called after last row"
        lo, hi = self.A.row_ptr[self.row], self.A.row_ptr[self.row + 1]
        assert hi - lo <= self.num_lanes, f"row {self.row} has {hi - lo} nonzeros, only {self.num_lanes} lanes"
        self.row += 1
        return list(zip(self.A.col_idx[lo:hi], self.A.val[lo:hi]))

class FiberReader:
    """One lane: walks one row of B and presents it to the merger one element at a time."""

    def __init__(self, lane: int, B: CSR):
        self.lane = lane
        self.B = B
        self.load(0, 0)  # start empty: head() is EOS until loaded

    def load(self, start: int, end: int):
        assert 0 <= start <= end <= len(self.B.col_idx), f"lane {self.lane}: bad range {start}:{end}"
        self.ptr, self.end = start, end

    @property
    def done(self) -> bool:
        return self.ptr == self.end

    def head(self) -> Elem:
        """Current element, not consumed. EOS once the row is exhausted."""
        if self.done:
            return Elem(col=0, val=0.0, lane=self.lane, eos=True)
        return Elem(self.B.col_idx[self.ptr], self.B.val[self.ptr], self.lane)

    def pop(self):
        """Consume the head. Only the lane the merger picked gets popped."""
        assert not self.done, f"lane {self.lane}: pop after EOS"
        self.ptr += 1

class Merger:
    """Behavioral K way merger: offers the smallest column across all lane heads.
    Same head()/pop() interface as FiberReader, so a structural MergeTree can replace it later."""

    def __init__(self, readers: list[FiberReader]):
        assert readers, "need at least one lane"
        assert all(r.lane == q for q, r in enumerate(readers)), "readers[q] must be lane q"
        self.readers = readers

    @property
    def done(self) -> bool:
        return all(r.done for r in self.readers)

    def head(self) -> Elem:
        """Smallest column across all lane heads, not consumed. EOS sorts after every real
        column, so this is EOS exactly when every lane is finished. Ties go to the lowest lane."""
        return min((r.head() for r in self.readers), key=lambda e: (e.eos, e.col, e.lane))

    def pop(self):
        """Consume the head by popping only the lane it came from. Popping EOS does nothing:
        the consumer pops every head it takes, EOS included, which a merge tree's root FIFO
        will need later."""
        e = self.head()
        if not e.eos:
            self.readers[e.lane].pop()

class ScaleAccumulate:
    """Scales each merged element by its lane's a_rc and sums runs of equal column.
    The input arrives sorted by column, so one (col, val) register is enough."""

    def __init__(self, num_lanes: int):
        self.num_lanes = num_lanes
        self.scale = [0.0] * num_lanes   # a_rc per lane, the scale table
        self.acc_valid = False           # acc register: is it holding a partial sum?
        self.acc_col = 0
        self.acc_val = 0.0
        self.row_done = True             # nothing loaded yet
        self.mults = 0
        self.adds = 0

    def load(self, scales: list[float]):
        """Start of a row: one a_rc per used lane, unused lanes left at 0."""
        assert self.row_done, "load before the previous row finished"
        assert len(scales) <= self.num_lanes, f"{len(scales)} scales, only {self.num_lanes} lanes"
        self.scale = scales + [0.0] * (self.num_lanes - len(scales))
        self.row_done = False

    def step(self, e: Elem) -> tuple[tuple[int, float] | None, bool]:
        """One cycle. Returns (finished element of C or None, row done)."""
        assert not self.row_done, "step after EOS without a new load"
        if e.eos:
            out = self._flush()
            self.row_done = True
            return out, True
        assert not (self.acc_valid and e.col < self.acc_col), \
            f"input not sorted: col {e.col} after col {self.acc_col}"
        prod = e.val * self.scale[e.lane]
        self.mults += 1
        if self.acc_valid and e.col == self.acc_col:
            self.acc_val += prod         # same output column: keep summing
            self.adds += 1
            return None, False
        out = self._flush()              # new column: previous sum is finished
        self.acc_valid, self.acc_col, self.acc_val = True, e.col, prod
        return out, False

    def _flush(self) -> tuple[int, float] | None:
        if not self.acc_valid:
            return None
        self.acc_valid = False
        return (self.acc_col, self.acc_val)

class CSRWriter:
    """Builds C in CSR from the stream ScaleAccumulate emits."""

    def __init__(self):
        self.row_ptr = [0]
        self.col_idx = []
        self.val = []
        self.last_col = -1   # last column written in the current row

    @property
    def num_rows(self) -> int:
        return len(self.row_ptr) - 1

    def step(self, out: tuple[int, float] | None, row_done: bool):
        """One cycle: takes exactly what ScaleAccumulate.step() returned."""
        if out is not None:
            col, v = out
            assert col > self.last_col, f"row {self.num_rows}: col {col} after col {self.last_col}"
            self.col_idx.append(col)
            self.val.append(v)
            self.last_col = col
        if row_done:  # after the append: on EOS the last element and the row end arrive together
            self.row_ptr.append(len(self.col_idx))
            self.last_col = -1

    def result(self) -> CSR:
        return CSR(list(self.row_ptr), list(self.col_idx), list(self.val))

class SpMSpM:
    """Gustavson SpMSpM, one PE with num_lanes lanes:
    row i of C = sum over the nonzeros (a_col, a_rc) of row i of A of a_rc * B[a_col, :]."""

    def __init__(self, A: CSR, B: CSR, num_lanes: int):
        assert max(A.col_idx, default=-1) < len(B.row_ptr) - 1, "A has a column with no matching row of B"
        self.A, self.B = A, B
        self.num_lanes = num_lanes
        self.loader = RowLoader(A, num_lanes)
        self.readers = [FiberReader(q, B) for q in range(num_lanes)]
        self.merger = Merger(self.readers)
        self.sa = ScaleAccumulate(num_lanes)
        self.writer = CSRWriter()
        self.cycles = 0
        self.row = -1      # row of A currently streaming
        self.pairs = []    # its (a_col, a_rc) pairs, for the trace

    def load_row(self):
        """Row start, zero cycles for now: one lane per nonzero of A, scale table gets the a_rc values."""
        self.row = self.loader.row
        self.pairs = self.loader.load_row()
        for q, reader in enumerate(self.readers):
            if q < len(self.pairs):
                a_col, _ = self.pairs[q]
                reader.load(self.B.row_ptr[a_col], self.B.row_ptr[a_col + 1])  # column of A selects row of B
            else:
                reader.load(0, 0)  # unused lane, emits EOS immediately
        self.sa.load([a_rc for _, a_rc in self.pairs])

    def step(self):
        """One cycle: merger -> ScaleAccumulate -> CSRWriter."""
        self.e = self.merger.head()
        self.merger.pop()
        self.out, self.row_done = self.sa.step(self.e)
        self.writer.step(self.out, self.row_done)
        self.cycles += 1

    def run(self, verbose=False) -> CSR:
        while not self.loader.done:
            self.load_row()
            if verbose:
                lanes = ", ".join(f"lane {q} <- B[{c}] x {a}" for q, (c, a) in enumerate(self.pairs))
                print(f"row {self.row}: {lanes or 'empty'}")
            while not self.sa.row_done:
                self.step()
                if verbose:
                    print(self)
        assert self.writer.num_rows == self.loader.num_rows, "C and A have different row counts"
        return self.writer.result()

    def __str__(self):
        # one line per cycle: what left the merger, the acc after this cycle, what was written to C
        e = self.e
        if e.eos:
            got = "EOS"
        else:
            got = f"lane {e.lane}: col {e.col} val {e.val} x {self.sa.scale[e.lane]}"
        acc = f"acc (col {self.sa.acc_col}, {self.sa.acc_val})" if self.sa.acc_valid else "acc empty"
        out = f"C[{self.row}][{self.out[0]}] = {self.out[1]}" if self.out else ""
        done = f"row {self.row} done" if self.row_done else ""
        tail = ", ".join(x for x in (out, done) if x)
        return f"cycle {self.cycles:>3} | {got:<28} | {acc:<20} | {tail}".rstrip()