from dataclasses import dataclass
from math import ceil

@dataclass
class CSR:
    row_ptr: list[int]
    col_idx: list[int]
    val: list[float]

@dataclass
class Word:
    last: bool
    col: int 
    val: float

class PE:
    def __init__(self, B_mem, lanes):
        self.B_mem = B_mem
        self.lanes = lanes
        self.accs = [0] * lanes
        self.base = 0  # B_mem address of the current strip

    def step(self, w: Word) -> list[float] | None:
        b = self.B_mem[self.base + w.col]
        for q in range(self.lanes):
            self.accs[q] += w.val * b[q]
        if w.last:
            row, self.accs = self.accs, [0] * self.lanes
            return row
        return None

class CSRWalker:
    """Hardware CSR walker: emits one Word per cycle."""
    def __init__(self, A: CSR):
        self.A = A
        self.num_rows = len(A.row_ptr) - 1
        self.reset()

    def reset(self):
        self.row_index = 0
        self.nonzero_index = self.A.row_ptr[0] #position in col_idx / val
        self.row_end = self.A.row_ptr[1] if self.num_rows else self.nonzero_index

    @property
    def done(self) -> bool:
        return self.row_index == self.num_rows

    def _advance_row(self):
        self.row_index += 1
        if not self.done:
            self.row_end = self.A.row_ptr[self.row_index + 1]

    def step(self) -> Word:
        if self.nonzero_index == self.row_end:
            self._advance_row()
            return Word(last=True, col=0, val=0.0)
        is_last_in_row = self.nonzero_index + 1 == self.row_end
        word = Word(
            last=is_last_in_row,
            col=self.A.col_idx[self.nonzero_index],
            val=self.A.val[self.nonzero_index]
        )
        self.nonzero_index += 1
        if is_last_in_row:
            self._advance_row()
        return word

class StripMajorMem:
    """B split into lane-wide column strips; row k of strip s lives at address s * B_rows + k."""
    def __init__(self, B: list[list[float]], lanes: int):
        self.B_rows = len(B)
        self.B_cols = len(B[0]) if B else 0
        self.lanes = lanes
        self.strips = ceil(self.B_cols / lanes)
        self.mem = []
        for s in range(self.strips):
            for k in range(self.B_rows):
                seg = B[k][s * lanes:(s + 1) * lanes]
                seg = seg + [0.0] * (lanes - len(seg))
                self.mem.append(seg)

    def base(self, s: int) -> int:
        """Address of the first row of strip s."""
        return s * self.B_rows

    def __getitem__(self, addr: int) -> list[float]:
        return self.mem[addr]

class SpMM:
    def __init__(self, A: CSR, B: list[list[float]], lanes: int):
        self.B_mem = StripMajorMem(B, lanes)
        self.B_rows = self.B_mem.B_rows
        self.B_cols = self.B_mem.B_cols
        self.lanes = lanes 
        self.strips = self.B_mem.strips
        self.pe = PE(self.B_mem, lanes)
        self.csr_walker = CSRWalker(A)
        self.cycles = 0

    def step(self):
        # [AI] begin: trace state recorded for __str__
        w = self.csr_walker
        self.a_row = w.row_index
        self.a_idx = w.nonzero_index
        self.nop = w.nonzero_index == w.row_end    # empty row: walker emits a nop
        # [AI] end
        self.word = self.csr_walker.step()
        self.cycles += 1
        self.out = self.pe.step(self.word)
        return self.out

    def run(self, verbose=False) -> list[list[float]]:
        A_rows = self.csr_walker.num_rows
        C = [[0.0] * self.B_cols for _ in range(A_rows)]
        for s in range(self.strips):
            self.pe.base = self.B_mem.base(s)
            self.csr_walker.reset()            # replay A for this strip
            lo = s * self.lanes
            hi = min(lo + self.lanes, self.B_cols)
            if verbose:
                print(f"strip {s}: C cols {lo}-{hi - 1}, pe.base = {self.pe.base}, walker reset")
            i = 0
            while not self.csr_walker.done:
                row = self.step()
                if verbose:
                    print(self)
                if row is not None:
                    C[i][lo:hi] = row[:hi - lo]
                    i += 1
            assert i == A_rows, f"strip {s}: got {i} rows, expected {A_rows}"
        return C

    # [AI] begin: __str__ written by AI
    def __str__(self):
        # one line per cycle: word from the walker, B row read, accumulators after the MAC
        w = self.word
        lo = self.pe.base // self.B_rows * self.lanes
        b_row = f"B[{w.col}][{lo}:{lo + self.lanes}]={self.B_mem[self.pe.base + w.col]}"
        if self.nop:
            mac = f"A[{self.a_row}] empty -> nop  0.0 x {b_row}"
        else:
            mac = f"A[{self.a_row}][{w.col}]={w.val} (nz {self.a_idx}) x {b_row}"
        if self.out is not None:
            acc = f"acc {self.out} -> C[{self.a_row}][{lo}:{min(lo + self.lanes, self.B_cols)}], acc cleared"
        else:
            acc = f"acc {self.pe.accs}"
        return f"cycle {self.cycles:>2} | {'last' if w.last else '    '} | {mac} | {acc}"
    # [AI] end
