class SystolicArray:
    def __init__(self, rows, cols):
        self.ROWS = rows  #size of matrix
        self.COLS = cols
        self.PEs = [[PE() for _ in range(cols)] for _ in range(rows)]
        self.cycle = 0  

    def step(self, row_in, col_in):
        # read every PE's inputs from the current registers first (the wires)...
        inputs = [[(col_in[c] if r == 0 else self.PEs[r - 1][c].get_bottom_reg(),
                    row_in[r] if c == 0 else self.PEs[r][c - 1].get_right_reg())
                   for c in range(self.COLS)] for r in range(self.ROWS)]

        # ...then clock all PEs
        for r in range(self.ROWS):
            for c in range(self.COLS):
                self.PEs[r][c].compute(*inputs[r][c])
        self.cycle += 1

    # [AI] begin: get_acc and __str__ written by AI
    def get_acc(self):
        return [[pe.acc for pe in row] for row in self.PEs]

    def __str__(self):
        # one cell per PE: acc, then the values it passes right (>) and down (v)
        cells = [[f"{pe.acc} >{pe.right_reg} v{pe.bottom_reg}" for pe in row] for row in self.PEs]
        width = max(len(cell) for row in cells for cell in row)
        lines = [f"cycle {self.cycle}"]
        for row in cells:
            lines.append(" | ".join(cell.rjust(width) for cell in row))
        return "\n".join(lines)
    # [AI] end


class PE: #multiply accumulate
    def __init__(self):
        self.acc = 0
        self.right_reg = 0
        self.bottom_reg = 0

    def compute(self, top_in, left_in):
        self.acc += top_in * left_in
        self.right_reg = left_in
        self.bottom_reg = top_in

    def get_right_reg(self):
        return self.right_reg

    def get_bottom_reg(self):
        return self.bottom_reg

    def reset(self):
        self.acc = 0
        self.right_reg = 0
        self.bottom_reg = 0