import time
import numpy as np
from itertools import product

def solve_sudoku(size, grid):
    row, column = size
    N = row * column
    X = ([("rc", rc) for rc in product(range(N), range(N))] +
         [("rn", rn) for rn in product(range(N), range(1, N + 1))] +
         [("cn", cn) for cn in product(range(N), range(1, N + 1))] +
         [("bn", bn) for bn in product(range(N), range(1, N + 1))])

    Y = dict()
    for r, c, n in product(range(N), range(N), range(1, N + 1)):
        b = (r // row) * row + (c // column)
        Y[(r, c, n)] = [
            ("rc", (r, c)),
            ("rn", (r, n)),
            ("cn", (c, n)),
            ("bn", (b, n))]
    X, Y = exact_cover(X, Y)
    for i, row in enumerate(grid):
        for j, n in enumerate(row):
            if n:
                select(X, Y, (i, j, n))
    for solution in solve(X, Y, []):
        for (r, c, n) in solution:
            grid[r][c] = n
        yield grid

def exact_cover(x, y):
    x = {j: set() for j in x}
    for i, row in y.items():
        for j in row:
            x[j].add(i)
    return x, y

def solve(x, y, solution):
    if not x:
        yield list(solution)
    else:
        c = min(x, key=lambda c: len(x[c]))
        for r in list(x[c]):
            solution.append(r)
            cols = select(x, y, r)
            for s in solve(x, y, solution):
                yield s
            deselect(x, y, r, cols)
            solution.pop()

def select(X, Y, r):
    cols = []
    for j in Y[r]:
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].remove(i)
        cols.append(X.pop(j))
    return cols

def deselect(X, Y, r, cols):
    for j in reversed(Y[r]):
        X[j] = cols.pop()
        for i in X[j]:
            for k in Y[i]:
                if k != j:
                    X[k].add(i)

def solve_wrapper(squares_num_array):
    if squares_num_array.count('0') >= 80:
        return None, None

    start = time.time()

    # convert string to 9x9 array
    arr = []
    for i in squares_num_array:
        arr.append(int(i))

    arr = np.array(arr, dtype=int)
    arr = np.reshape(arr, (9, 9))
    try:
        ans = list(solve_sudoku(size=(3, 3), grid=arr))[0]
        s = ""
        for a in ans:
            s += "".join(str(x) for x in a)
        return s, "Solved in %.4fs" % (time.time() - start)
    except:
        return None, None