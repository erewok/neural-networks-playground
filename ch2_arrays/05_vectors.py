"""
Stage 5: vectors and matrices, which is the notation everything else is in.

Chapter 1 taught neural networks one number at a time. Every paper you want to
read writes them a layer at a time instead, in matrices, and the translation is
not hard -- but until you have done it yourself the equations stay opaque.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

That is the whole of the thing you are aiming at, and four of the five symbols
in it are matrices. This file is the vocabulary.

THE FIVE FUNCTIONS

    dot()        two vectors in, ONE NUMBER out
    matvec()     a matrix and a vector in, a vector out
    matmul()     two matrices in, a matrix out
    transpose()  flip rows and columns
    add_bias()   add one number per column to every row

Only the first one is really an operation. The other four are dot() applied in
a pattern, which is worth holding on to when matmul starts looking mysterious:
it is just a lot of dot products arranged in a grid.

Properties they are supposed to have. Every one is visible in the tables.

  1. dot() is the only thing here that turns two vectors into a single number.
     Everything else keeps you inside arrays.

  2. matmul is NOT elementwise multiplication. This is the single most common
     wrong idea about it. (A @ B)[i][j] is the dot product of ROW i of A with
     COLUMN j of B -- one number in the answer draws on a whole row and a
     whole column, not on the one number sitting in the same place.

  3. Shapes are the thing to get fluent in. An (n x k) times a (k x m) gives
     an (n x m): the inner dimensions have to agree, and they disappear. If
     you can predict the output shape without doing the arithmetic, you have
     understood the operation. Some rows below are non-square on purpose --
     with square matrices a shape mistake can hide.

  4. Order matters. A @ B and B @ A are different matrices, and often only one
     of the two is even legal.

  5. A bias has one number PER COLUMN, and it is added to every row. Columns
     are output units and rows are examples, so the bias belongs to the unit,
     not to the example. Getting this backwards is the classic broadcasting
     bug and it does not always raise -- on a square matrix it silently
     computes the wrong thing.

  6. transpose() looks like the least interesting function here. It is not:
     it is what lets the same weights be used in both directions, which is
     exactly what the backward pass in 06 needs.

The integration test at the bottom takes 03's forward pass -- the loops you
already wrote -- and runs it beside the same computation built out of your five
functions, on random weights. They have to agree to the last decimal place.
That is the entire point of this file: the matrix version is not a different
algorithm, it is the same arithmetic with the loops moved.

`just plot 5` draws where one number in a matmul comes from.
"""

import math
import random
import sys

from harness import (CROSS, TICK, Cell, ExerciseTable, approx_nested, console,
                     fmt_grid, not_written, section, summary)


def sigmoid(z):
    """Copied from 03 so this file stands alone. Nothing to fix here."""
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


# --------------------------------------------------------------- the originals
# Left here on purpose. Don't edit these -- the harness runs them beside yours.


def dot_broken(u, v):
    return sum(u) * sum(v)


def matvec_broken(M, v):
    return [sum(M[k][i] * v[k] for k in range(len(M)))
            for i in range(len(M[0]))]


def matmul_broken(A, B):
    return [[a * b for a, b in zip(row_a, row_b)] for row_a, row_b in zip(A, B)]


def transpose_broken(M):
    return [list(row) for row in M][::-1]


def add_bias_broken(M, b):
    return [[v + b[i] for v in row] for i, row in enumerate(M)]


# ---------------------------------------------------------------------- yours
# Write these five. The tables score whichever ones exist, so go one at a time.
# Return lists, not tuples -- the tables compare shapes as well as numbers.


def dot(u, v):
    """Two vectors in, one number out."""
    raise NotImplementedError


def matvec(M, v):
    """M is (n x k), v has k entries. Out: a vector of n entries.

    Entry i of the answer is the dot product of row i of M with v.
    """
    raise NotImplementedError


def matmul(A, B):
    """A is (n x k), B is (k x m). Out: an (n x m) matrix.

    Entry [i][j] is the dot product of row i of A with column j of B.
    """
    raise NotImplementedError


def transpose(M):
    """An (n x m) matrix in, an (m x n) matrix out. Rows become columns."""
    raise NotImplementedError


def add_bias(M, b):
    """M is (n x m), b has m entries -- one per COLUMN. Out: an (n x m) matrix.

    The same b is added to every row.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- test harness
# Drawing only -- see harness.py. Nothing here needs fixing.


def is_written(fn, probe_args):
    try:
        fn(*probe_args)
    except NotImplementedError:
        return False
    except Exception:
        # Written, and wrong in some other way. That is a table row, not a
        # reason to hide the whole column.
        return True
    return True


def attempt(fn, args):
    """Run one case. Shape mistakes raise, and a raise is a failing row.

    From here on the exercises deal in shapes, and the commonest mistake --
    pairing the wrong dimensions -- shows up as an IndexError rather than a
    wrong number. Catching it keeps that in the table where you can see it.
    """
    try:
        return Cell(fn(*args))
    except Exception as exc:
        return Cell(None, detail=type(exc).__name__)


def compare(title, original, mine, argnames, cases, note=""):
    """cases: list of (args_tuple, expected)."""
    written = is_written(mine, cases[0][0])
    t = ExerciseTable(
        title=title,
        note=note,
        input_header="arguments",
        yours_written=written,
        fmt=fmt_grid(0, signed=False),
        compare=approx_nested,
    )
    for args, want in cases:
        show = fmt_grid(0, signed=False)
        label = ", ".join(f"{n}={show(v)}" for n, v in zip(argnames, args))
        t.add(label, want, attempt(original, args),
              attempt(mine, args) if written else None)
    return t.render()


def claim(text, holds):
    console.print(f"  [{'bold green' if holds else 'bold red'}]"
                  f"{TICK if holds else CROSS}[/] {text}")
    return holds


# ------------------------------------------------------- the integration test
# 03's forward pass, twice: once the way you wrote it, once out of your five
# functions. Same arithmetic, same numbers, loops in a different place.


def forward_loops(x, w_hidden, b_hidden, w_out, b_out):
    """Exactly 03's forward(), copied. This is the thing being matched."""
    h = []
    for w_row, b in zip(w_hidden, b_hidden):
        h.append(sigmoid(b + sum(xi * wi for xi, wi in zip(x, w_row))))
    return sigmoid(b_out + sum(hj * wj for hj, wj in zip(h, w_out)))


def forward_arrays(x, w_hidden, b_hidden, w_out, b_out):
    """The same forward pass with no loop over units, built out of yours.

    X is one row because this is a batch of one example. Nothing about the
    shapes below cares how many rows it has, which is the reason real networks
    are written this way: a batch of 64 is the same three lines.
    """
    X = [list(x)]                                  # (1 x n_in)
    H = add_bias(matmul(X, transpose(w_hidden)), b_hidden)     # (1 x n_hidden)
    H = [[sigmoid(v) for v in row] for row in H]
    Y = add_bias(matmul(H, transpose([list(w_out)])), [b_out])  # (1 x 1)
    return sigmoid(Y[0][0])


def random_net(rng, n_in, n_hidden):
    r = lambda: rng.uniform(-2.0, 2.0)
    return (
        [[r() for _ in range(n_in)] for _ in range(n_hidden)],
        [r() for _ in range(n_hidden)],
        [r() for _ in range(n_hidden)],
        r(),
    )


def run_integration(trials=200):
    rng = random.Random(5)
    worst = 0.0
    failed = None
    for _ in range(trials):
        n_in, n_hidden = rng.randint(1, 5), rng.randint(1, 6)
        w_hidden, b_hidden, w_out, b_out = random_net(rng, n_in, n_hidden)
        x = [rng.uniform(-3, 3) for _ in range(n_in)]
        want = forward_loops(x, w_hidden, b_hidden, w_out, b_out)
        try:
            got = forward_arrays(x, w_hidden, b_hidden, w_out, b_out)
        except Exception as exc:
            failed = f"{type(exc).__name__}: {exc}"
            break
        worst = max(worst, abs(got - want))

    console.print()
    if failed:
        console.print(f"  [bold red]{CROSS}[/] the array version raised: "
                      f"[red]{failed}[/red]")
        return False
    ok = claim(f"{trials} random networks, widths 1 to 6, and the two forward "
               f"passes never differed by more than {worst:.3g}",
               worst < 1e-12)
    if ok:
        console.print("\n  [dim]Same arithmetic. The loops moved into "
                      "matmul, and that is all that happened.[/dim]")
    return ok


def run_laws():
    """Four facts about matrices, checked with your functions rather than told."""
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]        # 2 x 3
    B = [[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]]     # 3 x 2
    C = [[3.0, 1.0], [0.0, 2.0]]                  # 2 x 2
    I = [[1.0, 0.0], [0.0, 1.0]]

    console.print()
    good = claim("an identity matrix changes nothing:  I @ C == C",
                 approx_nested(matmul(I, C), C))
    good &= claim("shapes compose:  (2x3) @ (3x2) gives 2x2, and the 3s vanish",
                  [len(matmul(A, B)), len(matmul(A, B)[0])] == [2, 2])
    good &= claim("order matters:  A @ B and B @ A are not the same matrix",
                  not approx_nested(matmul(A, B), matmul(B, A)))
    good &= claim("transpose reverses a product:  (A @ B)^T == B^T @ A^T",
                  approx_nested(transpose(matmul(A, B)),
                                matmul(transpose(B), transpose(A))))
    return good


def plot():
    import plots

    if not everything_written():
        console.print("\n[yellow]The matmul figure needs all five functions "
                      "written first.[/yellow]")
        return
    plots.matmul_figure(matmul, transpose)
    plots.done("05")


ALL = [
    (dot, ((1.0, 2.0), (3.0, 4.0))),
    (matvec, ([[1.0, 2.0]], (1.0, 1.0))),
    (matmul, ([[1.0, 2.0]], [[1.0], [1.0]])),
    (transpose, ([[1.0, 2.0]],)),
    (add_bias, ([[1.0, 2.0]], (0.0, 0.0))),
]


def everything_written():
    return all(is_written(fn, probe) for fn, probe in ALL)


if __name__ == "__main__":
    section("vectors and matrices, one function at a time")
    results = []

    results.append(compare(
        "dot(u, v)", dot_broken, dot, ["u", "v"],
        [
            (((1, 2, 3), (4, 5, 6)), 32.0),
            (((1, 0, 0), (7, 8, 9)), 7.0),     # picks out one component
            (((1, 1), (2, -2)), 0.0),          # can vanish with nothing zero
            (((0, 0), (5, 5)), 0.0),
            (((2, 3), (4, 5)), 23.0),
            (((1.5,), (4,)), 6.0),             # length 1 is still a vector
        ],
        "sum of the products, pair by pair -- one number comes out"))

    results.append(compare(
        "matvec(M, v)", matvec_broken, matvec, ["M", "v"],
        [
            (([[1, 2], [3, 4]], (1, 0)), [1.0, 3.0]),    # picks the first COLUMN
            (([[1, 2], [3, 4]], (0, 1)), [2.0, 4.0]),
            (([[1, 2], [3, 4]], (1, 1)), [3.0, 7.0]),
            (([[1, 0], [0, 1]], (5, -2)), [5.0, -2.0]),  # identity
            (([[1, 2, 3], [4, 5, 6]], (1, 1, 1)), [6.0, 15.0]),  # 2x3 times 3
        ],
        "entry i is the dot of ROW i with v -- so a 2x3 matrix gives 2 numbers"))

    results.append(compare(
        "matmul(A, B)", matmul_broken, matmul, ["A", "B"],
        [
            (([[1, 2], [3, 4]], [[1, 0], [0, 1]]), [[1.0, 2.0], [3.0, 4.0]]),
            (([[1, 2], [3, 4]], [[0, 1], [1, 0]]), [[2.0, 1.0], [4.0, 3.0]]),
            (([[1, 2], [3, 4]], [[5, 6], [7, 8]]), [[19.0, 22.0], [43.0, 50.0]]),
            (([[5, 6], [7, 8]], [[1, 2], [3, 4]]), [[23.0, 34.0], [31.0, 46.0]]),
            (([[1, 2, 3]], [[1], [1], [1]]), [[6.0]]),        # 1x3 @ 3x1 -> 1x1
            (([[1], [2]], [[3, 4]]), [[3.0, 4.0], [6.0, 8.0]]),  # 2x1 @ 1x2 -> 2x2
        ],
        "rows of A against columns of B. the last two rows change shape -- "
        "watch what the inner dimension does"))

    results.append(compare(
        "transpose(M)", transpose_broken, transpose, ["M"],
        [
            (([[1, 2], [3, 4]],), [[1.0, 3.0], [2.0, 4.0]]),
            (([[1, 2, 3], [4, 5, 6]],), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]),
            (([[1], [2], [3]],), [[1.0, 2.0, 3.0]]),
            (([[7]],), [[7.0]]),
        ],
        "rows become columns, so an n x m becomes an m x n"))

    results.append(compare(
        "add_bias(M, b)", add_bias_broken, add_bias, ["M", "b"],
        [
            (([[1, 2], [3, 4]], (10, 20)), [[11.0, 22.0], [13.0, 24.0]]),
            (([[1, 2], [3, 4]], (0, 0)), [[1.0, 2.0], [3.0, 4.0]]),
            (([[1, 2, 3], [4, 5, 6]], (1, 0, -1)), [[2.0, 2.0, 2.0],
                                                    [5.0, 5.0, 5.0]]),
            (([[1, 2]], (5, 5)), [[6.0, 7.0]]),
            (([[1, 2], [3, 4], [5, 6]], (0, 9)), [[1.0, 11.0], [3.0, 13.0],
                                                 [5.0, 15.0]]),
        ],
        "one bias per COLUMN, added to every row -- the first row and the "
        "third are the ones that tell the two readings apart"))

    summary(results)

    if everything_written():
        section("four facts about matrices, checked rather than asserted")
        run_laws()

        section("03's forward pass, written twice")
        run_integration()
    else:
        not_written("the forward pass comparison")

    if "plot" in sys.argv:
        plot()
