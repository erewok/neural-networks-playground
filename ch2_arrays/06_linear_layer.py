"""
Stage 6: the linear layer, forwards and backwards.

05 gave you matmul. This is the first thing built out of it, and the first
place a gradient stops being one number and becomes a grid of them.

SHAPES

    X   (n x d_in)      n examples, one per row, d_in features each
    W   (d_in x d_out)  one column per output unit
    b   (d_out)         one number per output unit
    Y   (n x d_out)     what the layer produced

    dY  (n x d_out)     for every number in Y, how fast the loss rises when
                        that number rises. Handed in from downstream.

W is stored the other way round from 03's w_hidden, which had one ROW per
unit. Both conventions are in use. This one makes the forward pass X @ W with
no transpose in it, and puts the transposes in the backward pass instead.

THE FOUR FUNCTIONS

    linear_forward(X, W, b)  ->  (n x d_out)
    grad_W(X, dY)            ->  (d_in x d_out)
    grad_b(dY)               ->  (d_out)
    grad_x(dY, W)            ->  (n x d_in)

The three gradient functions ask the same question about different numbers:
raise this one number by a little, and how fast does the loss go up? There is
one answer per number, and the answers are laid out the way the numbers are.
A gradient has the same shape as the thing it is about.

MEASURE THEM BEFORE YOU WRITE THEM

Write linear_forward first. Once it is right, `just run 6` measures all three
gradients for you and prints them as grids. The measurement takes one entry of
W, adds 0.00001 to it, runs the layer again, and divides the change in loss by
0.00001. That is a slope. It is what grad_W has to return for that entry, and
it is a subtraction of two losses with no calculus in it.

Read those grids before writing anything. The functions you are about to write
produce, in closed form, numbers you can already watch being made.

WHAT grad_x IS FOR

A single layer has no use for it. Nothing upstream of the input is asking.
It exists so layers can be stacked: layer 2's grad_x is layer 1's dY. That
handoff is backpropagation, and 03 did it by hand, one hidden unit at a time.
The integration test at the bottom stacks two of these layers and trains them,
and grad_x is the only thing joining them.

PROPERTIES

Each is checked against the measurement rather than asserted.

  1. Each gradient has the same shape as the thing it is about.

  2. A feature that is zero in every example gets an all-zero gradient row in
     W. It never reached the output, so it cannot have contributed to the
     error.

  3. Two identical examples give exactly twice the W gradient of one. A batch
     adds up. It does not average.

  4. Examples do not interact. Changing example 0's inputs leaves example 1's
     input gradient alone. They meet only in the weights.

  5. grad_b reads nothing but dY. b has no input of its own, same as in 02
     and 04.

  6. grad_x reads W and does not change it. The same weight is used in both
     directions -- forwards to make an output, backwards to assign blame for
     it. That is what transpose() in 05 was for.

Each function is a small amount of arithmetic over the numbers in its own
signature. Which index pairs with which is the work. Loops are fine. To write
them out of 05's functions instead, paste your dot/matmul/transpose in below:
this file cannot import them, because a module whose name starts with a digit
is not importable.

`just plot 6` draws the measured gradient beside yours, and the loss curve of
the two stacked layers.
"""

import random
import sys

from harness import (CROSS, TICK, Cell, ExerciseTable, approx_nested, console,
                     fmt_grid, not_written, section, summary)

# --------------------------------------------------------------- the originals
# Left here on purpose. Don't edit these -- the harness runs them beside yours.


def linear_forward_broken(X, W, b):
    Y = [[sum(X[n][k] * W[k][j] for k in range(len(W)))
          for j in range(len(W[0]))] for n in range(len(X))]
    return [[v + b[n] for v in row] for n, row in enumerate(Y)]


def grad_W_broken(X, dY):
    return [[sum(dY[n][i] * X[n][j] for n in range(len(X)))
             for j in range(len(X[0]))] for i in range(len(dY[0]))]


def grad_b_broken(dY):
    return list(dY[0])


def grad_x_broken(dY, W):
    return [[sum(dY[n][k] * W[k][j] for k in range(len(W)))
             for j in range(len(W[0]))] for n in range(len(dY))]


# ---------------------------------------------------------------------- yours
# Four functions. The tables score whichever ones exist, so go one at a time,
# and do linear_forward first -- the measurement below runs on it.
# Return lists, not tuples. The tables compare shapes as well as numbers.


def linear_forward(X, W, b):
    """A batch of examples in, a batch of outputs out.

    X is (n x d_in), W is (d_in x d_out), b has d_out entries. Out: (n x d_out).
    The same b is added to every row.
    """
    raise NotImplementedError


def grad_W(X, dY):
    """One slope per entry of W. Out: (d_in x d_out), the same shape as W."""
    raise NotImplementedError


def grad_b(dY):
    """One slope per entry of b. Out: a vector of d_out entries."""
    raise NotImplementedError


def grad_x(dY, W):
    """One slope per entry of X. Out: (n x d_in), the same shape as X."""
    raise NotImplementedError


# ------------------------------------------------------------ the measurement
# Working code. Nothing here needs fixing. This is the definition of a
# gradient, run as arithmetic: move one number, see what the loss does.


def loss(Y, T):
    """Sum of squared error over every entry. One number out."""
    return sum((y - t) ** 2 for row_y, row_t in zip(Y, T)
               for y, t in zip(row_y, row_t))


def dloss(Y, T):
    """dLoss/dY, entry by entry -- the dY the layer is handed from downstream.

    09 derives this one. It is 04's grad, applied to every entry at once.
    """
    return [[2.0 * (y - t) for y, t in zip(row_y, row_t)]
            for row_y, row_t in zip(Y, T)]


def _bumped(M, i, j, h):
    out = [list(row) for row in M]
    out[i][j] += h
    return out


def _loss_at(X, W, b, T):
    return loss(linear_forward(X, W, b), T)


def measured_grad_W(X, W, b, T, h=1e-5):
    """Central difference on every entry of W, one at a time."""
    return [[(_loss_at(X, _bumped(W, i, j, h), b, T) -
              _loss_at(X, _bumped(W, i, j, -h), b, T)) / (2 * h)
             for j in range(len(W[0]))] for i in range(len(W))]


def measured_grad_b(X, W, b, T, h=1e-5):
    return [(_loss_at(X, W, _bumped([list(b)], 0, j, h)[0], T) -
             _loss_at(X, W, _bumped([list(b)], 0, j, -h)[0], T)) / (2 * h)
            for j in range(len(b))]


def measured_grad_X(X, W, b, T, h=1e-5):
    return [[(_loss_at(_bumped(X, n, j, h), W, b, T) -
              _loss_at(_bumped(X, n, j, -h), W, b, T)) / (2 * h)
             for j in range(len(X[0]))] for n in range(len(X))]


# ---------------------------------------------------------------- test harness
# Drawing only -- see harness.py. Nothing here needs fixing.


def is_written(fn, probe_args):
    try:
        fn(*probe_args)
    except NotImplementedError:
        return False
    except Exception:
        return True
    return True


def attempt(fn, args):
    """Run one case. A shape mistake raises, and a raise is a failing row."""
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


def grid(M, places=4):
    return "[" + " | ".join(" ".join(f"{v:+.{places}f}" for v in row)
                            for row in M) + "]"


def vec(v, places=4):
    return "[" + " ".join(f"{x:+.{places}f}" for x in v) + "]"


def shape(M):
    return (len(M), len(M[0]))


# ------------------------------------------------- the layer being measured

X0 = [[1.0, 2.0, -1.0],
      [0.5, -1.0, 2.0]]                 # 2 examples, 3 features
W0 = [[0.5, -0.2],
      [0.1, 0.4],
      [-0.3, 0.7]]                      # 3 features, 2 output units
B0 = [0.05, -0.1]
T0 = [[1.0, 0.0],
      [0.0, 1.0]]


def run_measurement():
    """What the three gradients are, before any of them is written."""
    console.print()
    console.print("  [cyan]X[/cyan]  " + grid(X0, 2) + "   [dim]2 x 3[/dim]")
    console.print("  [cyan]W[/cyan]  " + grid(W0, 2) + "   [dim]3 x 2[/dim]")
    console.print("  [cyan]b[/cyan]  " + vec(B0, 2) + "        [dim]2[/dim]")
    console.print("  [cyan]T[/cyan]  " + grid(T0, 2) +
                  "   [dim]what the outputs should have been[/dim]")

    Y = linear_forward(X0, W0, B0)
    L = loss(Y, T0)
    console.print("  [cyan]Y[/cyan]  " + grid(Y, 4) + "   [dim]your forward pass[/dim]")
    console.print(f"  [cyan]loss[/cyan]  {L:.6f}   [dim]sum of squared error[/dim]")

    # One cell, spelled out. h is large here so the digits are readable.
    h = 1e-4
    i, j = 0, 1
    up = _loss_at(X0, _bumped(W0, i, j, h), B0, T0)
    console.print()
    console.print(f"  [dim]W[{i}][{j}] is {W0[i][j]:+.2f} and the loss is "
                  f"{L:.6f}.[/dim]")
    console.print(f"  [dim]Set it to {W0[i][j] + h:+.5f} and the loss is "
                  f"{up:.6f}.[/dim]")
    console.print(f"  [dim]({up:.6f} - {L:.6f}) / {h} = "
                  f"{(up - L) / h:+.4f}.  That is one entry of grad_W.[/dim]")

    gW = measured_grad_W(X0, W0, B0, T0)
    gb = measured_grad_b(X0, W0, B0, T0)
    gX = measured_grad_X(X0, W0, B0, T0)

    console.print()
    console.print("  [bold]measured grad_W[/bold]  " + grid(gW) +
                  "   [dim]3 x 2[/dim]")
    console.print("  [bold]measured grad_b[/bold]  " + vec(gb) +
                  "        [dim]2[/dim]")
    console.print("  [bold]measured grad_X[/bold]  " + grid(gX) +
                  "   [dim]2 x 3[/dim]")
    console.print()
    console.print("  [dim]dY, which the three functions are given, is "
                  "dloss(Y, T):[/dim]")
    console.print("  [cyan]dY[/cyan] " + grid(dloss(Y, T0)) +
                  "   [dim]2 x 2[/dim]")

    console.print()
    good = claim("each measured gradient has the shape of the thing it is about",
                 shape(gW) == shape(W0) and len(gb) == len(B0) and
                 shape(gX) == shape(X0))

    # A feature nobody has: column 2 of X zeroed out.
    Xz = [[row[0], row[1], 0.0] for row in X0]
    gWz = measured_grad_W(Xz, W0, B0, T0)
    good &= claim("zero out feature 2 in every example and row 2 of grad_W "
                  "goes to zero: " + vec(gWz[2]),
                  all(abs(v) < 1e-6 for v in gWz[2]))

    # One example, then the same example twice.
    one = measured_grad_W([X0[0]], W0, B0, [T0[0]])
    two = measured_grad_W([X0[0], X0[0]], W0, B0, [T0[0], T0[0]])
    good &= claim("the same example twice gives exactly twice the W gradient "
                  "of it once",
                  approx_nested(two, [[2 * v for v in r] for r in one], 1e-5))

    # Examples meet only in the weights.
    Xp = [[9.0, -4.0, 3.0], X0[1]]
    gXp = measured_grad_X(Xp, W0, B0, T0)
    good &= claim("change example 0's inputs and example 1's input gradient "
                  "does not move",
                  approx_nested(gXp[1], gX[1], 1e-5))
    return good


def run_against_measurement(trials=200):
    """Your three closed forms against the difference quotient, on random layers."""
    rng = random.Random(6)
    worst = 0.0
    failed = None
    for _ in range(trials):
        n = rng.randint(1, 4)
        d_in = rng.randint(1, 5)
        d_out = rng.randint(1, 4)
        r = lambda: rng.uniform(-2.0, 2.0)
        X = [[r() for _ in range(d_in)] for _ in range(n)]
        W = [[r() for _ in range(d_out)] for _ in range(d_in)]
        b = [r() for _ in range(d_out)]
        T = [[r() for _ in range(d_out)] for _ in range(n)]
        try:
            dY = dloss(linear_forward(X, W, b), T)
            pairs = [
                (grad_W(X, dY), measured_grad_W(X, W, b, T)),
                ([grad_b(dY)], [measured_grad_b(X, W, b, T)]),
                (grad_x(dY, W), measured_grad_X(X, W, b, T)),
            ]
        except Exception as exc:
            failed = f"{type(exc).__name__}: {exc}"
            break
        for got, want in pairs:
            if shape(got) != shape(want):
                failed = (f"shape {shape(got)} where {shape(want)} was wanted, "
                          f"on a {n}x{d_in} batch through a {d_in}x{d_out} layer")
                break
            worst = max(worst, max(abs(g - w) for rg, rw in zip(got, want)
                                   for g, w in zip(rg, rw)))
        if failed:
            break

    console.print()
    if failed:
        console.print(f"  [bold red]{CROSS}[/] {failed}")
        return False
    ok = claim(f"{trials} random layers, batches of 1 to 4 and widths to 5, "
               f"and no gradient differed from its measurement by more than "
               f"{worst:.3g}", worst < 1e-5)
    if ok:
        console.print("\n  [dim]The closed forms and the difference quotient "
                      "are computing the same slopes.[/dim]")
    return ok


# -------------------------------------------------- two layers, and the handoff


def two_layer_forward(X, W1, b1, W2, b2):
    H = linear_forward(X, W1, b1)
    Y = linear_forward(H, W2, b2)
    return H, Y


def train_two_layers(steps=600, rate=0.05):
    """Layer 2's grad_x is layer 1's dY. That line is the whole point."""
    rng = random.Random(60)
    r = lambda: rng.uniform(-0.5, 0.5)
    X = [[rng.uniform(-1, 1) for _ in range(3)] for _ in range(12)]
    true_W = [[1.5, -0.5], [0.0, 2.0], [-1.0, 0.5]]
    true_b = [0.3, -0.7]
    T = linear_forward(X, true_W, true_b)

    W1 = [[r() for _ in range(4)] for _ in range(3)]
    b1 = [0.0] * 4
    W2 = [[r() for _ in range(2)] for _ in range(4)]
    b2 = [0.0] * 2

    history = []
    for _ in range(steps):
        H, Y = two_layer_forward(X, W1, b1, W2, b2)
        history.append(loss(Y, T) / len(X))

        dY = dloss(Y, T)
        gW2, gb2 = grad_W(H, dY), grad_b(dY)
        dH = grad_x(dY, W2)                    # the handoff
        gW1, gb1 = grad_W(X, dH), grad_b(dH)

        step = rate / len(X)
        W2 = [[w - step * g for w, g in zip(rw, rg)] for rw, rg in zip(W2, gW2)]
        b2 = [w - step * g for w, g in zip(b2, gb2)]
        W1 = [[w - step * g for w, g in zip(rw, rg)] for rw, rg in zip(W1, gW1)]
        b1 = [w - step * g for w, g in zip(b1, gb1)]

    _, Y = two_layer_forward(X, W1, b1, W2, b2)
    history.append(loss(Y, T) / len(X))
    return history, (X, T, W1, b1, W2, b2)


def run_training():
    try:
        history, state = train_two_layers()
    except Exception as exc:
        console.print(f"\n  [bold red]{CROSS}[/] training raised: "
                      f"[red]{type(exc).__name__}: {exc}[/red]")
        return False, None

    first, last = history[0], history[-1]
    console.print()
    console.print(f"  [dim]mean squared error, step 0:[/dim] {first:.6f}"
                  f"   [dim]step {len(history) - 1}:[/dim] {last:.3e}")
    good = claim("two stacked layers, joined only by grad_x, drove the error "
                 "below a millionth of where it started",
                 last < first / 1e6)
    good &= claim("the error fell on every single step",
                  all(b <= a + 1e-12 for a, b in zip(history, history[1:])))

    # Two linear layers in a row are one linear layer. linear_forward(W1, W2, 0)
    # is W1 @ W2, and linear_forward([b1], W2, b2) is the composed bias.
    X, T, W1, b1, W2, b2 = state
    Wc = linear_forward(W1, W2, [0.0] * len(W2[0]))
    bc = linear_forward([b1], W2, b2)[0]
    _, Y = two_layer_forward(X, W1, b1, W2, b2)
    good &= claim("those two layers are one layer: a single "
                  f"{len(Wc)}x{len(Wc[0])} W and one b reproduce every output",
                  approx_nested(linear_forward(X, Wc, bc), Y, 1e-9))
    console.print("\n  [dim]Stacking linear layers buys nothing. 08 puts "
                  "something between them.[/dim]")
    return good, history


def plot():
    import plots

    if not everything_written():
        console.print("\n[yellow]The gradient figure needs all four functions "
                      "written first.[/yellow]")
        return
    dY = dloss(linear_forward(X0, W0, B0), T0)
    plots.linear_grad_figure(W0, measured_grad_W(X0, W0, B0, T0), grad_W(X0, dY))
    history, _ = train_two_layers()
    plots.two_layer_loss_figure(history)
    plots.done("06")


ALL = [
    (linear_forward, ([[1.0]], [[1.0]], (0.0,))),
    (grad_W, ([[1.0]], [[1.0]])),
    (grad_b, ([[1.0]],)),
    (grad_x, ([[1.0]], [[1.0]])),
]


def everything_written():
    return all(is_written(fn, probe) for fn, probe in ALL)


if __name__ == "__main__":
    section("a layer, forwards")
    results = []

    fwd = compare(
        "linear_forward(X, W, b)", linear_forward_broken, linear_forward,
        ["X", "W", "b"],
        [
            (([[1, 2]], [[1, 0], [0, 1]], (0, 0)), [[1.0, 2.0]]),
            (([[1, 2]], [[1, 0], [0, 1]], (10, 20)), [[11.0, 22.0]]),
            (([[0, 0]], [[1, 2], [3, 4]], (7, 8)), [[7.0, 8.0]]),
            (([[1, 0], [0, 1]], [[1, 2], [3, 4]], (0, 0)), [[1.0, 2.0],
                                                            [3.0, 4.0]]),
            (([[1, 2, 3]], [[1], [1], [1]], (0,)), [[6.0]]),
            (([[1, 2]], [[1, 2, 3], [4, 5, 6]], (0, 0, 0)),
             [[9.0, 12.0, 15.0]]),
            (([[1, 1], [2, 2]], [[1], [1]], (5,)), [[7.0], [9.0]]),
        ],
        "a batch of rows through one layer. row 3 has no input at all. rows 2 "
        "and 7 tell a per-column bias from a per-row one, and row 6 changes "
        "width")

    results.append(fwd)
    forward_ok = fwd[1] == fwd[2]

    section("the three gradients, measured")
    if forward_ok:
        console.print("[dim]No calculus below. Every number is a loss minus "
                      "a loss, divided by how far one knob moved.[/dim]")
        run_measurement()
    else:
        not_written("the measurement")
        console.print("[dim]It runs your linear_forward, so that one has to "
                      "be right first.[/dim]")

    section("the three gradients, written")
    results.append(compare(
        "grad_W(X, dY)", grad_W_broken, grad_W, ["X", "dY"],
        [
            (([[1, 0]], [[1]]), [[1.0], [0.0]]),
            (([[2, 3]], [[1]]), [[2.0], [3.0]]),
            (([[2, 3]], [[10]]), [[20.0], [30.0]]),
            (([[1, 1]], [[1, 2]]), [[1.0, 2.0], [1.0, 2.0]]),
            (([[1, 2], [3, 4]], [[1, 0], [0, 1]]), [[1.0, 3.0], [2.0, 4.0]]),
            (([[1, 0], [1, 0]], [[1], [1]]), [[2.0], [0.0]]),
            (([[1, 2, 3]], [[1, 10]]), [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]),
        ],
        "out is (d_in x d_out), the shape of W -- not the shape of either "
        "argument. row 1 has a feature that is absent, row 6 is the same "
        "example twice"))

    results.append(compare(
        "grad_b(dY)", grad_b_broken, grad_b, ["dY"],
        [
            (([[1, 2]],), [1.0, 2.0]),
            (([[1, 2], [3, 4]],), [4.0, 6.0]),
            (([[1, 2], [-1, -2]],), [0.0, 0.0]),
            (([[5]],), [5.0]),
            (([[1, 1], [1, 1], [1, 1]],), [3.0, 3.0]),
            (([[1, 2, 3]],), [1.0, 2.0, 3.0]),
        ],
        "d_out numbers out, whatever the batch size. every row but the first "
        "has more than one example in it"))

    results.append(compare(
        "grad_x(dY, W)", grad_x_broken, grad_x, ["dY", "W"],
        [
            (([[1]], [[1], [0]]), [[1.0, 0.0]]),
            (([[1]], [[2], [3]]), [[2.0, 3.0]]),
            (([[1, 1]], [[1, 0], [0, 1]]), [[1.0, 1.0]]),
            (([[1, 0]], [[1, 2], [3, 4]]), [[1.0, 3.0]]),
            (([[0, 1]], [[1, 2], [3, 4]]), [[2.0, 4.0]]),
            (([[1, 0], [0, 1]], [[1, 2], [3, 4]]), [[1.0, 3.0], [2.0, 4.0]]),
            (([[1, 2]], [[1, 0], [0, 1], [1, 1]]), [[1.0, 2.0, 3.0]]),
        ],
        "out is (n x d_in), the shape of X -- one number per input, not per "
        "weight. the last row sends a 2-wide gradient back through a 3-wide "
        "input"))

    summary(results)

    if everything_written():
        section("your gradients against the measurement")
        run_against_measurement()

        section("two layers, joined by grad_x")
        run_training()
    else:
        not_written("the measurement check and the two-layer training")

    if "plot" in sys.argv:
        plot()
