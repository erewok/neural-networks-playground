"""
Stage 4: gradient descent, with no network in the way.

02 and 03 both nudged weights when they got something wrong, but neither ever
said what "nudge" means or how far to go. This file is that question on its own,
using the smallest possible model -- a straight line through some points. No
layers, no sigmoid, no classes. Just: given some knobs and a way to measure how
bad we are, which way do we turn them?

THE PIECES

    predict()        the model. Given x and the two knobs, guess y.
    squared_error()  how bad one guess was.
    grad_w()         if I nudge w up a little, does the error go up or down,
    grad_b()         and how fast? Same question for b.
    gradient_step()  actually move a knob, given that answer.

A "gradient" is nothing more exotic than that: the slope of the error with
respect to one knob. Positive slope means turning the knob up makes things
worse, so you turn it down. That is the whole algorithm, and it is what 03's
deltas were secretly doing all along.

TRAINING vs INFERENCE -- the distinction you asked about lives here

    inference   run predict(). That is it. No targets are consulted, nothing
                is measured, no knob moves. You could delete the other four
                functions and inference would still work.

    training    run predict(), compare the guess against a known target,
                work out which way each knob should turn, and turn it.
                Repeat.

    So inference is a strict subset of training -- the first step of it, with
    the learning cut off. This is why a trained model can be shipped as nothing
    but its numbers: once training is over, the guessing is all that remains.
    `just plot 4` shows the fitted line meeting points it was never trained on.

Properties these should have:

  1. When x is zero the model returns the bias alone. The bias is the value the
     line has when there is no evidence at all -- where it crosses the y-axis.
  2. Error is never negative, and being wrong by 4 is much worse than being
     wrong by 1 -- not four times worse. Squaring is doing that.
  3. A guess that was already perfect produces zero error and zero gradient,
     so nothing moves.
  4. The w gradient scales with x; the b gradient does not. Same asymmetry as
     the bias in 02, for the same reason: b has no input of its own.
  5. Steps go DOWNHILL. A gradient points the way that makes things worse, so
     moving against it is what makes things better.
"""

import random
import sys

from harness import (CROSS, TICK, Cell, ExerciseTable, approx, console,
                     fmt_float, not_written, section, summary)

# --------------------------------------------------------------- the originals
# Left here on purpose. Don't edit these -- the harness runs them beside yours.


def predict_broken(x, w, b):
    return w * x


def squared_error_broken(pred, target):
    return pred - target


def grad_w_broken(x, pred, target):
    return 2 * (pred - target)


def grad_b_broken(pred, target):
    return 0.0


def gradient_step_broken(param, grad, rate):
    return param + rate * grad


# ---------------------------------------------------------------------- yours
# Five one-liners. The tables score whichever exist, so go one at a time.


def predict(x, w, b):
    """The model: given x and the knobs, guess y."""
    raise NotImplementedError


def squared_error(pred, target):
    """How bad was this one guess?"""
    raise NotImplementedError


def grad_w(x, pred, target):
    """Slope of that error with respect to w."""
    raise NotImplementedError


def grad_b(pred, target):
    """Slope of that error with respect to b."""
    raise NotImplementedError


def gradient_step(param, grad, rate):
    """Move one knob, given its slope and how big a step to take."""
    raise NotImplementedError


# ---------------------------------------------------------------- test harness
# Drawing only -- see harness.py. Nothing here needs fixing.


def is_written(fn, probe_args):
    try:
        fn(*probe_args)
    except NotImplementedError:
        return False
    return True


ALL = [
    (predict, (1.0, 1.0, 1.0)),
    (squared_error, (1.0, 1.0)),
    (grad_w, (1.0, 1.0, 1.0)),
    (grad_b, (1.0, 1.0)),
    (gradient_step, (1.0, 1.0, 0.1)),
]


def everything_written():
    return all(is_written(fn, probe) for fn, probe in ALL)


def compare(title, original, mine, argnames, cases, note=""):
    written = is_written(mine, cases[0][0])
    t = ExerciseTable(
        title=title, note=note, input_header="arguments",
        yours_written=written, fmt=fmt_float(2), compare=approx,
    )
    for args, want in cases:
        label = ", ".join(f"{n}={v:g}" for n, v in zip(argnames, args))
        t.add(label, want, Cell(original(*args)),
              Cell(mine(*args)) if written else None)
    return t.render()


# ------------------------------------------------------- the integration test
# Some points that lie roughly on a line, and the loop that goes looking for it.

TRUE_W, TRUE_B = 2.5, 1.3


def make_data(n=40, noise=0.9, seed=7):
    rng = random.Random(seed)
    data = []
    for _ in range(n):
        x = rng.uniform(0.0, 4.0)
        y = TRUE_W * x + TRUE_B + rng.gauss(0.0, noise)
        data.append((x, y))
    return data


def split(data, holdout=10):
    """Points the model may learn from, and points it must never see."""
    return data[:-holdout], data[-holdout:]


def mean_loss(data, w, b):
    return sum(squared_error(predict(x, w, b), y) for x, y in data) / len(data)


def train(data, rate=0.02, epochs=120, w=0.0, b=0.0):
    """Batch gradient descent. Returns the knobs and the whole journey."""
    history = [(w, b, mean_loss(data, w, b))]
    for _ in range(epochs):
        gw = sum(grad_w(x, predict(x, w, b), y) for x, y in data) / len(data)
        gb = sum(grad_b(predict(x, w, b), y) for x, y in data) / len(data)
        w = gradient_step(w, gw, rate)
        b = gradient_step(b, gb, rate)
        history.append((w, b, mean_loss(data, w, b)))
    return w, b, history


# How close the fitted line has to land before we call it a fit. The points
# carry noise, so the answer is not supposed to be exact -- it is supposed to
# be near, and to have got there by going downhill.
W_TOLERANCE = B_TOLERANCE = 0.3


def claim(text, holds):
    console.print(f"  [{'bold green' if holds else 'bold red'}]"
                  f"{TICK if holds else CROSS}[/] {text}")
    return holds


def num(v):
    """A rate set too high sends these to 1e130, which is unreadable as .3f."""
    return f"{v:.3f}" if abs(v) < 1e6 else f"{v:.3g}"


def verdict(history, w, b, holdout_mse):
    """The section above prints numbers. This decides whether they are good.

    Without it the file can be read but never failed -- 02 and 03 both end on a
    pass/fail and this one did not. It catches what the tables cannot: the five
    functions can each be right row by row while the loop built out of them
    still walks the wrong way, which is what a rate set too high looks like.
    """
    start, end = history[0][2], history[-1][2]
    console.print()
    good = claim(f"the loss went down, {num(start)} to {num(end)}", end < start)
    good &= claim("and went down at every single epoch, never uphill",
                  all(history[i + 1][2] <= history[i][2] + 1e-12
                      for i in range(len(history) - 1)))
    good &= claim(f"w landed within {W_TOLERANCE} of the line the points came "
                  f"from ({num(abs(w - TRUE_W))} away)",
                  abs(w - TRUE_W) < W_TOLERANCE)
    good &= claim(f"b did too ({num(abs(b - TRUE_B))} away)",
                  abs(b - TRUE_B) < B_TOLERANCE)
    # Only meaningful once the fit above actually worked: a line that exploded
    # is equally terrible on both sets, and that is not generalization.
    good &= claim(f"and it does about as well on points it never trained on "
                  f"({num(holdout_mse)} against {num(end)})",
                  good and holdout_mse < 3.0 * end)
    return good


def plot():
    import plots

    if not everything_written():
        console.print("\n[yellow]The plots need all five functions written "
                      "first.[/yellow]")
        console.print("[dim]They draw the path your gradients actually "
                      "take, so there is nothing to show until then.[/dim]")
        return

    data = make_data()
    train_pts, holdout = split(data)
    w, b, history = train(train_pts)

    plots.fit_figure(train_pts, history, TRUE_W, TRUE_B)
    plots.descent_figure(train_pts, history, mean_loss)
    plots.inference_figure(train_pts, holdout, w, b, predict)
    plots.done("04")


if __name__ == "__main__":
    section("gradient descent, one function at a time")
    results = []

    results.append(compare(
        "predict(x, w, b)", predict_broken, predict, ["x", "w", "b"],
        [
            ((2.0, 3.0, 1.0), +7.00),
            ((0.0, 3.0, 1.0), +1.00),   # no evidence at all -- just the bias
            ((2.0, 0.0, 1.0), +1.00),   # a flat line sits at the bias
            ((-1.0, 2.0, 0.5), -1.50),
            ((1.0, 2.5, 0.0), +2.50),
        ],
        "w is the slope of the line, b is where it crosses the y-axis"))

    results.append(compare(
        "squared_error(pred, target)", squared_error_broken, squared_error,
        ["pred", "target"],
        [
            ((3.0, 1.0), +4.00),    # too high by 2
            ((1.0, 3.0), +4.00),    # too low by 2 -- equally bad
            ((2.0, 2.0), +0.00),    # spot on
            ((0.5, 0.0), +0.25),    # small misses shrink
            ((4.0, 0.0), +16.00),   # big misses dominate
        ],
        "wrong is wrong in either direction, and big misses count for far more"))

    results.append(compare(
        "grad_w(x, pred, target)", grad_w_broken, grad_w,
        ["x", "pred", "target"],
        [
            ((1.0, 3.0, 1.0), +4.00),    # guessed high, so turn w down
            ((0.0, 3.0, 1.0), +0.00),    # x was 0 -- w did not cause this
            ((2.0, 3.0, 1.0), +8.00),    # twice the x, twice the blame
            ((1.0, 1.0, 3.0), -4.00),    # guessed low, turn w the other way
            ((1.0, 2.0, 2.0), +0.00),    # no error, no slope
            ((-1.0, 3.0, 1.0), -4.00),   # negative x flips the sign
        ],
        "which way does the error move if w goes up a little?"))

    results.append(compare(
        "grad_b(pred, target)", grad_b_broken, grad_b, ["pred", "target"],
        [
            ((3.0, 1.0), +4.00),
            ((1.0, 3.0), -4.00),
            ((2.0, 2.0), +0.00),
            ((2.5, 2.0), +1.00),
        ],
        "same question for b -- note there is no x to scale it"))

    results.append(compare(
        "gradient_step(param, grad, rate)", gradient_step_broken, gradient_step,
        ["param", "grad", "rate"],
        [
            ((1.0, 4.0, 0.1), +0.60),    # slope says up is worse, so go down
            ((1.0, -4.0, 0.1), +1.40),
            ((1.0, 0.0, 0.1), +1.00),    # flat here, stay put
            ((0.0, 10.0, 0.01), -0.10),  # a small rate takes a small step
        ],
        "rate is how much of the slope to act on"))

    summary(results)

    section("those same functions, fitting a line to 30 points")
    if everything_written():
        data = make_data()
        train_pts, holdout = split(data)
        w, b, history = train(train_pts)
        start_loss = history[0][2]
        console.print(f"\n  started at  w=[bold]{history[0][0]:+.3f}[/bold] "
                      f"b=[bold]{history[0][1]:+.3f}[/bold]   "
                      f"loss [bold]{start_loss:.3f}[/bold]")
        console.print(f"  ended at    w=[bold]{w:+.3f}[/bold] "
                      f"b=[bold]{b:+.3f}[/bold]   "
                      f"loss [bold]{history[-1][2]:.3f}[/bold]")
        console.print(f"  [dim]the line the points actually came from: "
                      f"w={TRUE_W} b={TRUE_B} (plus noise)[/dim]")
        console.print(f"\n  [bold]inference[/bold] on {len(holdout)} points it "
                      f"never trained on:")
        err = sum(squared_error(predict(x, w, b), y) for x, y in holdout)
        console.print(f"    mean squared error [bold]{err / len(holdout):.3f}[/bold]"
                      f"   [dim](training loss was {history[-1][2]:.3f})[/dim]")
        console.print("    [dim]no targets were consulted to make those "
                      "guesses -- only predict()[/dim]")

        if verdict(history, w, b, err / len(holdout)):
            console.print("\n  [bold green]the line was found by going "
                          "downhill, and it holds up off the training set."
                          "[/bold green]")
        console.print("\n  [dim]run `just plot 4` to see the line move[/dim]")
    else:
        not_written("fitting a line")

    if "plot" in sys.argv:
        plot()
