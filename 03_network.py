"""
Stage 3: two layers and a sigmoid -- the thing that can finally solve XOR.

One neuron draws one straight line. XOR needs more than one line, so it needs
more than one layer. Stacking is easy; the hard part is that a hidden unit never
sees the target and so has no idea what it should have produced. Somebody has to
tell it how much of the mistake was its fault. That is the whole subject here.

WHY SIGMOID REPLACED THE STEP FUNCTION
  A step function is flat everywhere, so it can only say "right" or "wrong",
  never "close" or "which way is better". Sigmoid is a smooth S-curve, so at
  every point it has a slope, and a slope is a direction to move. Its slope,
  written in terms of the value it just produced, is sigmoid_slope() below.
  Notice that it goes to nearly zero when the output is near 0 or near 1: a unit
  that is already confident barely moves, however wrong it happens to be. Some
  of the table rows below are there to show you that.

THE THREE FUNCTIONS TO FIX
  Every unit gets a "delta": how wrong it is and which way it should go. Once a
  unit has a delta, updating its incoming weights is the same operation for
  every unit in the network.

    output_delta()    the output unit can see the target, so it works out its
                      own delta directly.
    hidden_delta()    a hidden unit cannot see the target. All it has is the
                      delta of the unit it fed, and the weight it sent its
                      signal along -- that weight is the only record of how much
                      it contributed.
    updated_weight()  move one weight, given the delta of the unit it feeds
                      into and the value that arrived on it.

SIGN CONVENTION -- worth reading, it differs from 02
  In 02 the error was `target - prediction` and updates were added. Here a delta
  points in the direction that makes things WORSE, and we move against it, so
  updates subtract. This is the usual convention in the literature and it is
  what the tables below expect. It is not a trick; it just flips which way the
  arithmetic leans.

WHY THREE HIDDEN UNITS AND NOT TWO
  Two is the theoretical minimum for XOR, and hand_wire() below uses exactly
  two to prove it. But gradient descent starting from random weights only finds
  the answer with two units about half the time -- the rest of the time it slides
  into a corner it cannot climb out of and sits there at three-out-of-four
  forever. Three units finds it every time. Worth remembering: "the network can
  represent the answer" and "training will find the answer" are separate claims,
  and only the first one is guaranteed.

Properties these should have:

  1. A unit that is already saturated (output near 0 or 1) barely moves, even
     when it is badly wrong.
  2. A hidden unit connected by a zero weight gets zero blame -- it did not
     contribute. A negative connection gets blame of the opposite sign. Twice
     the connection, twice the blame.
  3. A weight only moves in proportion to what actually arrived on it. Nothing
     arrived, nothing moves.
  4. No mistake anywhere means nothing moves anywhere.
"""

import math
import random
import sys

from harness import (TICK, Cell, ExerciseTable, approx, console, fmt_float,
                     not_written, prediction_row, section, summary)


def sigmoid(z):
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_slope(a):
    """Slope of the sigmoid, written in terms of the value it already produced."""
    return a * (1.0 - a)


# --------------------------------------------------------------- the originals
# Left here on purpose. Don't edit these -- the harness runs them beside yours.


def output_delta_broken(y, target):
    return y - target


def hidden_delta_broken(h, w_out, out_delta):
    return 0.0


def updated_weight_broken(w, delta, incoming, rate):
    return w - rate * delta


# ---------------------------------------------------------------------- yours
# Write these three. Each is a single expression, and sigmoid_slope() is already
# written for you. The tables score whichever ones exist, so go one at a time.


def output_delta(y, target):
    """Delta for the output unit, which can see the target."""
    raise NotImplementedError


def hidden_delta(h, w_out, out_delta):
    """Delta for a hidden unit, which cannot.

    h        what this hidden unit produced
    w_out    the weight it sent that value along
    out_delta  the delta of the unit on the other end of that weight
    """
    raise NotImplementedError


def updated_weight(w, delta, incoming, rate):
    """Move one weight. `delta` belongs to the unit this weight feeds into,
    `incoming` is the value that travelled along it."""
    raise NotImplementedError


# ---------------------------------------------------------------- test harness
# Drawing only -- see harness.py. Nothing here needs fixing.


def is_written(fn, probe_args):
    try:
        fn(*probe_args)
    except NotImplementedError:
        return False
    return True


def compare(title, original, mine, argnames, cases, note=""):
    """cases: list of (args_tuple, expected, why)."""
    written = is_written(mine, cases[0][0])
    t = ExerciseTable(
        title=title,
        note=note,
        input_header="arguments",
        why_header="what this row is showing",
        yours_written=written,
        fmt=fmt_float(6),
        compare=approx,
    )
    for args, want, why in cases:
        label = ", ".join(f"{n}={v:g}" for n, v in zip(argnames, args))
        t.add(label, want, Cell(original(*args)),
              Cell(mine(*args)) if written else None, why=why)
    return t.render()


# ------------------------------------------------------- the integration test


class Network:
    def __init__(self, out_fn, hid_fn, w_fn, n_in=2, n_hidden=3,
                 rate=1.0, seed=0, init_scale=1.0):
        rng = random.Random(seed)
        r = lambda: rng.uniform(-init_scale, init_scale)
        self.w_hidden = [[r() for _ in range(n_in)] for _ in range(n_hidden)]
        self.b_hidden = [r() for _ in range(n_hidden)]
        self.w_out = [r() for _ in range(n_hidden)]
        self.b_out = r()
        self.rate = rate
        self.out_fn, self.hid_fn, self.w_fn = out_fn, hid_fn, w_fn

    def forward(self, x):
        h = []
        for w_row, b in zip(self.w_hidden, self.b_hidden):
            h.append(sigmoid(b + sum(xi * wi for xi, wi in zip(x, w_row))))
        return h, sigmoid(self.b_out + sum(hj * wj for hj, wj in zip(h, self.w_out)))

    def predict(self, x):
        return self.forward(x)[1]

    def learn(self, x, target):
        h, y = self.forward(x)
        od = self.out_fn(y, target)

        # The hidden deltas must be worked out BEFORE w_out is touched -- they
        # are computed from the weights the signal actually travelled along.
        hd = [self.hid_fn(h[j], self.w_out[j], od) for j in range(len(h))]

        # A bias is just a weight on an input that is always 1, so the same
        # function updates both.
        for j in range(len(h)):
            self.w_out[j] = self.w_fn(self.w_out[j], od, h[j], self.rate)
        self.b_out = self.w_fn(self.b_out, od, 1.0, self.rate)

        for j in range(len(h)):
            for i in range(len(x)):
                self.w_hidden[j][i] = self.w_fn(self.w_hidden[j][i], hd[j],
                                                x[i], self.rate)
            self.b_hidden[j] = self.w_fn(self.b_hidden[j], hd[j], 1.0, self.rate)
        return (y - target) ** 2

    def train(self, examples, epochs=4000):
        loss = 0.0
        for _ in range(epochs):
            loss = sum(self.learn(x, t) for x, t in examples) / len(examples)
        return loss

    def solves(self, examples):
        return all(round(self.predict(x)) == t for x, t in examples)

    def hand_wire(self):
        """A known-good XOR solution set by hand, for 2 hidden units.

        XOR is "at least one, but not both": one hidden unit detects 'at least
        one', the other detects 'both', and the output takes the first while the
        second vetoes it. Large weights make each unit saturate, which is how a
        smooth sigmoid imitates a hard yes/no. Proof that the shape of this
        network can hold the answer -- learning it is the separate problem.
        """
        self.w_hidden = [[20.0, 20.0], [20.0, 20.0]]
        self.b_hidden = [-10.0, -30.0]
        self.w_out = [20.0, -20.0]
        self.b_out = -10.0
        return self


XOR = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]


def run_xor(label, out_fn, hid_fn, w_fn):
    console.print(f"\n[bold]{label}[/bold]")
    net = Network(out_fn, hid_fn, w_fn)
    before = [f"{v:+.3f}" for row in net.w_hidden for v in row]
    loss = net.train(XOR)
    after = [f"{v:+.3f}" for row in net.w_hidden for v in row]

    moved = before != after
    console.print(f"  final loss [bold]{loss:.4f}[/bold]"
                  f"   [dim](0.25 is what guessing scores)[/dim]")
    console.print(f"  [dim]hidden weights before[/dim]  [cyan]{', '.join(before)}[/cyan]")
    console.print(f"  [dim]hidden weights after [/dim]  "
                  f"[{'cyan' if moved else 'yellow'}]{', '.join(after)}[/]")
    if not moved:
        console.print("  [yellow]^ identical to before -- the hidden layer "
                      "never learned anything[/yellow]")
    for x, t in XOR:
        y = net.predict(x)
        prediction_row(x, t, round(y), raw=y)
    solved = net.solves(XOR)
    console.print(f"  [bold green]{TICK} XOR SOLVED[/bold green]" if solved
                  else "  [yellow]not solved[/yellow]")
    return solved


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "handwired":
        section("XOR by hand -- no training at all")
        net = Network(output_delta_broken, hidden_delta_broken,
                      updated_weight_broken, n_hidden=2).hand_wire()
        for x, t in XOR:
            y = net.predict(x)
            prediction_row(x, t, round(y), raw=y)
        console.print(f"  [bold green]{TICK} XOR SOLVED[/bold green]"
                      if net.solves(XOR) else "  [yellow]not solved[/yellow]")
        console.print("\n[dim italic]The shape of this network can hold the "
                      "answer. Finding it by training is the other problem.[/dim italic]")
        sys.exit(0)

    section("backpropagation, one function at a time")
    results = []

    results.append(compare(
        "output_delta(y, target)", output_delta_broken, output_delta,
        ["y", "target"],
        [
            ((0.5, 1), -0.125000, "unsure and should have fired"),
            ((0.5, 0), +0.125000, "unsure and should not have"),
            ((0.9, 1), -0.009000, "nearly right already, small move"),
            ((0.1, 0), +0.009000, "nearly right the other way"),
            ((0.99, 0), +0.009801, "badly wrong but saturated -- barely moves"),
            ((0.5, 0.5), 0.000000, "no error at all"),
        ],
        "y is what the output unit produced"))

    results.append(compare(
        "hidden_delta(h, w_out, out_delta)", hidden_delta_broken, hidden_delta,
        ["h", "w_out", "out_delta"],
        [
            ((0.5, 1.0, 0.2), +0.050000, "ordinary case"),
            ((0.5, 0.0, 0.2), +0.000000, "not connected -- not its fault"),
            ((0.5, -1.0, 0.2), -0.050000, "negative connection, opposite blame"),
            ((0.5, 2.0, 0.2), +0.100000, "twice connected, twice the blame"),
            ((0.99, 1.0, 0.2), +0.001980, "saturated -- barely moves"),
            ((0.5, 1.0, 0.0), +0.000000, "nothing went wrong downstream"),
        ],
        "this unit produced h and sent it along w_out to a unit whose delta is out_delta"))

    results.append(compare(
        "updated_weight(w, delta, incoming, rate)", updated_weight_broken,
        updated_weight, ["w", "delta", "incoming", "rate"],
        [
            ((0.0, 0.5, 1.0, 0.5), -0.250000, "move against the delta"),
            ((0.0, 0.5, 0.0, 0.5), +0.000000, "nothing arrived, nothing moves"),
            ((0.4, -0.2, 1.0, 0.5), +0.500000, "negative delta pushes the other way"),
            ((0.4, 0.5, 2.0, 0.5), -0.100000, "twice the input, twice the move"),
            ((0.4, 0.0, 1.0, 0.5), +0.400000, "no delta, no move"),
        ],
        "delta belongs to the unit this weight feeds into"))

    summary(results)

    section("those same functions, wired into a network, trained on XOR")
    run_xor("with the originals", output_delta_broken, hidden_delta_broken,
            updated_weight_broken)
    if all(m is not None for _, m, _ in results):
        if run_xor("with yours", output_delta, hidden_delta, updated_weight):
            console.print("\n  [bold green]Two layers, and the thing one neuron "
                          "could never do is done.[/bold green]")
    else:
        not_written("with yours")
