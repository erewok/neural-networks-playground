"""
Stage 3: two layers and a sigmoid, which together can solve XOR.

One neuron draws one straight line. XOR is not separable by one straight line,
so it needs more than one layer. Stacking layers is easy. The difficulty is
that a hidden unit never sees the target, so something has to work out how much
of the error was its doing.

WHAT TRAINING ACTUALLY IS

  The network is a pile of numbers: every weight and every bias. Run the four
  XOR examples through it and you get a loss, one number saying how wrong it
  currently is. Training means changing the pile so the loss goes down.

  There is a method for this that needs no theory. Take one weight, add a
  millionth to it, recompute the loss. Subtract a millionth, recompute the
  loss. The difference between those two says how much the loss responds to
  that weight, and its sign says which way to move it. Repeat for every weight.

  That method works, and it solves XOR. `python 03_network.py knobs` runs it.

  It also costs a full forward pass per weight per example. Backpropagation
  produces the same numbers in one pass. The three functions below are that
  optimisation.

WHAT A DELTA IS

  Write z for the sum arriving at a unit, before the sigmoid:

      z = bias + sum(incoming_value * weight)
      output = sigmoid(z)

  Measure the loss sensitivity of each weight feeding one unit the brute-force
  way, and the answers share a factor:

      weight      arrived on it    dLoss/dweight       ratio
      w_out[0]         0.576399        -0.080666   -0.139948
      w_out[1]         0.597010        -0.083550   -0.139948
      w_out[2]         0.416625        -0.058306   -0.139948
      b_out            1.000000        -0.139948   -0.139948

  Those weights differ only in what arrives on them.
  The shared ratio is the unit's DELTA:

    delta = how much the loss changes per unit change in z

  A delta belongs to a unit, not to a weight. This network has three hidden
  units and one output unit, so it has four deltas per example, and every
  weight update in it is one of those four times what arrived on the weight.
  `python 03_network.py deltas` prints the table above from live measurements.

  `output - target` is a different quantity. It says how wrong the output is.
  A delta says how much the loss would move if the unit's input sum moved. The
  two disagree exactly where it matters: a saturated unit can be as wrong as it
  is possible to be and still have a delta near zero, because moving its z
  barely moves its output.

SENSITIVITIES ALONG A PATH MULTIPLY

  If moving A moves B at two units per unit, and moving B moves C at three
  units per unit, then moving A moves C at six. Ratios in series multiply.

  A hidden unit reaches the loss along a path: its own z, its output h, the
  weight w_out that carries h forward, the output unit's z, the loss. Its delta
  is the product of the ratios along that path. Two of those ratios are handed
  to you: w_out is the weight itself, and the output unit's delta already
  accounts for everything past it. That is why hidden_delta() takes the three
  arguments it takes.

SIGMOID SLOPE

  sigmoid_slope(a) is the ratio between a change in z and the change it causes
  in the unit's output. Its argument is an ACTIVATION: a number the sigmoid
  produced, between 0 and 1. Its result is between 0 and 0.25.

  An error term like (y - target) runs from -1 to
  1, and sigmoid_slope(-0.5) returns -0.75, which is a slope no sigmoid has at
  any point.

  Its result approaches zero as the activation approaches 0 or 1. A unit that
  has committed to an answer changes very little when z moves, so its delta is
  small however wrong it is. Several rows in the tables below test this.

WHY SIGMOID AND NOT THE STEP FUNCTION FROM 02

  A step function is flat everywhere and vertical at one point, so its slope is
  either zero or undefined. Every sensitivity computed through it is therefore
  zero or undefined, and the whole scheme collapses. The sigmoid has a usable
  slope at every point.

SIGN CONVENTION

  02 computed `target - prediction` and added the update. This file computes a
  loss sensitivity, which points uphill, and subtracts it.

THE THREE FUNCTIONS TO FIX

  output_delta()    the output unit's delta. It can see the target, so it can
                    be computed directly.
  hidden_delta()    a hidden unit's delta. It cannot see the target, so it is
                    computed from the delta of the unit it feeds and the weight
                    it fed along.
  updated_weight()  one weight's move, given the delta of the unit the weight
                    feeds into and the value that arrived on it.

CHECKING YOUR WORK

  The rows in the tables are not the only evidence available. true_delta()
  below measures the correct output delta for any (y, target) by nudging z and
  watching the loss, with no formula involved, so you can test as many cases as
  you like. `python 03_network.py check` sweeps it against yours.

  One sweep worth running holds the error fixed and varies y. Anything that
  depends only on the error stays flat across it. Anything that responds to
  saturation does not.

  When a row passes, check whether it could have failed. Where target is 0,
  `y - target` and `y` are the same number, so those rows cannot tell the two
  apart.

WHY THREE HIDDEN UNITS AND NOT TWO

  Two is the minimum needed to represent XOR, and hand_wire() below sets two by
  hand to prove it. Whether training FINDS a solution depends on where the
  random weights started. Over the first 100 seeds, at the rate, epochs and
  init_scale set below:

      two hidden units    80 of 100 starts
      three hidden units  99 of 100 starts   (seed 51 is the one that fails)

  These are measured. `python 03_network.py seeds 100` reruns the sweep at any
  width. "The network can represent the answer" and "training will find the
  answer" are separate claims, and only the first one is guaranteed.

Properties these should have:

  1. A unit that is already saturated (output near 0 or 1) barely moves, even
     when it is badly wrong.
  2. A hidden unit connected by a zero weight gets zero blame. It did not
     contribute. A negative connection gets blame of the opposite sign. Twice
     the connection, twice the blame.
  3. A weight only moves in proportion to what actually arrived on it. Nothing
     arrived, nothing moves.
  4. No mistake anywhere means nothing moves anywhere.
"""

import math
import random
import sys

from harness import (CROSS, TICK, Cell, ExerciseTable, approx, console,
                     fmt_float, not_written, prediction_row, section, summary)


def sigmoid(z):
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_slope(a):
    """Slope of the sigmoid at the point where it produced the activation `a`.

    `a` is an ACTIVATION: a number the sigmoid produced, so 0 < a < 1. The
    result is between 0 and 0.25, and approaches 0 at both ends. Passing an
    error term or any other quantity here is meaningless, and the result will
    be a number no sigmoid slope can take.
    """
    return a * (1.0 - a)


# ------------------------------------------------------- measurement, not math
# Tools for checking your answers against the network's actual behaviour.
# Nothing here needs fixing, and none of it is used during training -- it is
# slow on purpose, because it assumes nothing and derives nothing.

NUDGE = 1e-6


def true_delta(y, target, nudge=NUDGE):
    """Measure the output unit's delta, with no formula involved.

    Reconstructs the z that produced `y`, moves it a hair each way, and reports
    how much the loss changed per unit of z. That is the definition of a delta,
    so this is the answer output_delta() has to reproduce -- for any inputs you
    like, not only the six rows in the table.
    """
    z = math.log(y / (1.0 - y))
    loss = lambda zz: 0.5 * (sigmoid(zz) - target) ** 2
    return (loss(z + nudge) - loss(z - nudge)) / (2.0 * nudge)


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


SEED_SWEEP = 30


def seed_sweep(out_fn, hid_fn, w_fn, n_hidden, seeds):
    """How often does training from scratch actually land on XOR?

    run_xor() above trains one network from one seed. That says nothing about
    how often it works, and how often it works is the whole point of the
    "three units, not two" claim in the header.
    """
    failed = []
    for seed in range(seeds):
        net = Network(out_fn, hid_fn, w_fn, n_hidden=n_hidden, seed=seed)
        net.train(XOR)
        if not net.solves(XOR):
            failed.append(seed)
    return seeds - len(failed), failed


def claim(text, holds):
    console.print(f"  [{'bold green' if holds else 'bold red'}]"
                  f"{TICK if holds else CROSS}[/] {text}")
    return holds


def run_seed_sweep(out_fn, hid_fn, w_fn, seeds=SEED_SWEEP):
    console.print(f"\n[dim]training from scratch {seeds} times at each width, "
                  f"seeds 0..{seeds - 1}[/dim]\n")
    rates = {}
    for n_hidden in (2, 3):
        ok, failed = seed_sweep(out_fn, hid_fn, w_fn, n_hidden, seeds)
        rates[n_hidden] = ok
        bad = (f"  [dim]failed from seed{'s' if len(failed) > 1 else ''} "
               f"{', '.join(str(f) for f in failed[:8])}"
               f"{' ...' if len(failed) > 8 else ''}[/dim]") if failed else ""
        console.print(f"  [bold cyan]{n_hidden} hidden units[/bold cyan]  "
                      f"found XOR from [bold]{ok}/{seeds}[/bold] starts{bad}")

    console.print()
    good = claim("two units does not always get there -- capacity is not "
                 "the same as trainability", rates[2] < seeds)
    good &= claim("three units gets there more often than two",
                  rates[3] > rates[2])
    good &= claim(f"three units is reliable: at least 90% of starts "
                  f"({rates[3]}/{seeds})", rates[3] >= 0.9 * seeds)
    return good


# ------------------------------------------------- backprop, the slow way round
# Everything in this section arrives at the same numbers the three functions
# above are supposed to produce, without using them. It is here so that "delta"
# refers to something you can watch happen.


def all_knobs(net):
    """Every adjustable number in the network, as (read, write) pairs.

    A network is a pile of knobs and one dial reading how wrong it is. This
    enumerates the knobs.
    """
    for j in range(len(net.w_hidden)):
        for i in range(len(net.w_hidden[j])):
            yield (lambda j=j, i=i: net.w_hidden[j][i],
                   lambda v, j=j, i=i: net.w_hidden[j].__setitem__(i, v))
        yield (lambda j=j: net.b_hidden[j],
               lambda v, j=j: net.b_hidden.__setitem__(j, v))
        yield (lambda j=j: net.w_out[j],
               lambda v, j=j: net.w_out.__setitem__(j, v))
    yield (lambda: net.b_out, lambda v: setattr(net, "b_out", v))


def batch_loss(net, examples):
    return sum((net.predict(x) - t) ** 2 for x, t in examples) / len(examples)


def knob_step(net, examples, rate=2.0, nudge=1e-5):
    """One training step containing no backpropagation at all.

    Turn each knob a hair each way, keep whichever direction lowered the loss,
    move it that way. No deltas, no chain rule, no sigmoid slope.
    """
    moves = []
    for read, write in all_knobs(net):
        original = read()
        write(original + nudge)
        up = batch_loss(net, examples)
        write(original - nudge)
        down = batch_loss(net, examples)
        write(original)
        moves.append((write, original - rate * (up - down) / (2.0 * nudge)))
    for write, moved in moves:
        write(moved)


def run_knobs(epochs=3000):
    net = Network(None, None, None, seed=0)
    console.print(f"\n  loss before  [bold]{batch_loss(net, XOR):.4f}[/bold]"
                  f"   [dim](0.25 is what guessing scores)[/dim]")
    for _ in range(epochs):
        knob_step(net, XOR)
    console.print(f"  loss after   [bold]{batch_loss(net, XOR):.4f}[/bold]\n")
    for x, t in XOR:
        y = net.predict(x)
        prediction_row(x, t, round(y), raw=y)
    solved = net.solves(XOR)
    console.print(f"  [bold green]{TICK} XOR SOLVED[/bold green]" if solved
                  else "  [yellow]not solved[/yellow]")
    console.print("\n[dim italic]No deltas were involved. This is the whole "
                  "problem, solved the expensive way: one forward pass per "
                  "weight, per example, per step.[/dim italic]")
    return solved


def weight_sensitivity(net, read, write, x, target, nudge=NUDGE):
    """How much the loss responds to one weight, measured by nudging it."""
    original = read()
    write(original + nudge)
    up = 0.5 * (net.predict(x) - target) ** 2
    write(original - nudge)
    down = 0.5 * (net.predict(x) - target) ** 2
    write(original)
    return (up - down) / (2.0 * nudge)


def _sensitivity_table(net, x, target, pairs):
    console.print(f"  [dim]{'weight':<16}{'arrived on it':>16}"
                  f"{'dLoss/dweight':>16}{'ratio':>13}[/dim]")
    for name, arrived, read, write in pairs:
        s = weight_sensitivity(net, read, write, x, target)
        ratio = "  [dim]undefined[/dim]" if arrived == 0 else f"{s / arrived:>13.6f}"
        console.print(f"  [cyan]{name:<16}[/cyan]{arrived:>16.6f}{s:>16.6f}{ratio}")


def show_deltas(seed=4):
    """Where the word "delta" comes from: the factor the weights share."""
    net = Network(None, None, None, seed=seed)
    x, target = (1, 0), 1
    h, _ = net.forward(x)

    console.print(f"\n  [dim]input {x}, target {target}[/dim]")

    console.print("\n[bold]every weight feeding the OUTPUT unit[/bold]")
    _sensitivity_table(net, x, target, [
        *[(f"w_out[{j}]", h[j],
           (lambda j=j: net.w_out[j]),
           (lambda v, j=j: net.w_out.__setitem__(j, v))) for j in range(len(h))],
        ("b_out", 1.0, (lambda: net.b_out),
         (lambda v: setattr(net, "b_out", v))),
    ])

    console.print("\n[bold]every weight feeding HIDDEN UNIT 1[/bold]")
    _sensitivity_table(net, x, target, [
        *[(f"w_hidden[1][{i}]", float(x[i]),
           (lambda i=i: net.w_hidden[1][i]),
           (lambda v, i=i: net.w_hidden[1].__setitem__(i, v)))
          for i in range(len(x))],
        ("b_hidden[1]", 1.0, (lambda: net.b_hidden[1]),
         (lambda v: net.b_hidden.__setitem__(1, v))),
    ])

    console.print("\n[dim italic]One ratio per unit, shared by every weight "
                  "that feeds it. Those weights differ only in what arrives on "
                  "them, so they cannot help but share it. That shared number "
                  "is the unit's delta.[/dim italic]")


def run_check():
    """Sweep output_delta() against measurement, well past the six table rows."""
    if not is_written(output_delta, (0.5, 1)):
        not_written("the check")
        return False

    console.print("\n[bold]error held constant at 0.10; only y changes[/bold]")
    console.print("[dim]anything driven by the error alone is flat down this "
                  "column[/dim]\n")
    console.print(f"  [dim]{'y':>8}{'target':>9}{'measured':>13}"
                  f"{'yours':>13}[/dim]")
    worst = 0.0
    for y in (0.15, 0.3, 0.5, 0.7, 0.9, 0.99):
        t = y - 0.10
        want, got = true_delta(y, t), output_delta(y, t)
        worst = max(worst, abs(want - got))
        mark = "green" if approx(want, got) else "red"
        console.print(f"  {y:>8.2f}{t:>9.2f}{want:>13.6f}"
                      f"  [{mark}]{got:>11.6f}[/{mark}]")

    console.print("\n[bold]y held constant at 0.90; only the error "
                  "changes[/bold]")
    console.print("[dim]the saturation factor is frozen down this one[/dim]\n")
    console.print(f"  [dim]{'y':>8}{'target':>9}{'measured':>13}"
                  f"{'yours':>13}[/dim]")
    for t in (0.0, 0.25, 0.5, 0.75, 0.9, 1.0):
        want, got = true_delta(0.90, t), output_delta(0.90, t)
        worst = max(worst, abs(want - got))
        mark = "green" if approx(want, got) else "red"
        console.print(f"  {0.90:>8.2f}{t:>9.2f}{want:>13.6f}"
                      f"  [{mark}]{got:>11.6f}[/{mark}]")

    ok = worst < 1e-6
    console.print()
    claim(f"output_delta() matches measurement everywhere "
          f"(worst gap {worst:.2e})", ok)
    return ok


def plot():
    """The sigmoid always draws. The network's region needs your three."""
    import plots

    plots.sigmoid_figure(sigmoid, sigmoid_slope)

    if all(is_written(fn, probe) for fn, probe in
           [(output_delta, (0.5, 1)), (hidden_delta, (0.5, 1.0, 0.2)),
            (updated_weight, (0.0, 0.5, 1.0, 0.5))]):
        net = Network(output_delta, hidden_delta, updated_weight)
        net.train(XOR)
        plots.network_region_figure(net.predict, net.solves(XOR))
    else:
        console.print("\n[yellow]Drawing the sigmoid only.[/yellow]")
        console.print("[dim]The network's decision surface needs all three "
                      "functions -- an untrained one has nothing to show.[/dim]")
    plots.done("03")


if __name__ == "__main__":
    if "plot" in sys.argv:
        plot()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "knobs":
        section("XOR with no backpropagation at all")
        run_knobs()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "deltas":
        section("where the word delta comes from")
        show_deltas()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "check":
        section("output_delta() against measurement, beyond the table rows")
        run_check()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "seeds":
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        section(f"how often does training find XOR at all? ({count} starts)")
        if all(is_written(fn, probe) for fn, probe in
               [(output_delta, (0.5, 1)), (hidden_delta, (0.5, 1.0, 0.2)),
                (updated_weight, (0.0, 0.5, 1.0, 0.5))]):
            run_seed_sweep(output_delta, hidden_delta, updated_weight, count)
        else:
            not_written("the seed sweep")
        sys.exit(0)

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

        section("that was one random start -- how often does it work?")
        if run_seed_sweep(output_delta, hidden_delta, updated_weight):
            console.print("\n  [dim]run `python 03_network.py seeds 100` for a "
                          "longer sweep[/dim]")
    else:
        not_written("with yours")
