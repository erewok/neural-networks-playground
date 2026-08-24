"""
Stage 1: a single neuron's forward pass.

This is the smallest unit in a neural network. It does not learn yet -- it is a
weighted vote followed by a yes/no decision.

    inputs  x = [x1, x2, ...]      the evidence
    weights w = [w1, w2, ...]      how much each piece of evidence matters
    bias    b                      how much evidence is needed before firing

The neuron combines all the evidence into a single number, then decides whether
that number is big enough to fire.

Properties a neuron is supposed to have. Every one of them is observable in
the test output below, if you pick the right test to stare at.

  1. The bias must actually take part in the vote. A neuron with a very negative
     bias should be hard to trigger; one with a positive bias should fire on
     almost anything. If you can change the bias and watch nothing happen,
     the bias is not doing its job.

  2. The bias must be the ONLY thing with an opinion about where the bar sits.
     Moving the bias alone should be able to slide the decision boundary
     anywhere you like. Nothing else in the code gets a vote on the threshold.

  3. A negative weight is evidence AGAINST firing. Weights are allowed to
     subtract, not just add -- that is how a neuron says "this input makes me
     less likely to fire", and it is the only way to build anything involving
     NOT. A weight's sign must survive all the way into the total.

  4. Each input must be paired with its own weight, and only its own. A neuron
     with weights [1, 0] must respond to the first input and completely ignore
     the second -- not the other way around.

  5. Everything above has a picture. A neuron fires on one side of a straight
     line and stays quiet on the other, and weights and bias are simply where
     that line sits: the weights tilt it, the bias slides it. boundary_x2()
     below computes that line, and `just plot 1` draws it.

A test that comes out correct is telling you which of these properties it
never depended on. That is often more informative than one that comes out
wrong, so read the whole table, not just the bad rows.
"""

import sys
from itertools import product
from operator import mul
from itertools import starmap

from harness import (Cell, ExerciseTable, approx, console, exact, fmt_float,
                     fmt_int, section, summary)


# --------------------------------------------------------------- the originals
# Left here on purpose. Don't edit these -- they are the thing you are comparing
# against, and the harness below runs them next to yours row by row.


def weighted_sum_broken(inputs, weights, bias):
    """Combine the evidence into a single number."""
    total = 0.0
    for x, w in zip(inputs, reversed(weights)):
        total += abs(x * w)
    return total


def step_broken(z):
    """Turn a number into a decision: fire (1) or stay quiet (0)."""
    return 1 if z > 0.5 else 0


def boundary_x2_broken(x1, weights, bias):
    """For this x1, the x2 that puts the neuron exactly on the fence."""
    return -(weights[0] * x1) / weights[1]


# -------------------------------------------------------------------- yours
# Write these two. Until you do, the harness runs with the "yours" column blank,
# so you can start on either one and see partial progress immediately.


def weighted_sum(inputs, weights, bias):
    """Combine the evidence into a single number."""
    total = sum(starmap(mul, zip(inputs, weights)))
    return total + bias

def step(z):
    """Turn a number into a decision: fire (1) or stay quiet (0)."""
    if z > 0:
        return 1
    return 0


def boundary_x2(x1, weights, bias):
    """For this x1, the x2 that puts the neuron exactly on the fence.

    The fence is every point where the weighted sum comes out to exactly zero
    -- one step either side and the answer flips. Given x1, solve for the x2
    that lands there.

    Assume weights[0] is not zero. (When it is, the fence is a vertical line
    and no function of x1 can describe it. Nothing here tests that case.)
    """
    return (x1 / weights[0]) - bias


# ---------------------------------------------------------------- test harness
# Drawing only -- see harness.py. Nothing here needs fixing.


def ready():
    """Have both of yours been written? Only NotImplementedError counts as 'no'
    -- any other exception is a real error and should reach you as a traceback."""
    try:
        weighted_sum((0, 0), (0.0, 0.0), 0.0)
        step(0.0)
    except NotImplementedError:
        return False
    return True


def gate(name, weights, bias, cases, note=""):
    """cases: list of (inputs, expected_output)."""
    mine = ready()
    t = ExerciseTable(
        title=f"{name}    weights={weights}  bias={bias}",
        note=note,
        input_header="x",
        yours_written=mine,
        fmt=fmt_int,
        compare=exact,
    )
    for x, want in cases:
        zb = weighted_sum_broken(x, weights, bias)
        original = Cell(step_broken(zb), detail=f"z={zb:+.2f}")
        yours = None
        if mine:
            zm = weighted_sum(x, weights, bias)
            yours = Cell(step(zm), detail=f"z={zm:+.2f}")
        label = "(" + ", ".join(f"{v:g}" for v in x) + ")"
        t.add(label, want, original, yours)
    return t.render()


def compare(title, original, mine, argnames, cases, note=""):
    """cases: list of (args_tuple, expected). Same idea as gate(), different shape."""
    written = is_written(mine, cases[0][0])
    t = ExerciseTable(
        title=title,
        note=note,
        input_header="arguments",
        yours_written=written,
        fmt=fmt_float(2),
        compare=approx,
    )
    for args, want in cases:
        label = ", ".join(f"{n}={v}" for n, v in zip(argnames, args))
        t.add(label, want, Cell(original(*args)),
              Cell(mine(*args)) if written else None)
    return t.render()


def is_written(fn, probe_args):
    try:
        fn(*probe_args)
    except NotImplementedError:
        return False
    return True


def fence_z(x1, weights, bias, ws_fn, fence_fn):
    """(the x2 the fence returned, the neuron's own z at that spot).

    Nothing hardcoded: it asks the fence where the boundary is, then asks the
    neuron what it computes there. On the fence that number is zero -- not
    because a table says so, but because zero is what "on the fence" means.
    So the column reads like every other table here: a number you can trace
    back to the inputs, and a miss you can measure rather than just count.
    """
    x2 = fence_fn(x1, weights, bias)
    return x2, ws_fn((x1, x2), weights, bias)


def fences(cases, note=""):
    """cases: list of (x1, weights, bias). Every row wants a z of zero."""
    written = ready() and is_written(boundary_x2, (0.0, (1.0, 1.0), -1.5))
    t = ExerciseTable(
        title="the fence and the neuron, checked against each other",
        note=note,
        input_header="x1, weights, bias",
        yours_written=written,
        fmt=fmt_float(2),
        compare=approx,
    )
    for x1, weights, bias in cases:
        label = f"x1={x1:g}, w={weights}, b={bias:+g}"
        x2b, zb = fence_z(x1, weights, bias, weighted_sum_broken,
                          boundary_x2_broken)
        yours = None
        if written:
            x2m, zm = fence_z(x1, weights, bias, weighted_sum, boundary_x2)
            yours = Cell(zm, detail=f"x2={x2m:+.2f}  ->")
        t.add(label, 0.0, Cell(zb, detail=f"x2={x2b:+.2f}  ->"), yours)
    return t.render()


def bits(n):
    """All 0/1 combinations of n inputs, in counting order."""
    return list(product([0, 1], repeat=n))


def truth(expected_bits, n=2):
    return list(zip(bits(n), expected_bits))


# Four neurons for the fences figure, each with the answers it is supposed to
# give. AND and OR deliberately share a weight vector and differ only in the
# bias, so "the bias slides the fence without turning it" is a comparison
# between two neighbouring panels rather than a claim in a caption.
FAMILIES = [
    ("AND", (1.0, 1.0), -1.5, truth([0, 0, 0, 1]), "one corner sliced off"),
    ("OR", (1.0, 1.0), -0.5, truth([0, 1, 1, 1]),
     "same weights as AND, bias slid: same tilt, moved"),
    ("x1 AND NOT x2", (1.0, -1.0), -0.5, truth([0, 0, 1, 0]),
     "one weight negative, and the tilt turns"),
    ("x1 worth double", (2.0, 1.0), -1.5, truth([0, 0, 1, 1]),
     "x2 has a weight and still never decides anything"),
]


def plot():
    """Draw what the tables can only describe.

    Each figure asks only for the functions it actually uses, so partial
    progress draws partial output rather than nothing: the surface figure needs
    weighted_sum and step, and the overlay needs boundary_x2 on top of those.
    """
    import plots

    if not ready():
        console.print("\n[yellow]The gate plots need weighted_sum and step "
                      "written first.[/yellow]")
        console.print("[dim]Drawing the XOR figure only -- that one needs "
                      "nothing from you.[/dim]")
        plots.xor_figure()
        plots.done("01-xor")
        return

    # One gate, in full: the plane, a cut through it, and what step() leaves.
    # This one never calls boundary_x2 -- it finds the fence in the surface it
    # already computed -- so it draws as soon as the neuron itself works.
    plots.neuron_surface_figure("AND", (1.0, 1.0), -1.5, truth([0, 0, 0, 1]),
                                weighted_sum, step)

    # Then every gate on shared axes, plus the one gate none of them reach.
    if is_written(boundary_x2, (0.0, (1.0, 1.0), -1.5)):
        plots.line_limit_figure(FAMILIES, boundary_x2, weighted_sum, step)
    else:
        console.print("\n[yellow]Skipping the fences figure: it draws "
                      "boundary_x2, which is not written yet. Drawing the "
                      "XOR half on its own instead.[/yellow]")
        plots.xor_figure()

    plots.done("01")


if __name__ == "__main__":
    section("one neuron, forward pass only")
    results = []

    # --- two-input logic gates, all weights positive ---------------------
    results.append(gate(
        "AND", [1.0, 1.0], -1.5, truth([0, 0, 0, 1]),
        "each input is worth 1 point; you need more than 1.5 points to fire"))

    results.append(gate(
        "OR", [1.0, 1.0], -0.5, truth([0, 1, 1, 1]),
        "each input is worth 1 point; you need more than 0.5 points"))

    # --- a bar the evidence can land exactly on -------------------------
    # Same truth table as AND, reached differently: here (0,1) and (1,0) put
    # the total at precisely the bar rather than half a point short of it.
    # Everywhere else in this file the total clears or misses by 0.5 or more,
    # so without these rows nothing says what happens ON the bar.
    results.append(gate(
        "AND, landing on the bar", [1.0, 1.0], -1.0, truth([0, 0, 0, 1]),
        "each input is worth 1 point and you need more than 1 -- "
        "so one input alone lands exactly on the bar"))

    # --- asymmetric weights: does each input reach its own weight? -------
    results.append(gate(
        "PASS-THROUGH x1", [1.0, 0.0], -0.5, truth([0, 0, 1, 1]),
        "only x1 counts; x2 is weighted zero and should be ignored entirely"))

    # --- negative weights: can this neuron vote against itself? ----------
    results.append(gate(
        "NOT x1", [-1.0, 0.0], 0.5, truth([1, 1, 0, 0]),
        "fires by default; x1 being on should push it back down"))

    results.append(gate(
        "NAND", [-1.0, -1.0], 1.5, truth([1, 1, 1, 0]),
        "fires unless both inputs are on"))

    results.append(gate(
        "x1 AND NOT x2", [1.0, -1.0], -0.5, truth([0, 0, 1, 0]),
        "mixed signs: x1 argues for firing, x2 argues against"))

    results.append(gate(
        "NOR", [-1.0, -1.0], 0.5, truth([1, 0, 0, 0]),
        "fires only when both inputs are off"))

    # --- three inputs with unequal weights -------------------------------
    results.append(gate(
        "WEIGHTED VOTE", [3.0, 2.0, 1.0], -3.5,
        truth([0, 0, 0, 0, 0, 1, 1, 1], n=3),
        "x1 is worth 3 points, x2 is worth 2, x3 is worth 1; you need 3.5+"))

    # --- inputs that are not 0/1 -----------------------------------------
    # Nothing says inputs have to be binary. Real ones rarely are.
    results.append(gate(
        "REAL-VALUED", [2.0, -1.0], -1.0, [
            ((1.0, 0.0), 1),
            ((0.5, 0.0), 0),
            ((0.5, 1.0), 0),
            ((2.0, 1.0), 1),
            ((0.0, -2.0), 1),
            ((3.0, 5.0), 0),
        ],
        "x1 helps twice as much as x2 hurts; a negative input flips its own sign"))

    results.append(compare(
        "boundary_x2(x1, weights, bias)", boundary_x2_broken, boundary_x2,
        ["x1", "weights", "bias"],
        [
            ((0.0, (1.0, 1.0), -1.5), +1.50),   # AND's fence, at the left edge
            ((1.0, (1.0, 1.0), -1.5), +0.50),   # and at the right edge
            ((0.0, (1.0, 1.0), -0.5), +0.50),   # OR sits lower down
            ((1.0, (1.0, 1.0), -0.5), -0.50),
            ((0.0, (1.0, -1.0), -0.5), -0.50),  # a fence that tilts the other way
            ((1.0, (1.0, -1.0), -0.5), +0.50),
            ((1.0, (2.0, -1.0), -1.0), +1.00),  # unequal weights tilt it further
            ((0.0, (-1.0, -1.0), 1.5), +1.50),  # NAND: same fence as AND
        ],
        "two points make a line; these are the ends of each gate's fence"))

    # The table above scores boundary_x2 against eight answers written down in
    # advance. It never asks the neuron whether that is really where it changes
    # its mind. Property 5 says the fence IS the flip, so check the two halves
    # against each other rather than against a number -- and on weights the
    # eight rows never visit.
    results.append(fences(
        [
            (0.0, (1.0, 1.0), -1.5),     # AND, same fence as above
            (1.0, (1.0, -1.0), -0.5),    # mixed signs
            (-2.0, (0.5, 4.0), -1.0),    # a steep fence, weights unlike any row
            (3.0, (3.0, -0.5), 2.0),     # shallow, and tilted the other way
            (-0.5, (-2.0, -3.0), -1.5),  # both weights negative
            (0.0, (1.0, 1.0), 0.0),      # a fence through the origin
        ],
        "no answer key here: each row feeds the x2 your fence returned back "
        "into your own neuron. on the fence, z is zero"))

    summary(results)

    if "plot" in sys.argv:
        plot()
