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

A test that comes out correct is telling you which of these properties it
never depended on. That is often more informative than one that comes out
wrong, so read the whole table, not just the bad rows.
"""

from itertools import product

from harness import Cell, ExerciseTable, exact, fmt_int, section, summary


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


# -------------------------------------------------------------------- yours
# Write these two. Until you do, the harness runs with the "yours" column blank,
# so you can start on either one and see partial progress immediately.


def weighted_sum(inputs, weights, bias):
    """Combine the evidence into a single number."""
    raise NotImplementedError


def step(z):
    """Turn a number into a decision: fire (1) or stay quiet (0)."""
    raise NotImplementedError


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


def bits(n):
    """All 0/1 combinations of n inputs, in counting order."""
    return list(product([0, 1], repeat=n))


def truth(expected_bits, n=2):
    return list(zip(bits(n), expected_bits))


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

    summary(results)
