"""
Stage 2: the perceptron -- the neuron from 01, plus a learning rule.

The forward pass here is its own correct copy, so this file stands alone.

The only new idea: when the neuron gets an answer wrong, nudge the weights so
that next time it would be less wrong. Repeat until it stops being wrong.

That nudge breaks into three small decisions, and each one is its own function
to fix:

    error()           how wrong were we, and in which direction?
    updated_weight()  given that, where should one weight move to?
    updated_bias()    given that, where should the bias move to?

Properties they are supposed to have. Each is visible in the tables below.

  1. The correction has to point at the answer, not away from it. If the neuron
     said 0 when it should have said 1, whatever happens next must make firing
     MORE likely the next time it sees that input, not less.

  2. A weight only moves if its own input had something to do with the mistake.
     An input that was switched off contributed nothing and its weight must come
     back unchanged. An input twice as large should pull its weight twice as far.
     (This is credit assignment, the idea backpropagation later generalizes to
     many layers.)

  3. The bias moves on every mistake. It is the neuron's threshold, and a neuron
     that cannot move its threshold is stuck wherever it started.

  4. When there is no error, nothing moves at all. A converged neuron has to
     stay converged.

The last section wires your three functions into a real training loop, so you
can see arithmetic that passes the tables turn into a neuron that actually
learns. Both parts run automatically.

One thing worth knowing so you don't chase it: a single neuron genuinely cannot
learn XOR. That is a real mathematical limit, not a defect in this file. AND and
OR converging while XOR never does is the correct end state, and it is why 03
exists.
"""

from harness import (Cell, ExerciseTable, approx, console, fmt_float,
                     not_written, outcome, section, summary)


def weighted_sum(inputs, weights, bias):
    total = bias
    for x, w in zip(inputs, weights):
        total += x * w
    return total


def step(z):
    return 1 if z > 0 else 0


# --------------------------------------------------------------- the originals
# Left here on purpose. Don't edit these -- the harness runs them beside yours.


def error_broken(target, prediction):
    return prediction - target


def updated_weight_broken(w, x, err, rate):
    return w + rate * err


def updated_bias_broken(b, err, rate):
    return b


# ---------------------------------------------------------------------- yours
# Write these three. Each one is a single expression. The tables score whichever
# ones you have written and leave the rest blank, so you can go one at a time.


def error(target, prediction):
    """How wrong were we, and in which direction?"""
    raise NotImplementedError


def updated_weight(w, x, err, rate):
    """Where should the weight on input x move to?"""
    raise NotImplementedError


def updated_bias(b, err, rate):
    """Where should the bias move to?"""
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
    """cases: list of (args_tuple, expected)."""
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
        label = ", ".join(f"{n}={v:g}" for n, v in zip(argnames, args))
        t.add(label, want, Cell(original(*args)),
              Cell(mine(*args)) if written else None)
    return t.render()


# ------------------------------------------------------- the integration test
# Your three functions, dropped into a real training loop.


class Perceptron:
    def __init__(self, n_inputs, err_fn, w_fn, b_fn, rate=0.1):
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.rate = rate
        self.err_fn, self.w_fn, self.b_fn = err_fn, w_fn, b_fn

    def predict(self, inputs):
        return step(weighted_sum(inputs, self.weights, self.bias))

    def train(self, examples, epochs=30):
        for _ in range(epochs):
            mistakes = 0
            for inputs, target in examples:
                err = self.err_fn(target, self.predict(inputs))
                if err == 0:
                    continue
                mistakes += 1
                self.weights = [self.w_fn(w, x, err, self.rate)
                                for w, x in zip(self.weights, inputs)]
                self.bias = self.b_fn(self.bias, err, self.rate)
            if mistakes == 0:
                return True
        return False

    def solves(self, examples):
        return all(self.predict(x) == t for x, t in examples)


AND = [((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)]
OR  = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]
XOR = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]

TASKS = [("AND", AND, True), ("OR", OR, True), ("XOR", XOR, False)]


def train_all(label, err_fn, w_fn, b_fn):
    console.print(f"\n[bold]{label}[/bold]")
    good = True
    for name, data, should in TASKS:
        p = Perceptron(2, err_fn, w_fn, b_fn)
        p.train(data)
        w = ", ".join(f"{v:+.1f}" for v in p.weights)
        good &= outcome(
            name, p.solves(data), should,
            detail=f"w=[{w}] b={p.bias:+.1f}",
            why="should solve" if should else "one neuron cannot solve this")
    return good


if __name__ == "__main__":
    section("the learning rule, one function at a time")
    results = []

    results.append(compare(
        "error(target, prediction)", error_broken, error,
        ["target", "prediction"],
        [((1, 0), +1.0), ((0, 1), -1.0), ((1, 1), 0.0), ((0, 0), 0.0)],
        "we wanted `target`, the neuron said `prediction`"))

    results.append(compare(
        "updated_weight(w, x, err, rate)", updated_weight_broken, updated_weight,
        ["w", "x", "err", "rate"],
        [
            ((0.0, 1, +1.0, 0.1), +0.10),    # input on, we fired too little
            ((0.0, 0, +1.0, 0.1), +0.00),    # input off -- not this weight's fault
            ((0.5, 1, -1.0, 0.1), +0.40),    # input on, we fired too much
            ((0.5, 0, -1.0, 0.1), +0.50),    # input off again
            ((0.2, 2.0, +1.0, 0.1), +0.40),  # twice the input, twice the move
            ((0.2, 1, 0.0, 0.1), +0.20),     # no error, no movement
        ],
        "w is the current weight on input x; err came from error()"))

    results.append(compare(
        "updated_bias(b, err, rate)", updated_bias_broken, updated_bias,
        ["b", "err", "rate"],
        [
            ((0.0, +1.0, 0.1), +0.10),
            ((0.0, -1.0, 0.1), -0.10),
            ((0.3, +1.0, 0.1), +0.40),
            ((0.3, 0.0, 0.1), +0.30),
        ],
        "the bias has no input of its own to be scaled by"))

    summary(results)

    section("those same functions, wired into a training loop")
    train_all("with the originals", error_broken, updated_weight_broken,
              updated_bias_broken)
    if all(m is not None for _, m, _ in results):
        if train_all("with yours", error, updated_weight, updated_bias):
            console.print("\n  [bold green]AND and OR learned, "
                          "XOR correctly not learned.[/bold green]")
    else:
        not_written("with yours")
