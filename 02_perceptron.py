"""
Stage 2: the perceptron -- the neuron from 01, plus a learning rule.

The forward pass here is its own copy, and it is correct, so you can work on
this file independently of 01.

The only new idea: when the neuron gets an answer wrong, nudge the weights so
that next time it would be less wrong. Do that over and over and the weights
drift into a shape that gets everything right.

HOW THIS FILE IS LAID OUT
  PerceptronBase   the parts that are not in question: the forward pass, and
                   the training loop that walks the examples epoch by epoch.
  Original         the existing rule, left untouched to compare against.
  Yours            one method to write: update().

  Everything hinges on that one method. It is handed a single example, applies
  whatever change it thinks is warranted, and returns True if the neuron got
  that example wrong. The loop around it is already correct, and the `mistakes`
  counter it prints does nothing but add up those return values, so it is an
  honest signal.

Properties the learning rule is supposed to have:

  1. It must move toward the right answer, not away from it. If the neuron said
     0 when it should have said 1, the evidence it saw should end up counting
     for MORE afterward, not less.

  2. Only weights that were actually responsible for the mistake should change.
     If an input was switched off, it did not contribute to the error, so its
     weight has no business moving. (This is called credit assignment, and it
     is the same idea that backpropagation generalizes to many layers.)

  3. The bias has to learn too. It is the neuron's threshold; if it never moves,
     the neuron is stuck with whatever threshold it happened to start with.

One thing worth knowing before you start, so you don't chase it: a single
neuron genuinely cannot learn XOR. That is a real mathematical limit, not a
defect in this file. AND and OR converging while XOR never does is the correct
end state here, and it is the reason 03 exists.
"""


def weighted_sum(inputs, weights, bias):
    total = bias
    for x, w in zip(inputs, weights):
        total += x * w
    return total


def step(z):
    return 1 if z > 0 else 0


# ------------------------------------------------------------ shared machinery


class PerceptronBase:
    def __init__(self, n_inputs, rate=0.1):
        # Starting from all zeros is fine for a single perceptron.
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.rate = rate

    def predict(self, inputs):
        return step(weighted_sum(inputs, self.weights, self.bias))

    def update(self, inputs, target):
        """Learn from ONE example. Return True if the neuron got it wrong."""
        raise NotImplementedError

    def train(self, examples, epochs=20, verbose=True):
        """examples: list of (inputs, target). This loop is correct."""
        for epoch in range(epochs):
            mistakes = 0
            for inputs, target in examples:
                if self.update(inputs, target):
                    mistakes += 1
            if verbose:
                w = ", ".join(f"{v:+.1f}" for v in self.weights)
                print(f"  epoch {epoch:2d}  mistakes={mistakes}  w=[{w}]  b={self.bias:+.1f}")
            if mistakes == 0:
                if verbose:
                    print("  converged -- no mistakes left, weights stop moving")
                return True
        if verbose:
            print("  gave up: still making mistakes after all epochs")
        return False


# ---------------------------------------------------------------- the original
# Left here on purpose. Don't edit it -- it is what you are comparing against.


class Original(PerceptronBase):
    def update(self, inputs, target):
        error = self.predict(inputs) - target
        if error == 0:
            return False
        for i, x in enumerate(inputs):
            self.weights[i] += self.rate * error
        return True


# ---------------------------------------------------------------------- yours


class Yours(PerceptronBase):
    def update(self, inputs, target):
        """Learn from ONE example. Return True if the neuron got it wrong.

        You have self.weights, self.bias, self.rate, and self.predict(inputs).
        """
        raise NotImplementedError


def ready():
    try:
        Yours(2).update((0, 0), 0)
    except NotImplementedError:
        return False
    return True


# ------------------------------------------------------------------ the tasks


AND = [((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)]
OR  = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]
XOR = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]

TASKS = [("AND", AND), ("OR", OR), ("XOR", XOR)]

# XOR is here to fail. Everything else should end up solved.
SHOULD_SOLVE = {"AND": True, "OR": True, "XOR": False}


def report(p, examples):
    ok = True
    for inputs, target in examples:
        got = p.predict(inputs)
        hit = got == target
        ok = ok and hit
        print(f"    x={inputs} want={target} got={got}  {'ok ' if hit else 'WRONG'}")
    return ok


def run(cls, label):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
    outcomes = {}
    for name, data in TASKS:
        print(f"\ntraining on {name}:")
        p = cls(n_inputs=2)
        p.train(data)
        print(f"  {name} result:")
        outcomes[name] = report(p, data)
    return outcomes


def verdict(outcomes):
    print("\n  scorecard:")
    for name, _ in TASKS:
        want = SHOULD_SOLVE[name]
        got = outcomes[name]
        state = "solved" if got else "not solved"
        expect = "should solve" if want else "cannot be solved by one neuron"
        flag = "as expected" if got == want else ">>> not what we want"
        print(f"    {name:<4} {state:<11} ({expect})  {flag}")
    return all(outcomes[n] == SHOULD_SOLVE[n] for n, _ in TASKS)


if __name__ == "__main__":
    verdict(run(Original, "ORIGINAL"))

    if ready():
        good = verdict(run(Yours, "YOURS"))
        if good:
            print("\n  AND and OR solved, XOR correctly not solved.")
    else:
        print(f"\n{'=' * 60}\nYOURS: not written yet -- see Yours.update()\n{'=' * 60}")
