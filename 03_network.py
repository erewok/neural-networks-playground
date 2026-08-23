"""
Stage 3: a two-layer network -- the thing that can finally solve XOR.
This one is a partial build: some of it is written, some of it is left to you.

WHAT IS ALREADY WRITTEN
  - the forward pass, completely (2 inputs -> N hidden units -> 1 output)
  - sigmoid instead of the step function, so everything is now smooth
  - the loss, the training loop, the reporting
  - applying updates to both layers, once you tell it what the updates are
  - hand_wire(), which sets the weights by hand to a known XOR solution

WHAT IS LEFT TO YOU
  - see backward()

WHY SIGMOID REPLACED THE STEP FUNCTION
  A step function is flat everywhere, so it can never tell you "you were close"
  or "which direction is better" -- only "right" or "wrong". Sigmoid is a smooth
  S-curve, so at any point it has a slope, and a slope is a direction to move.
  Its slope, in terms of its own output a, is a * (1 - a). Notice that this is
  near zero when a is near 0 or 1: a unit that is already very confident barely
  moves. That fact will matter to you later.

THE SHAPE OF LEARNING HERE
  Each unit gets a "delta": how wrong it is, and in which direction. Once a unit
  knows its delta, its update is always the same shape -- nudge each incoming
  weight in proportion to both the delta and the input that arrived on that
  weight. That part is already written for both layers.

  The output unit can compute its own delta, because it can see the target.
  A hidden unit cannot. Nobody ever told it what it should have produced. All it
  can know is how much it contributed to somebody else's mistake -- and the only
  record of how much it contributed is the weight it sent its signal along.

  That is the missing piece.

Run it. Then read EXPERIMENTS.md.
"""

import math
import random
import sys


def sigmoid(z):
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def slope_from_output(a):
    """Slope of the sigmoid, expressed using the value it already produced."""
    return a * (1.0 - a)


class Network:
    def __init__(self, n_in=2, n_hidden=2, rate=0.5, seed=0, init_scale=1.0):
        rng = random.Random(seed)
        r = lambda: rng.uniform(-init_scale, init_scale)
        self.w_hidden = [[r() for _ in range(n_in)] for _ in range(n_hidden)]
        self.b_hidden = [r() for _ in range(n_hidden)]
        self.w_out = [r() for _ in range(n_hidden)]
        self.b_out = r()
        self.rate = rate

    # ---------- forward: complete and correct ----------

    def forward(self, x):
        """Returns (hidden activations, output activation)."""
        h = []
        for w_row, b in zip(self.w_hidden, self.b_hidden):
            z = b + sum(xi * wi for xi, wi in zip(x, w_row))
            h.append(sigmoid(z))
        z_out = self.b_out + sum(hj * wj for hj, wj in zip(h, self.w_out))
        return h, sigmoid(z_out)

    def predict(self, x):
        return self.forward(x)[1]

    # ---------- backward: one piece missing ----------

    def backward(self, x, h, y, target):
        # The output unit can see the target, so it can work out its own delta:
        # how far off it is, scaled by how much room the sigmoid has left to move.
        out_delta = (y - target) * slope_from_output(y)

        # A hidden unit never sees the target. It has to be handed its share of
        # the blame for out_delta. Right now each one is told it did nothing
        # wrong, so the hidden layer is frozen for the entire run.
        hidden_deltas = [0.0 for _ in h]        # <-- the missing piece

        # Applying the updates is correct for both layers. Note the two lines
        # have the identical shape: rate * (this unit's delta) * (what came in
        # on that weight). Nothing below needs changing.
        for j in range(len(h)):
            self.w_out[j] -= self.rate * out_delta * h[j]
        self.b_out -= self.rate * out_delta

        for j in range(len(h)):
            for i in range(len(x)):
                self.w_hidden[j][i] -= self.rate * hidden_deltas[j] * x[i]
            self.b_hidden[j] -= self.rate * hidden_deltas[j]

    # ---------- training loop: complete ----------

    def train(self, examples, epochs=4000, report_every=500, verbose=True):
        for epoch in range(epochs + 1):
            loss = 0.0
            for x, target in examples:
                h, y = self.forward(x)
                loss += (y - target) ** 2
                self.backward(x, h, y, target)
            loss /= len(examples)
            if verbose and epoch % report_every == 0:
                print(f"  epoch {epoch:5d}   loss {loss:.4f}")
        return loss

    def hand_wire(self):
        """A known-good XOR solution, set by hand. Requires exactly 2 hidden units.

        XOR is "at least one of them, but not both". So: one hidden unit detects
        'at least one', the other detects 'both', and the output says yes to the
        first while the second vetoes it. Large weights make each unit saturate,
        which is how a smooth sigmoid imitates a hard yes/no.
        """
        self.w_hidden = [[20.0, 20.0], [20.0, 20.0]]
        self.b_hidden = [-10.0, -30.0]
        self.w_out = [20.0, -20.0]
        self.b_out = -10.0
        return self


XOR = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]
AND = [((0, 0), 0), ((0, 1), 0), ((1, 0), 0), ((1, 1), 1)]
OR  = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]


def report(net, examples, name="result"):
    print(f"\n{name}:")
    ok = True
    for x, target in examples:
        y = net.predict(x)
        hit = round(y) == target
        ok = ok and hit
        print(f"  x={x} want={target}  got={y:.3f}  ->  {round(y)}  {'ok ' if hit else 'WRONG'}")
    print(f"  {'ALL CORRECT' if ok else 'not solved'}")
    return ok


def show_weights(net, label):
    print(f"\n{label}")
    for j, (row, b) in enumerate(zip(net.w_hidden, net.b_hidden)):
        cells = ", ".join(f"{v:+.3f}" for v in row)
        print(f"  hidden[{j}]  w=[{cells}]  b={b:+.3f}")
    cells = ", ".join(f"{v:+.3f}" for v in net.w_out)
    print(f"  output     w=[{cells}]  b={net.b_out:+.3f}")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"

    if mode == "handwired":
        net = Network(n_hidden=2).hand_wire()
        show_weights(net, "weights set by hand (no training at all):")
        report(net, XOR, "XOR, hand-wired")

    else:
        net = Network(n_hidden=2, rate=0.5, seed=0)
        show_weights(net, "weights before training:")
        print("\ntraining on XOR:")
        net.train(XOR)
        show_weights(net, "weights after training:")
        report(net, XOR, "XOR, trained")
