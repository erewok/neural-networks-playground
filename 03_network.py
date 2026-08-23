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


def is_written(fn, probe_args):
    try:
        fn(*probe_args)
    except NotImplementedError:
        return False
    return True


def table(name, original, mine, argnames, cases, note=""):
    written = is_written(mine, cases[0][0])
    print(f"\n{name}({', '.join(argnames)})")
    if note:
        print(f"  ({note})")
    ok_o = ok_m = 0
    for args, want, why in cases:
        shown = ", ".join(f"{n}={v:g}" for n, v in zip(argnames, args))
        go = original(*args)
        ho = abs(go - want) < 1e-6
        ok_o += ho
        line = f"  {shown:<36} want {want:+.6f} | orig {go:+.6f} {'ok   ' if ho else 'WRONG'}"
        if written:
            gm = mine(*args)
            hm = abs(gm - want) < 1e-6
            ok_m += hm
            line += f" | yours {gm:+.6f} {'ok   ' if hm else 'WRONG'}"
        else:
            line += " | yours --"
        print(line + f"   {why}")
    tail = f"  --> original {ok_o}/{len(cases)}"
    if written:
        tail += f"   yours {ok_m}/{len(cases)}"
    print(tail)
    return (ok_o, ok_m if written else None, len(cases))


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
    print(f"\n{label}")
    net = Network(out_fn, hid_fn, w_fn)
    before = [f"{v:+.3f}" for row in net.w_hidden for v in row]
    loss = net.train(XOR)
    after = [f"{v:+.3f}" for row in net.w_hidden for v in row]
    print(f"  final loss {loss:.4f}   (0.25 is what guessing scores)")
    print(f"  hidden weights before  [{', '.join(before)}]")
    print(f"  hidden weights after   [{', '.join(after)}]")
    for x, t in XOR:
        y = net.predict(x)
        print(f"    x={x} want={t}  got={y:.3f} -> {round(y)}  "
              f"{'ok ' if round(y) == t else 'WRONG'}")
    print(f"  {'XOR SOLVED' if net.solves(XOR) else 'not solved'}")
    return net.solves(XOR)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "handwired":
        net = Network(output_delta_broken, hidden_delta_broken,
                      updated_weight_broken, n_hidden=2).hand_wire()
        print("weights set by hand, no training at all:")
        for x, t in XOR:
            y = net.predict(x)
            print(f"  x={x} want={t}  got={y:.3f} -> {round(y)}  "
                  f"{'ok ' if round(y) == t else 'WRONG'}")
        print(f"  {'XOR SOLVED' if net.solves(XOR) else 'not solved'}")
        sys.exit(0)

    results = []

    results.append(table(
        "output_delta", output_delta_broken, output_delta, ["y", "target"],
        [
            ((0.5, 1), -0.12500, "unsure and should have fired"),
            ((0.5, 0), +0.12500, "unsure and should not have"),
            ((0.9, 1), -0.00900, "nearly right already, small move"),
            ((0.1, 0), +0.00900, "nearly right the other way"),
            ((0.99, 0), +0.009801, "badly wrong but saturated -- barely moves"),
            ((0.5, 0.5), 0.00000, "no error at all"),
        ],
        "y is what the output unit produced"))

    results.append(table(
        "hidden_delta", hidden_delta_broken, hidden_delta,
        ["h", "w_out", "out_delta"],
        [
            ((0.5, 1.0, 0.2), +0.05000, "ordinary case"),
            ((0.5, 0.0, 0.2), +0.00000, "not connected -- not its fault"),
            ((0.5, -1.0, 0.2), -0.05000, "negative connection, opposite blame"),
            ((0.5, 2.0, 0.2), +0.10000, "twice connected, twice the blame"),
            ((0.99, 1.0, 0.2), +0.00198, "saturated -- barely moves"),
            ((0.5, 1.0, 0.0), +0.00000, "nothing went wrong downstream"),
        ],
        "this unit produced h and sent it along w_out to a unit whose delta is out_delta"))

    results.append(table(
        "updated_weight", updated_weight_broken, updated_weight,
        ["w", "delta", "incoming", "rate"],
        [
            ((0.0, 0.5, 1.0, 0.5), -0.25000, "move against the delta"),
            ((0.0, 0.5, 0.0, 0.5), +0.00000, "nothing arrived, nothing moves"),
            ((0.4, -0.2, 1.0, 0.5), +0.50000, "negative delta pushes the other way"),
            ((0.4, 0.5, 2.0, 0.5), -0.10000, "twice the input, twice the move"),
            ((0.4, 0.0, 1.0, 0.5), +0.40000, "no delta, no move"),
        ],
        "delta belongs to the unit this weight feeds into"))

    total = sum(n for _, _, n in results)
    ob = sum(o for o, _, _ in results)
    done = all(m is not None for _, m, _ in results)
    print(f"\n{'=' * 74}")
    print(f"ORIGINAL: {ob}/{total} rows correct, {total - ob} failing")
    print(f"YOURS:    {sum(m for _, m, _ in results)}/{total} rows correct" if done
          else "YOURS:    not written yet")

    print(f"\n{'=' * 74}\nwired into a real network, trained on XOR\n{'=' * 74}")
    run_xor("with the originals:", output_delta_broken, hidden_delta_broken,
            updated_weight_broken)
    if done:
        if run_xor("with yours:", output_delta, hidden_delta, updated_weight):
            print("\n  Two layers, and the thing one neuron could never do is done.")
    else:
        print("\nwith yours: not written yet")
