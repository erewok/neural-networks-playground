"""
Stage 8: activation functions, and what a stack of them does to a gradient.

06 ended by composing two linear layers into one linear layer, using your own
forward pass to build the single W and b that reproduced every output. Stacking
linear layers adds parameters and adds nothing else. Something has to go
between them, and it has to be something a straight line cannot imitate.

Sigmoid has been doing that job since 03. This file is about what else can do
it, and about the cost sigmoid was charging the whole time.

THE FOUR FUNCTIONS

    relu(z)         0 below zero, z above it
    relu_grad(z)    its slope
    tanh_grad(a)    the slope of tanh, given an activation tanh produced
    gelu(z)         a smooth thing that behaves like relu far from zero

sigmoid, sigmoid_grad and math.tanh are already here and working. tanh_grad is
the one piece of that pair missing.

SLOPES ARE THE POINT

03 established that sensitivities along a path multiply. A stack of d layers
puts d activation slopes into that product, one per layer, alongside the
weights. So the ceiling on an activation's slope is a ceiling on what a deep
stack can pass backwards.

The first section measures sigmoid's and tanh's slopes by nudging, with no
formula involved, and prints the largest value each one reaches. The two
ceilings differ by a factor of four. Four to the twenty-fourth power is about
3e14, and the last section is where that number shows up.

GELU, WITHOUT A FORMULA

relu asks one question of z: is it positive? Keep z if so, drop it if not. A
hard gate.

The third section replaces that test with a coin. Instead of asking whether z
is positive, it draws a standard normal sample and asks whether z is above the
sample, then averages the outcome over many draws. The average is smooth in z,
it dips below zero for moderately negative z, and it approaches z for large z.
That average is gelu, and the section prints it before you write it.

The proportion of standard normal draws below z is that distribution's CDF.
math.erf is in the standard library, and the CDF and erf differ by a shift and
a scale.

PROPERTIES

  1. relu has no ceiling on the positive side, so a relu unit cannot saturate
     the way sigmoid and tanh do.
  2. relu_grad returns one of two numbers with nothing in between. At exactly
     0 the slope is undefined; return 0, which is the convention every
     framework uses.
  3. tanh_grad takes an ACTIVATION, a number tanh produced, so -1 < a < 1.
     Same convention as 03's sigmoid_slope. Passing it a z is meaningless.
  4. tanh and sigmoid are both steepest at 0 and flatten towards both ends.
     Their maximum slopes differ by a factor of 4.
  5. gelu(0) is 0, gelu is smooth everywhere, and gelu is not monotonic: it
     goes below zero for moderately negative inputs and comes back up. relu
     does neither.
  6. gelu and relu converge as z grows. By z = 6 they differ by about 6e-9,
     and past z = 10 they are the same float.

LAST SECTION

Stacks of width 16 and depth 24, with a gradient pushed back through them and
its norm recorded at every depth. Four activations, at two weight scales,
because the activation and the weight scale interact and neither settles the
question alone. 10 is about the scales.

`just plot 8` draws the four functions with their slopes, and the depth curves.
"""

import math
import random
import sys

from harness import (CROSS, TICK, Cell, ExerciseTable, approx, console,
                     fmt_float, not_written, section, summary)


# ---------------------------------------------------------- already working
# From 03, unchanged. Nothing in this block needs fixing.


def sigmoid(z):
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


def sigmoid_grad(a):
    """Slope of sigmoid where it produced the activation `a`, so 0 < a < 1."""
    return a * (1.0 - a)


def measure(f, x, h=1e-5):
    """The slope of a one-argument function at x, by nudging. No calculus."""
    return (f(x + h) - f(x - h)) / (2 * h)


# --------------------------------------------------------------- the originals
# Left here on purpose. Don't edit these -- the harness runs them beside yours.


def relu_broken(z):
    return abs(z)


def relu_grad_broken(z):
    return max(0.0, z)


def tanh_grad_broken(a):
    return a * (1.0 - a)


def gelu_broken(z):
    return max(0.0, z)


# ---------------------------------------------------------------------- yours


def relu(z):
    """0 below zero, z above it. One number in, one number out."""
    raise NotImplementedError


def relu_grad(z):
    """The slope of relu at z. Takes the input, not the activation."""
    raise NotImplementedError


def tanh_grad(a):
    """The slope of tanh where it produced the activation `a`, so -1 < a < 1."""
    raise NotImplementedError


def gelu(z):
    """The soft gate the third section below measures."""
    raise NotImplementedError


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
    try:
        return Cell(fn(*args))
    except Exception as exc:
        return Cell(None, detail=type(exc).__name__)


def compare(title, original, mine, argname, cases, note="", places=4):
    """cases: list of (arg, expected)."""
    written = is_written(mine, (cases[0][0],))
    t = ExerciseTable(
        title=title, note=note, input_header="argument",
        yours_written=written, fmt=fmt_float(places), compare=approx,
    )
    for arg, want in cases:
        t.add(f"{argname}={arg:g}", want, attempt(original, (arg,)),
              attempt(mine, (arg,)) if written else None)
    return t.render()


def claim(text, holds):
    console.print(f"  [{'bold green' if holds else 'bold red'}]"
                  f"{TICK if holds else CROSS}[/] {text}")
    return holds


# ------------------------------------------------ how steep each one ever gets


SWEEP = [-6.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 6.0]


def steepest(f, lo=-30.0, hi=30.0, n=20001):
    """The largest measured slope of f over a fine sweep, and where it is."""
    best, at = 0.0, lo
    for k in range(n):
        z = lo + (hi - lo) * k / (n - 1)
        s = measure(f, z)
        if s > best:
            best, at = s, z
    return best, at


def run_slope_measurement():
    console.print("[dim]Slopes measured by nudging, the same way 03 and 07 "
                  "did it. Nothing here uses a derivative.[/dim]")
    console.print()
    console.print("       z     sigmoid(z)     slope         tanh(z)     slope")
    for z in SWEEP:
        console.print(f"  {z:>6.1f}     {sigmoid(z):>9.6f}  {measure(sigmoid, z):>8.5f}"
                      f"     {math.tanh(z):>9.6f}  {measure(math.tanh, z):>8.5f}")

    s_best, s_at = steepest(sigmoid)
    t_best, t_at = steepest(math.tanh)
    console.print()
    console.print(f"  [bold]steepest sigmoid[/bold]  {s_best:.5f}  at z = {s_at:+.3f}")
    console.print(f"  [bold]steepest tanh   [/bold]  {t_best:.5f}  at z = {t_at:+.3f}")
    console.print(f"  [dim]ratio {t_best / s_best:.3f}[/dim]")
    console.print()
    console.print(f"  [dim]Through 24 layers those ceilings differ by "
                  f"{(t_best / s_best) ** 24:.3g}, before any weight is "
                  f"involved.[/dim]")


# ------------------------------------------------------- a hard gate, made soft


def hard_gate(z):
    """relu, described as a decision: keep z when z is positive."""
    return z if z > 0.0 else 0.0


def soft_gate(z, draws, rng):
    """Keep z when a standard normal draw comes in below z. Average the result."""
    kept = sum(1 for _ in range(draws) if rng.gauss(0.0, 1.0) < z)
    return z * kept / draws


def run_gate_simulation(draws=200000):
    console.print("[dim]relu asks whether z is positive. This asks whether z "
                  "beats a standard normal draw,\nand averages the answer over "
                  f"{draws:,} of them.[/dim]")
    console.print()
    console.print("       z    hard gate    soft gate  [dim](simulated)[/dim]")
    rng = random.Random(8)
    for z in [-3.0, -2.0, -1.0, -0.75, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0]:
        console.print(f"  {z:>6.2f}   {hard_gate(z):>9.4f}   "
                      f"{soft_gate(z, draws, rng):>10.4f}")
    console.print()
    console.print("  [dim]The soft column has no corner at 0, dips below zero "
                  "on the way in, and\n  catches the hard column up by z = 5. "
                  "It is a simulation, so the last\n  digits move between "
                  "runs.[/dim]")


# ------------------------------------------------- gradient magnitude by depth


WIDTH = 16
DEPTH = 24
SHOWN_DEPTHS = (1, 2, 4, 8, 16, 24)


def gradient_profile(act, slope, width, depth, scale, rng):
    """Push a unit gradient back through `depth` layers, norm at every step.

    Two nested loops per layer, forwards and backwards. This is 05's matvec and
    06's grad_x written out longhand, so that this file stands alone.
    """
    h = _unit([rng.gauss(0.0, 1.0) for _ in range(width)])
    Ws, Zs, Hs = [], [], []
    for _ in range(depth):
        W = [[rng.gauss(0.0, scale) for _ in range(width)]
             for _ in range(width)]
        z = [sum(W[i][j] * h[j] for j in range(width)) for i in range(width)]
        h = [act(v) for v in z]
        Ws.append(W)
        Zs.append(z)
        Hs.append(h)

    g = _unit([rng.gauss(0.0, 1.0) for _ in range(width)])
    norms = []
    for k in range(depth - 1, -1, -1):
        d = [g[i] * slope(Zs[k][i], Hs[k][i]) for i in range(width)]
        g = [sum(Ws[k][i][j] * d[i] for i in range(width)) for j in range(width)]
        norms.append(math.sqrt(sum(v * v for v in g)))
    return norms


def _unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def _median(values):
    s = sorted(values)
    mid = len(s) // 2
    return s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2.0


def activation_pairs():
    """(name, value function, slope function) for each activation.

    The slope functions take both z and the activation, since relu's rule reads
    z and tanh's reads the activation. gelu's is measured from yours.
    """
    return [
        ("sigmoid", sigmoid, lambda z, a: sigmoid_grad(a)),
        ("tanh", math.tanh, lambda z, a: tanh_grad(a)),
        ("relu", relu, lambda z, a: relu_grad(z)),
        ("gelu", gelu, lambda z, a: measure(gelu, z)),
    ]


SCALES = [("xavier, w ~ N(0, 1/sqrt(16))", 1.0 / math.sqrt(WIDTH)),
          ("he,     w ~ N(0, sqrt(2/16))", math.sqrt(2.0 / WIDTH))]


_PROFILES = {}


def depth_profiles(trials=8):
    """{scale label: {activation name: [median norm at each depth]}}

    Cached, so `just run 8 plot` builds the stacks once rather than twice.
    """
    if _PROFILES:
        return _PROFILES
    out = _PROFILES
    for label, scale in SCALES:
        rows = {}
        for name, act, slope in activation_pairs():
            rng = random.Random(8)
            runs = [gradient_profile(act, slope, WIDTH, DEPTH, scale, rng)
                    for _ in range(trials)]
            rows[name] = [_median([r[k] for r in runs]) for k in range(DEPTH)]
        out[label] = rows
    return out


def run_depth(profiles):
    for label, rows in profiles.items():
        console.print()
        console.print(f"  [bold]{label}[/bold]")
        header = "".join(f"{d:>10}" for d in SHOWN_DEPTHS)
        console.print(f"    [cyan]{'depth':<9}[/cyan]{header}")
        for name, norms in rows.items():
            cells = "".join(f"{norms[d - 1]:>10.2e}" for d in SHOWN_DEPTHS)
            console.print(f"    {name:<9}{cells}")

    xavier = profiles[SCALES[0][0]]
    he = profiles[SCALES[1][0]]
    console.print()
    good = claim(
        f"at depth {DEPTH} and the xavier scale, the sigmoid stack passed back "
        f"{xavier['tanh'][DEPTH - 1] / xavier['sigmoid'][DEPTH - 1]:.2g} times "
        f"less than the tanh stack",
        xavier["sigmoid"][DEPTH - 1] < xavier["tanh"][DEPTH - 1] / 1e10)
    good &= claim(
        "sigmoid was the worst of the four at the deepest point, at both scales",
        all(min(rows, key=lambda n: rows[n][DEPTH - 1]) == "sigmoid"
            for rows in profiles.values()))
    good &= claim(
        f"relu went from {xavier['relu'][DEPTH - 1]:.2g} to "
        f"{he['relu'][DEPTH - 1]:.2g} when only the weight scale changed",
        he["relu"][DEPTH - 1] > xavier["relu"][DEPTH - 1] * 100)
    console.print("  [dim]The activation does not settle this on its own. 10 "
                  "is about the scales.[/dim]")
    return good


# --------------------------------------------------------------------- claims


def run_claims():
    console.print()
    steps = [-30.0 + 60.0 * k / 20000 for k in range(20001)]
    good = claim("relu_grad returned nothing but 0 and 1 over 20001 points "
                 "from -30 to 30",
                 set(relu_grad(z) for z in steps) <= {0.0, 1.0})
    good &= claim("tanh_grad's largest value was 1, reached at a = 0",
                  approx(tanh_grad(0.0), 1.0) and
                  all(tanh_grad(a) <= 1.0 + 1e-12
                      for a in [-0.999 + 0.001 * k for k in range(1999)]))
    good &= claim("gelu and relu agreed to within 1e-8 for every z above 6",
                  all(abs(gelu(z) - relu(z)) < 1e-8
                      for z in [6.0 + 0.1 * k for k in range(200)]))

    dip = min(((gelu(-4.0 + 0.001 * k), -4.0 + 0.001 * k) for k in range(4001)))
    good &= claim(f"gelu's lowest value was {dip[0]:.4f}, at z = {dip[1]:+.3f}",
                  dip[0] < -0.16 and -1.0 < dip[1] < -0.5)

    rng = random.Random(80)
    worst = max(abs(measure(relu, z) - relu_grad(z))
                for z in [rng.uniform(-8, 8) for _ in range(2000)])
    good &= claim(f"relu_grad matched relu's measured slope everywhere off "
                  f"zero, worst gap {worst:.2g}", worst < 1e-6)
    return good


# ---------------------------------------------------------------------- plots


def plot():
    import plots

    if not everything_written():
        console.print("\n[yellow]The figures need all four functions written "
                      "first.[/yellow]")
        return
    plots.activation_figure(
        [("sigmoid", sigmoid, lambda z: sigmoid_grad(sigmoid(z))),
         ("tanh", math.tanh, lambda z: tanh_grad(math.tanh(z))),
         ("relu", relu, relu_grad),
         ("gelu", gelu, lambda z: measure(gelu, z))])
    plots.depth_figure(depth_profiles(), SHOWN_DEPTHS)
    plots.done("08")


ALL = [
    (relu, (1.0,)),
    (relu_grad, (1.0,)),
    (tanh_grad, (0.5,)),
    (gelu, (1.0,)),
]


def everything_written():
    return all(is_written(fn, probe) for fn, probe in ALL)


if __name__ == "__main__":
    section("how steep sigmoid and tanh ever get")
    run_slope_measurement()

    section("relu, and its slope")
    results = []

    results.append(compare(
        "relu(z)", relu_broken, relu, "z",
        [
            (-3.0, 0.0),
            (-0.5, 0.0),
            (0.0, 0.0),
            (0.5, 0.5),
            (3.0, 3.0),
            (12.5, 12.5),
        ],
        "no ceiling above zero, flat below it. the original agrees on "
        "everything from 0 up"))

    results.append(compare(
        "relu_grad(z)", relu_grad_broken, relu_grad, "z",
        [
            (-3.0, 0.0),
            (-0.5, 0.0),
            (0.0, 0.0),
            (0.001, 1.0),
            (3.0, 1.0),
            (100.0, 1.0),
        ],
        "two values and nothing between them. the original returns relu's "
        "value rather than its slope, which is the same number only at z = 1"))

    section("tanh's slope")
    results.append(compare(
        "tanh_grad(a)", tanh_grad_broken, tanh_grad, "a",
        [
            (0.0, 1.0),
            (0.5, 0.75),
            (-0.5, 0.75),
            (0.9, 0.18999999999999995),
            (0.99, 0.01990000000000003),
            (0.7615941559557649, 0.41997434161402614),
        ],
        "a is an ACTIVATION, so -1 < a < 1. the original applies sigmoid's "
        "rule, which is a different rule. rows 2 and 3 are symmetric"))

    section("a hard gate, made soft")
    run_gate_simulation()

    section("gelu")
    results.append(compare(
        "gelu(z)", gelu_broken, gelu, "z",
        [
            (0.0, 0.0),
            (1.0, 0.8413447460685429),
            (-1.0, -0.15865525393145707),
            (-0.75, -0.16997051428265114),
            (2.0, 1.9544997361036416),
            (-2.0, -0.04550026389635842),
            (3.0, 2.99595030590511),
            (5.0, 4.999998566742141),
        ],
        "the numbers the simulation printed, without the simulation. the "
        "negative rows are where relu and gelu part company", places=6))

    summary(results)

    if everything_written():
        section("what a stack of them does to a gradient")
        run_depth(depth_profiles())

        section("five things checked rather than asserted")
        run_claims()
    else:
        not_written("the depth stacks and the checks")

    if "plot" in sys.argv:
        plot()
