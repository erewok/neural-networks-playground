"""Plots. Like harness.py, there is nothing to fix in here -- it only draws.

    just plot 1      open the figures for exercise 1
    just run 1 plot  same thing

Set NN_PLOT_SAVE=<dir> to write PNGs instead of opening windows, which is what
you want over ssh or in a script.

COLOURS
  One rule: colour means *class*, never magnitude, and never decoration. Blue is
  "the neuron fires", orange is "it stays quiet", everywhere in every figure. The
  two hues are checked for colourblind separation, and every series is labelled
  as well as coloured so nothing depends on colour alone.
"""

import os

import matplotlib

_SAVE_DIR = os.environ.get("NN_PLOT_SAVE")
if _SAVE_DIR:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import seaborn as sns                    # noqa: E402

# Validated light-surface palette. Slots 1-3 clear all-pairs CVD separation.
FIRES = "#2a78d6"      # blue   -- output 1
QUIET = "#eb6834"      # orange -- output 0
ACCENT = "#1baf7a"     # aqua   -- a third thing, always directly labelled
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#dedcd7"

# The same orange, named for a second job: the one thing that MOVES in a
# figure whose background is a sequential ramp. No class meaning there.
TRAIL = QUIET

# Sequential blue, light -> dark, for "how far into the run is this". Discrete
# ordered marks, so the lightest step is 250 -- anything lighter recedes into
# the paper. Used by exercise 2 for updates and exercise 4 for epochs.
EPOCH_RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
              "#1c5cab", "#0d366b"]


def _signed_ramp():
    """Orange -> paper -> blue, for a quantity that can be either side of zero.

    This does not break the colour rule at the top of the file. The hue still
    means the class it always means -- orange is the quiet side, blue is the
    firing side -- and only the LIGHTNESS carries magnitude. Paper-white is
    exactly zero, which is where the decision boundary sits.
    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "signed", [QUIET, "#f6d3c2", SURFACE, "#bcd6f2", FIRES])


def boundary_points(weights, bias, lo, hi):
    """The fence as two endpoints, or None when there is no line to draw.

    plots only. Exercise 1 asks you to write this as a function of x1; here it
    also has to cope with a vertical fence and with a neuron whose weights are
    still all zero, neither of which the exercise tests.
    """
    w1, w2 = weights[0], weights[1]
    if abs(w2) > 1e-12:
        xs = np.array([lo, hi])
        return xs, -(w1 * xs + bias) / w2
    if abs(w1) > 1e-12:
        x = -bias / w1
        return np.array([x, x]), np.array([lo, hi])
    return None

_FIGS = []


def theme():
    sns.set_theme(
        style="white",
        rc={
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SOFT,
            "text.color": INK,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "grid.color": GRID,
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        },
    )


def figure(*args, **kwargs):
    fig = plt.figure(*args, **kwargs)
    _FIGS.append(fig)
    return fig


def title(ax, text, sub=""):
    """Bold title above a dim subtitle, both left-aligned to the axes."""
    ax.set_title(text, loc="left", fontsize=11, color=INK,
                 pad=24 if sub else 8, fontweight="bold")
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9,
                color=INK_SOFT, va="bottom", ha="left")


def done(name="figure"):
    """Open the windows, or write PNGs if NN_PLOT_SAVE is set."""
    if _SAVE_DIR:
        os.makedirs(_SAVE_DIR, exist_ok=True)
        for i, fig in enumerate(_FIGS, 1):
            path = os.path.join(_SAVE_DIR, f"{name}-{i}.png")
            fig.savefig(path, dpi=110, bbox_inches="tight", facecolor=SURFACE)
            print(f"wrote {path}")
    else:
        plt.show()
    _FIGS.clear()


# ------------------------------------------------------------------ exercise 1


CORNERS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def _corners(ax, cases=None, size=150):
    """The four binary inputs. Coloured by the answer wanted, when there is one."""
    for x1, x2 in CORNERS:
        want = dict(cases).get((x1, x2)) if cases else None
        ax.scatter([x1], [x2], s=size, zorder=5,
                   color=(FIRES if want else QUIET) if cases else SURFACE,
                   edgecolor=SURFACE if cases else INK_SOFT, linewidth=2)
        if cases:
            ax.annotate(str(want), (x1, x2), color=SURFACE, fontsize=9,
                        fontweight="bold", ha="center", va="center", zorder=6)


def _square(ax, lo, hi):
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")


def neuron_surface_figure(name, weights, bias, cases, ws_fn, step_fn):
    """The one thing the old version of this figure threw away: the z axis.

    A neuron is a tilted plane over the (x1, x2) floor, and step() slices it at
    zero. Draw the plane, draw a cut through it, then draw what is left after
    the slice -- in that order, because the last panel on its own is what made
    the old figure look like an arbitrary line on a shaded background.
    """
    theme()
    lo, hi = -0.6, 1.6
    fig = figure(figsize=(14.2, 4.9))

    gx, gy = np.meshgrid(np.linspace(lo, hi, 240), np.linspace(lo, hi, 240))
    zz = np.array([[ws_fn((a, b), weights, bias) for a, b in zip(rx, ry)]
                   for rx, ry in zip(gx, gy)])
    span = float(np.abs(zz).max())

    # -- panel 1: the weighted sum itself, before any decision is taken -----
    ax = fig.add_subplot(1, 3, 1)
    im = ax.contourf(gx, gy, zz, levels=np.linspace(-span, span, 25),
                     cmap=_signed_ramp(), vmin=-span, vmax=span)
    ax.contour(gx, gy, zz, levels=[0.0], colors=[INK], linewidths=2.2)
    bar = fig.colorbar(im, ax=ax, pad=0.02, ticks=[-span, 0, span])
    bar.set_label("the weighted sum, z", color=INK_SOFT, fontsize=9)
    bar.ax.set_yticklabels([f"{-span:.1f}", "0", f"{span:.1f}"], fontsize=8)
    _corners(ax, cases)
    _square(ax, lo, hi)
    title(ax, "1. What the neuron computes",
          "a tilted plane. white is exactly zero")

    # -- panel 2: a cut straight across it ---------------------------------
    # Walk along the weight vector: that is the direction the plane climbs
    # fastest, so the cut is as steep as this neuron ever gets.
    ax2 = fig.add_subplot(1, 3, 2)
    w1, w2 = weights[0], weights[1]
    norm = (w1 ** 2 + w2 ** 2) ** 0.5 or 1.0
    ux, uy = w1 / norm, w2 / norm
    ts = np.linspace(-1.4, 1.4, 400)
    cx, cy = 0.5, 0.5
    zs = [ws_fn((cx + t * ux, cy + t * uy), weights, bias) for t in ts]
    outs = [step_fn(z) for z in zs]

    ax2.axhline(0, color=GRID, lw=1, zorder=0)
    ax2.plot(ts, zs, color=INK_SOFT, lw=2.2, label="z, the weighted sum")
    ax2.plot(ts, outs, color=FIRES, lw=2.6, label="step(z), what comes out")
    ax2.fill_between(ts, -span, 0, where=[z <= 0 for z in zs], color=QUIET,
                     alpha=0.10)
    crossing = ts[int(np.argmin(np.abs(zs)))]
    ax2.axvline(crossing, color=INK, lw=2.2, zorder=3)
    ax2.annotate("the fence:\nz = 0", xy=(crossing, -span * 0.75),
                 xytext=(10, 0), textcoords="offset points", fontsize=9,
                 color=INK, fontweight="bold")
    ax2.set_xlabel("distance walked across the plane")
    ax2.set_ylabel("z,  and step(z)")
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    title(ax2, "2. Cut across it",
          "z ramps smoothly; step keeps only which side of zero it reached")

    # -- panel 3: what survives the cut ------------------------------------
    ax3 = fig.add_subplot(1, 3, 3)
    oo = np.array([[step_fn(v) for v in row] for row in zz], dtype=float)
    ax3.contourf(gx, gy, oo, levels=[-0.5, 0.5, 1.5], colors=[QUIET, FIRES],
                 alpha=0.16)
    ax3.contour(gx, gy, zz, levels=[0.0], colors=[INK], linewidths=2.2)
    _corners(ax3, cases)
    _square(ax3, lo, hi)
    title(ax3, "3. What is left afterwards",
          "above the line becomes 1, below becomes 0")

    fig.suptitle(f"A neuron is a tilted plane, cut at zero    ({name}: "
                 f"weights={list(weights)}, bias={bias})",
                 x=0.02, y=0.985, ha="left", fontsize=13, fontweight="bold",
                 color=INK)
    fig.text(0.02, 0.905,
             "Both axes are INPUTS. The output is not an axis here -- it is "
             "the colour. That is why the black line is not a line of best "
             "fit: nothing is being fitted, it is just where z crosses zero.",
             fontsize=9.5, color=INK_SOFT, ha="left")
    fig.tight_layout(rect=[0, 0, 0.995, 0.87])
    return fig


XOR_CASES = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]

# The closest a single straight fence gets to XOR: OR's neuron, right about
# three corners out of four. Hardcoded so both XOR panels draw with none of
# the exercise's functions written.
_XOR_TRY = ((1.0, 1.0), -0.5)


def _regions(ax, gx, gy, fired):
    """Paint the two half-planes. The fence is wherever they meet."""
    ax.contourf(gx, gy, fired, levels=[-0.5, 0.5, 1.5],
                colors=[QUIET, FIRES], alpha=0.16)


def _gate_panel(ax, label, weights, bias, cases, note, boundary_fn, ws_fn,
                step_fn, lo, hi):
    """One gate: the two regions, the fence between them, the four corners.

    The regions come from the neuron itself, the fence comes from your
    boundary function, and they are drawn on top of each other on purpose --
    if your fence is not where the colours change, that is the bug, visible.
    """
    gx, gy = np.meshgrid(np.linspace(lo, hi, 200), np.linspace(lo, hi, 200))
    fired = np.array([[step_fn(ws_fn((a, b), weights, bias))
                       for a, b in zip(rx, ry)]
                      for rx, ry in zip(gx, gy)], dtype=float)
    _regions(ax, gx, gy, fired)

    xs = np.array([lo, hi])
    ax.plot(xs, [boundary_fn(x, weights, bias) for x in xs],
            color=INK, lw=2.2, zorder=4)

    _corners(ax, cases)
    _square(ax, lo, hi)
    title(ax, label, f"w={list(weights)}  b={bias:+g}   {note}")


def _xor_try_panel(ax, lo, hi):
    """The best a straight fence manages on XOR, and the corner it loses."""
    (w1, w2), b = _XOR_TRY
    gx, gy = np.meshgrid(np.linspace(lo, hi, 200), np.linspace(lo, hi, 200))
    _regions(ax, gx, gy, np.where(w1 * gx + w2 * gy + b > 0, 1.0, 0.0))

    xs = np.array([lo, hi])
    ax.plot(xs, -(w1 * xs + b) / w2, color=INK, lw=2.2, zorder=4)

    for (x1, x2), want in XOR_CASES:
        if (w1 * x1 + w2 * x2 + b > 0) == bool(want):
            continue
        ax.scatter([x1], [x2], s=560, facecolors="none", edgecolor=INK,
                   linewidth=2.4, zorder=7)
        ax.annotate(f"wants {want}, sits in\nthe other half",
                    (x1, x2), xytext=(-10, -40), textcoords="offset points",
                    ha="right", fontsize=9, color=INK, fontweight="bold")

    _corners(ax, XOR_CASES)
    _square(ax, lo, hi)
    title(ax, "XOR, best attempt",
          "three corners land right, and three is the ceiling")


def _xor_why_panel(ax, lo, hi):
    """Why nothing does better: the two pairs cross.

    No shading here, because the claim is not about one particular fence --
    it is about all of them at once.
    """
    ax.plot([0, 1], [1, 0], color=FIRES, lw=2.4, ls=(0, (5, 3)), zorder=3)
    ax.plot([0, 1], [0, 1], color=QUIET, lw=2.4, ls=(0, (5, 3)), zorder=3)
    ax.scatter([0.5], [0.5], s=90, color=INK, zorder=6)
    ax.annotate("both segments\nwant this point", (0.5, 0.5),
                xytext=(16, 12), textcoords="offset points", fontsize=9,
                color=INK, fontweight="bold")

    _corners(ax, XOR_CASES)
    _square(ax, lo, hi)
    title(ax, "XOR, why never",
          "a half-plane holding both ends holds everything between")


_HOW_TO_READ = ("Each dot is coloured by the answer WANTED there; the half it "
                "sits in is the answer the neuron GIVES. A corner is right "
                "when the two colours match, which is why the ringed one "
                "is not.")

_XOR_WHY = ("A fence that keeps both blue corners in the firing half keeps "
            "the whole blue segment there, crossing point included -- and "
            "the same is true of the orange pair. One point, two halves, "
            "no line.")


def line_limit_figure(families, boundary_fn, ws_fn, step_fn):
    """What one neuron can do, and the one thing it cannot, on shared axes.

    Separate panels rather than the old single overlay: the claim being made
    is about WHICH CORNERS end up in which region, and regions cannot be
    stacked on one set of axes the way bare lines can. The gates keep their
    neighbours, though, so the comparisons still cost one glance -- AND and
    OR sit side by side with identical weights.

    families: (label, weights, bias, cases, note), four of them, filling the
    top row and the first cell of the bottom one.
    """
    theme()
    lo, hi = -0.6, 1.6
    fig = figure(figsize=(12.8, 8.8))

    for i, (label, weights, bias, cases, note) in enumerate(families, 1):
        _gate_panel(fig.add_subplot(2, 3, i), label, weights, bias, cases,
                    note, boundary_fn, ws_fn, step_fn, lo, hi)

    _xor_try_panel(fig.add_subplot(2, 3, len(families) + 1), lo, hi)
    _xor_why_panel(fig.add_subplot(2, 3, len(families) + 2), lo, hi)

    fig.suptitle("One neuron is one straight fence    (blue half fires, "
                 "orange half stays quiet)",
                 x=0.02, y=0.985, ha="left", fontsize=13, fontweight="bold",
                 color=INK)
    fig.text(0.02, 0.945, _HOW_TO_READ + "  " + _XOR_WHY,
             fontsize=9.5, color=INK_SOFT, ha="left", va="top", wrap=True)
    fig.tight_layout(rect=[0, 0, 0.995, 0.90])
    return fig


def xor_figure():
    """The XOR half of the figure above, alone. Needs none of your functions.

    This is what gets drawn while the exercise is still unwritten: the limit
    does not depend on your code, so it is available from the first minute.
    """
    theme()
    lo, hi = -0.6, 1.6
    fig = figure(figsize=(9.4, 5.0))

    _xor_try_panel(fig.add_subplot(1, 2, 1), lo, hi)
    _xor_why_panel(fig.add_subplot(1, 2, 2), lo, hi)

    fig.suptitle("XOR: the gate one neuron never gets",
                 x=0.02, y=0.985, ha="left", fontsize=13, fontweight="bold",
                 color=INK)
    fig.text(0.02, 0.94, _HOW_TO_READ + "  " + _XOR_WHY
             + "  Which is why networks have more than one layer: two layers "
             "bend the space first, then draw the fence in the bent space.",
             fontsize=9.5, color=INK_SOFT, ha="left", va="top", wrap=True)
    fig.tight_layout(rect=[0, 0, 0.995, 0.80])
    return fig


# ------------------------------------------------------------------ exercise 2


def _trail_panel(ax, name, examples, history, converged):
    """One task: the fence swinging into place, one update at a time."""
    lo, hi = -0.6, 1.6

    # A run can take a hundred updates. Show a handful, spread over the whole
    # run rather than bunched at the start where the big corrections happen.
    n = len(history)
    picks = sorted({int(round(i * (n - 1) / (len(EPOCH_RAMP) - 1)))
                    for i in range(len(EPOCH_RAMP))})

    for colour, i in zip(EPOCH_RAMP, picks):
        pts = boundary_points(history[i][0], history[i][1], lo, hi)
        if pts is None:      # weights still all zero: no fence exists yet
            continue
        last = i == picks[-1]
        # Early fences fade back so the run reads as one motion rather than a
        # bundle of equally loud lines. The last one is the only black one.
        ax.plot(pts[0], pts[1], color=INK if last else colour,
                lw=2.6 if last else 1.4, alpha=1.0 if last else 0.6,
                zorder=4 if last else 2)

    _corners(ax, examples)
    _square(ax, lo, hi)
    title(ax, name, f"{n - 1} updates, pale to dark"
          + ("" if converged else "  --  and still moving"))


def perceptron_trail_figure(runs):
    """runs: list of (name, examples, history, mistakes, converged).

    The learning rule in 02 is a MOTION, and a table of final weights is the
    one thing that cannot show a motion. Each update leaves a line behind, so
    the whole run is visible at once: AND and OR walk to a fence and stop, XOR
    never stops, because there is no fence for it to stop at.
    """
    theme()
    cols = len(runs)
    fig = figure(figsize=(4.6 * cols, 8.6))

    for i, (name, examples, history, mistakes, converged) in enumerate(runs):
        ax = fig.add_subplot(2, cols, i + 1)
        _trail_panel(ax, name, examples, history, converged)

        ax2 = fig.add_subplot(2, cols, cols + i + 1)
        epochs = range(1, len(mistakes) + 1)
        ax2.bar(epochs, mistakes, color=FIRES if converged else QUIET,
                width=0.75, zorder=3)
        ax2.set_xlabel("epoch")
        ax2.set_ylabel("mistakes made")
        # Headroom above a full bar, so the note never sits on the data.
        ax2.set_ylim(0, 5.8)
        ax2.set_yticks([0, 1, 2, 3, 4])
        note = (f"epoch {len(mistakes)} made no mistakes at all,\n"
                f"so there was nothing left to correct\nand training stopped"
                if converged else
                f"still wrong about {mistakes[-1]} of the 4 inputs\n"
                f"at epoch {len(mistakes)}, and it would stay\nthat way forever")
        ax2.text(0.97, 0.93, note, transform=ax2.transAxes, ha="right",
                 va="top", fontsize=9, color=INK, linespacing=1.5)
        title(ax2, "Mistakes per epoch",
              "falls to zero, and stays there" if converged
              else "never reaches zero, however long you run it")

    fig.suptitle("Learning is the fence moving",
                 x=0.02, y=0.995, ha="left", fontsize=13, fontweight="bold",
                 color=INK)
    fig.text(0.02, 0.962,
             "Every wrong answer nudges the weights, and every nudge moves "
             "this line. Light lines are early, the black line is where it "
             "finished. Same axes and same colours as exercise 1.",
             fontsize=9.5, color=INK_SOFT, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    return fig


# ------------------------------------------------------------------ exercise 4

def fit_figure(train_pts, history, true_w, true_b):
    """The line moving into place, and the error falling as it goes."""
    theme()
    fig = figure(figsize=(11.5, 4.8))
    xs = [x for x, _ in train_pts]
    ys = [y for _, y in train_pts]
    lo, hi = min(xs) - 0.2, max(xs) + 0.2

    ax = fig.add_subplot(1, 2, 1)
    ax.scatter(xs, ys, s=55, color=FIRES, alpha=0.55, edgecolor=SURFACE,
               linewidth=1.2, zorder=3, label="training points")

    picks = [0, 2, 5, 12, 30, 80, len(history) - 1]
    picks = [i for i in picks if i < len(history)]
    line_xs = np.array([lo, hi])
    for colour, i in zip(EPOCH_RAMP, picks):
        w, b, _ = history[i]
        last = i == picks[-1]
        ax.plot(line_xs, w * line_xs + b, color=colour,
                lw=2.4 if last else 1.6, zorder=4 if last else 2,
                label=f"model after {i} epochs" if last else None)

    # The true line very nearly lands on the fitted one, so it is labelled in
    # the legend rather than annotated on top of it.
    ax.plot(line_xs, true_w * line_xs + true_b, color=ACCENT, lw=2,
            ls=(0, (5, 3)), zorder=5, label="the line the points came from")
    ax.annotate("epoch 0", xy=(hi, history[0][0] * hi + history[0][1]),
                xytext=(-8, 8), textcoords="offset points", ha="right",
                fontsize=9, color=EPOCH_RAMP[0], fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    title(ax, "The line moving into place",
          "each blue line is the model a few epochs later, light to dark")

    ax2 = fig.add_subplot(1, 2, 2)
    losses = [h[2] for h in history]
    ax2.plot(range(len(losses)), losses, color=FIRES, lw=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("mean squared error (log scale)")
    title(ax2, "The error falling",
          "steep while the line is far off, flat once it has arrived")
    fig.tight_layout()
    return fig


def descent_figure(train_pts, history, mean_loss):
    """The loss surface, and the path the knobs actually took across it."""
    theme()
    fig = figure(figsize=(7.6, 6.2))
    ax = fig.add_subplot(1, 1, 1)

    ws = np.linspace(-0.5, 4.5, 90)
    bs = np.linspace(-2.5, 4.5, 90)
    grid = np.array([[mean_loss(train_pts, w, b) for w in ws] for b in bs])

    # Levels bunched towards the low end: linear spacing spends most of its
    # contours on the steep outer walls and leaves the basin a flat blob.
    levels = np.geomspace(max(grid.min(), 1e-3), grid.max(), 16)
    cs = ax.contourf(ws, bs, grid, levels=levels, cmap="Blues_r", alpha=0.9)
    ax.contour(ws, bs, grid, levels=levels, colors=SURFACE, linewidths=0.5,
               alpha=0.55)
    bar = fig.colorbar(cs, ax=ax, pad=0.02)
    bar.set_label("mean squared error", color=INK_SOFT, fontsize=9)
    bar.ax.tick_params(labelsize=8, color=GRID)

    path_w = [h[0] for h in history]
    path_b = [h[1] for h in history]
    ax.plot(path_w, path_b, color=TRAIL, lw=2.2, zorder=4)
    ax.scatter([path_w[0]], [path_b[0]], s=110, color=TRAIL, zorder=5,
               edgecolor=SURFACE, linewidth=2)
    ax.scatter([path_w[-1]], [path_b[-1]], s=170, marker="*", color=TRAIL,
               zorder=5, edgecolor=SURFACE, linewidth=1.5)
    ax.annotate("start: both knobs at 0", (path_w[0], path_b[0]),
                xytext=(0, -26), textcoords="offset points", ha="center",
                fontsize=9.5, color=TRAIL, fontweight="bold")
    ax.annotate("where it stopped", (path_w[-1], path_b[-1]),
                xytext=(14, -20), textcoords="offset points", fontsize=9.5,
                color=TRAIL, fontweight="bold")

    ax.set_xlabel("w  (slope)")
    ax.set_ylabel("b  (intercept)")
    title(ax, "The landscape gradient descent is walking down",
          "every pair of knobs is a point; DARKER means LOWER error, so the dark basin is the goal")
    fig.tight_layout()
    return fig


def inference_figure(train_pts, holdout, w, b, predict_fn):
    """Training used the left-hand points. Inference meets the right-hand ones."""
    theme()
    fig = figure(figsize=(11.5, 5.0))

    ax = fig.add_subplot(1, 2, 1)
    xs = [x for x, _ in train_pts] + [x for x, _ in holdout]
    lo, hi = min(xs) - 0.2, max(xs) + 0.2
    line_xs = np.array([lo, hi])

    ax.scatter([x for x, _ in train_pts], [y for _, y in train_pts], s=55,
               color=FIRES, alpha=0.45, edgecolor=SURFACE, linewidth=1.2,
               zorder=3, label="trained on these")
    ax.plot(line_xs, w * line_xs + b, color=INK, lw=2, zorder=4,
            label="the fitted line")

    for x, y in holdout:
        guess = predict_fn(x, w, b)
        ax.plot([x, x], [y, guess], color=QUIET, lw=1.2, alpha=0.8, zorder=4)
    ax.scatter([x for x, _ in holdout], [y for _, y in holdout], s=95,
               marker="D", color=QUIET, edgecolor=SURFACE, linewidth=1.5,
               zorder=5, label="never seen during training")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    title(ax, "Training vs inference",
          "orange stems are how far off each unseen point was")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    ax2.text(0, 0.97,
             "TRAINING used only the blue points.\n"
             "It looked at each y, measured how wrong\n"
             "the guess was, and moved w and b.\n\n"
             "INFERENCE is the orange diamonds. The\n"
             "model had never seen them. It was handed\n"
             "an x and returned predict(x, w, b) -- one\n"
             "multiply and one add. No target was read,\n"
             "no error computed, nothing moved.\n\n"
             "The stems are drawn only so you can see\n"
             "the miss. The model cannot see them: at\n"
             "inference time there is no y to compare\n"
             "against, which is the entire point of\n"
             "having a model.\n\n"
             f"Everything it learned is two numbers:\n"
             f"   w = {w:.3f}\n"
             f"   b = {b:.3f}",
             va="top", fontsize=10.5, color=INK, linespacing=1.55)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ exercise 5


def _cells(ax, M, hl_row=None, hl_col=None, hl_cell=None, rows_max=None):
    """Draw a matrix as a grid of numbers, tinting the parts under discussion."""
    from matplotlib.patches import Rectangle

    n, m = len(M), len(M[0])
    for i in range(n):
        for j in range(m):
            contributes = (hl_row is not None and i == hl_row) or \
                          (hl_col is not None and j == hl_col)
            answer = hl_cell is not None and (i, j) == hl_cell
            face = ACCENT if answer else (FIRES if contributes else SURFACE)
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=face,
                                   alpha=0.22 if (contributes or answer) else 1.0,
                                   edgecolor=GRID, linewidth=1.2, zorder=1))
            if answer or contributes:
                ax.add_patch(Rectangle((j, i), 1, 1, facecolor="none",
                                       edgecolor=ACCENT if answer else FIRES,
                                       linewidth=2.2, zorder=3))
            ax.text(j + 0.5, i + 0.5, f"{M[i][j]:g}", ha="center", va="center",
                    fontsize=13, color=INK, zorder=4,
                    fontweight="bold" if (answer or contributes) else "normal")
    ax.set_xlim(-0.15, m + 0.15)
    # A shared vertical extent so panels of different heights top-align and
    # their titles sit on one line.
    ax.set_ylim((rows_max or n) + 0.15, -0.15)
    ax.set_aspect("equal")
    ax.axis("off")


def matmul_figure(matmul_fn, transpose_fn):
    """Where ONE number in a matrix product comes from.

    The wrong idea this figure exists to kill is that matmul multiplies numbers
    that sit in the same place. It does not: one number in the answer draws on
    a whole row and a whole column.
    """
    theme()
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[1, 0], [0, 1], [2, -1]]
    C = matmul_fn(A, B)
    i, j = 1, 0                       # the answer cell being explained

    fig = figure(figsize=(13.6, 4.6))
    specs = [
        (1, "A", A, dict(hl_row=i)),
        (2, "@   B", B, dict(hl_col=j)),
        (3, "=   A @ B", C, dict(hl_cell=(i, j))),
    ]
    tallest = max(len(M) for _, _, M, _ in specs)
    for pos, label, M, kw in specs:
        ax = fig.add_subplot(1, 4, pos)
        _cells(ax, M, rows_max=tallest, **kw)
        title(ax, f"{label}", f"{len(M)} x {len(M[0])}")

    terms = " + ".join(f"{A[i][k]}x{B[k][j]}" for k in range(len(B)))
    ax = fig.add_subplot(1, 4, 4)
    ax.axis("off")
    ax.text(0, 0.97,
            "The one green number was made by\n"
            "the whole blue row and the whole\n"
            "blue column, paired off and added:\n\n"
            f"   {terms}  =  {C[i][j]}\n\n"
            "That is a dot product, and every\n"
            "number in the answer is another one.\n\n"
            "NOT elementwise. Nothing here\n"
            "multiplied the two numbers sitting\n"
            "in the same position.\n\n"
            "SHAPES\n"
            f"   ({len(A)} x {len(A[0])}) @ ({len(B)} x {len(B[0])})"
            f"  ->  ({len(C)} x {len(C[0])})\n\n"
            f"   the inner {len(B)}s had to match,\n"
            "   and then they vanished.",
            va="top", fontsize=10.5, color=INK, linespacing=1.55, family="monospace")

    fig.suptitle("One number in a matrix product",
                 x=0.02, y=0.99, ha="left", fontsize=13, fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# ------------------------------------------------------------------ exercise 6


def _heat(ax, M, label, sub, vmax):
    """A grid of numbers, tinted by sign and magnitude.

    Hue still means what it means everywhere else -- blue one side of zero,
    orange the other -- and only lightness carries magnitude, with paper-white
    at exactly zero.
    """
    A = np.asarray(M, dtype=float)
    ax.imshow(A, cmap=_signed_ramp(), vmin=-vmax, vmax=vmax, aspect="auto",
              extent=(0, A.shape[1], A.shape[0], 0), zorder=1)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(j + 0.5, i + 0.5, f"{A[i, j]:+.3f}", ha="center",
                    va="center", fontsize=11, color=INK, zorder=3)
    ax.set_xticks(np.arange(A.shape[1]) + 0.5)
    ax.set_yticks(np.arange(A.shape[0]) + 0.5)
    ax.set_xticklabels([f"unit {j}" for j in range(A.shape[1])], fontsize=8.5)
    ax.set_yticklabels([f"feature {i}" for i in range(A.shape[0])], fontsize=8.5)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    title(ax, label, sub)


def linear_grad_figure(W, measured, analytic):
    """The difference quotient and the closed form, entry by entry."""
    theme()
    vmax = max(abs(v) for M in (measured, analytic) for row in M for v in row)
    vmax = max(vmax, 1e-9)
    worst = max(abs(a - b) for ra, rb in zip(measured, analytic)
                for a, b in zip(ra, rb))

    fig = figure(figsize=(13.2, 5.2))
    ax = fig.add_subplot(1, 3, 1)
    _heat(ax, measured, "measured", "one loss minus another, over h", vmax)
    ax = fig.add_subplot(1, 3, 2)
    _heat(ax, analytic, "your grad_W", "one matmul, no h anywhere", vmax)

    ax = fig.add_subplot(1, 3, 3)
    ax.axis("off")
    ax.text(0, 0.97,
            "Every cell is one slope:\n"
            "add h to that ONE weight,\n"
            "see how much the loss moves,\n"
            "divide by h.\n\n"
            f"grad_W is {len(W)} x {len(W[0])}, the shape\n"
            "of W. One answer per knob,\n"
            "laid out the way the knobs are.\n\n"
            f"largest disagreement between\n"
            f"the two panels:  {worst:.2e}\n\n"
            "The left panel costs one whole\n"
            f"forward pass per cell, {len(W) * len(W[0])} of them.\n"
            "The right panel costs one\n"
            "matmul for the entire grid.\n"
            "That is why nobody trains by\n"
            "difference quotient.",
            va="top", fontsize=9.8, color=INK, linespacing=1.5,
            family="monospace")

    fig.suptitle("The same gradient, measured and derived",
                 x=0.02, y=0.99, ha="left", fontsize=13, fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig


def two_layer_loss_figure(history):
    """Two linear layers trained through grad_x, on a log axis."""
    theme()
    fig = figure(figsize=(7.6, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(len(history)), history, color=FIRES, lw=2.2)
    ax.set_yscale("log")
    ax.set_xlabel("gradient step")
    ax.set_ylabel("mean squared error")
    ax.annotate(f"start  {history[0]:.4f}", (0, history[0]),
                xytext=(14, 6), textcoords="offset points", fontsize=9.5,
                color=FIRES, fontweight="bold")
    ax.annotate(f"end  {history[-1]:.2e}", (len(history) - 1, history[-1]),
                xytext=(-8, 28), textcoords="offset points", ha="right",
                fontsize=9.5, color=FIRES, fontweight="bold")
    title(ax, "Two linear layers, trained",
          "the only thing connecting them is layer 2's grad_x becoming "
          "layer 1's dY")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ exercise 7


def _node_box(ax, cx, cy, spec, w=1.5, h=0.92):
    """One node: its label, the value it holds, and the gradient on it."""
    from matplotlib.patches import FancyBboxPatch

    leaf = spec["op"] == "leaf"
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.06,rounding_size=0.12",
        facecolor="#eef2f7" if leaf else SURFACE,
        edgecolor=INK_SOFT if leaf else GRID, linewidth=1.4, zorder=3))
    ax.text(cx, cy + 0.26, spec["label"], ha="center", va="center",
            fontsize=11, fontweight="bold", color=INK, zorder=4)
    ax.text(cx, cy + 0.02, f"{spec['value']:.4f}", ha="center", va="center",
            fontsize=9.5, color=INK_SOFT, zorder=4)
    ax.text(cx, cy - 0.25, f"grad {spec['grad']:+.4f}", ha="center",
            va="center", fontsize=9.5, zorder=4,
            color=FIRES if spec["grad"] >= 0 else QUIET)


def graph_figure(spec):
    """A computation graph with the local gradient written on every edge.

    spec: the list of dicts 07 builds, one per node, each with id, label, op,
    value, grad, depth and (parent_id, local_gradient) pairs.
    """
    theme()
    by_id = {n["id"]: n for n in spec}
    columns = {}
    for n in spec:
        columns.setdefault(n["depth"], []).append(n)

    pos = {}
    tallest = max(len(c) for c in columns.values())
    for depth, nodes in columns.items():
        for k, n in enumerate(nodes):
            span = (len(nodes) - 1) / 2.0
            pos[n["id"]] = (depth * 2.9, (span - k) * 1.5)

    fig = figure(figsize=(12.6, 1.9 + 1.5 * tallest))
    ax = fig.add_subplot(1, 3, (1, 2))

    for n in spec:
        x1, y1 = pos[n["id"]]
        for pid, local in n["parents"]:
            x0, y0 = pos[pid]
            ax.annotate("", xy=(x1 - 0.82, y1), xytext=(x0 + 0.82, y0),
                        arrowprops=dict(arrowstyle="-|>", color=INK_SOFT,
                                        linewidth=1.3, shrinkA=0, shrinkB=0,
                                        connectionstyle="arc3,rad=0.0"),
                        zorder=1)
            # 0.72 of the way along rather than the midpoint: edges that
            # cross would otherwise stack their labels on the crossing point.
            ax.text(x0 + 0.72 * (x1 - x0), y0 + 0.72 * (y1 - y0) + 0.17,
                    f"{local:+.4f}",
                    ha="center", va="bottom", fontsize=9,
                    color=ACCENT, fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=SURFACE,
                              edgecolor="none"))

    for n in spec:
        _node_box(ax, *pos[n["id"]], n)

    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    ax.set_xlim(min(xs) - 1.2, max(xs) + 1.2)
    ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    title(ax, "One expression, drawn",
          "green numbers are local gradients: how much this node moves when "
          "that parent moves")

    leaves = [n for n in spec if n["op"] == "leaf"]
    out = spec[-1]
    lines = ["The green number on an edge is\n"
             "one local rule, evaluated with\n"
             "up = 1. It only involves the two\n"
             "nodes the edge joins.\n"]
    for lf in leaves:
        paths = _paths(by_id, out["id"], lf["id"])
        if not paths:
            continue
        terms = " + ".join(
            " x ".join(f"{g:.4f}" for g in path) or "1" for path in paths)
        total = sum(_product(path) for path in paths)
        lines.append(f"grad on {lf['label']}: {len(paths)} path"
                     f"{'s' if len(paths) != 1 else ''} from {out['label']}\n"
                     f"  {terms}\n  = {total:+.4f}\n")
    lines.append("Multiply along a path, add across\n"
                 "paths. backward() does exactly\n"
                 "that without ever listing a path,\n"
                 "by accumulating into each node.")

    ax = fig.add_subplot(1, 3, 3)
    ax.axis("off")
    ax.text(0, 0.98, "\n".join(lines), va="top", fontsize=9.6, color=INK,
            linespacing=1.5, family="monospace")

    fig.suptitle("Local rules on the edges, gradients in the boxes",
                 x=0.02, y=0.99, ha="left", fontsize=13, fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig


def _paths(by_id, from_id, to_id):
    """Every route from the output down to one leaf, as lists of edge values."""
    if from_id == to_id:
        return [[]]
    out = []
    for pid, local in by_id[from_id]["parents"]:
        for rest in _paths(by_id, pid, to_id):
            out.append([local] + rest)
    return out


def _product(values):
    total = 1.0
    for v in values:
        total *= v
    return total


def xor_autodiff_figure(history):
    """XOR trained through the three local rules, with nothing derived."""
    theme()
    fig = figure(figsize=(7.6, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(range(len(history)), history, color=FIRES, lw=2.2)
    ax.set_yscale("log")
    ax.set_xlabel("gradient step")
    ax.set_ylabel("mean loss over the four XOR examples")
    ax.annotate(f"start  {history[0]:.4f}", (0, history[0]),
                xytext=(14, 8), textcoords="offset points", fontsize=9.5,
                color=FIRES, fontweight="bold")
    ax.annotate(f"end  {history[-1]:.2e}", (len(history) - 1, history[-1]),
                xytext=(-8, 26), textcoords="offset points", ha="right",
                fontsize=9.5, color=FIRES, fontweight="bold")
    title(ax, "03's network, trained by autodiff",
          "same architecture as 03, and no gradient in it was derived by hand")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ exercise 8

# Four series, and the palette only guarantees separation for three. The fourth
# takes INK, and every line is labelled at its own right-hand end, so nothing
# in these two figures depends on colour to be read.
ACT_COLOURS = {"sigmoid": QUIET, "tanh": ACCENT, "relu": FIRES, "gelu": INK}

# relu and gelu sit on top of each other for most of the positive axis, so relu
# is drawn wide and gelu is drawn through it.
ACT_WIDTH = {"sigmoid": 2.1, "tanh": 2.1, "relu": 3.4, "gelu": 1.8}


def _label_series(ax, x, items, min_gap=0.065):
    """items: [(y, text, colour)], labelled just right of x.

    Two lines that finish at the same height would print their labels on top of
    each other, so the labels are pushed apart in axes-fraction space first.
    Call after the limits and the scale are set.
    """
    ylo, yhi = ax.get_ylim()
    log = ax.get_yscale() == "log"
    lo, hi = (np.log10(ylo), np.log10(yhi)) if log else (ylo, yhi)

    def to_frac(y):
        return ((np.log10(y) if log else y) - lo) / (hi - lo)

    def to_data(f):
        return 10 ** (lo + f * (hi - lo)) if log else lo + f * (hi - lo)

    fracs = [to_frac(y) for y, _, _ in items]
    order = sorted(range(len(items)), key=lambda i: fracs[i])
    for k in range(1, len(order)):
        below, above = order[k - 1], order[k]
        if fracs[above] - fracs[below] < min_gap:
            fracs[above] = fracs[below] + min_gap
    # Pushing apart can carry the top label off the axes, where it would not
    # be drawn at all. Slide the whole set back down if that happened.
    overshoot = max(fracs) - 1.0
    if overshoot > 0:
        fracs = [f - overshoot for f in fracs]

    for (_, text, colour), f in zip(items, fracs):
        ax.annotate(text, (x, to_data(f)), xytext=(8, 0),
                    textcoords="offset points", fontsize=9.5, color=colour,
                    fontweight="bold", va="center")


def activation_figure(entries):
    """entries: [(name, value_fn, slope_fn)] -- the function, then its slope."""
    theme()
    zs = np.linspace(-5, 5, 601)

    fig = figure(figsize=(12.4, 4.8))
    panels = [
        (1, lambda name, f, s: [f(z) for z in zs], (-1.4, 5.2), "The functions",
         "relu and gelu run off the top of the panel; sigmoid and tanh cannot"),
        (2, lambda name, f, s: [s(z) for z in zs], (-0.15, 1.25), "Their slopes",
         "a stack of d layers multiplies d of these together"),
    ]
    for pos, values, ylim, head, sub in panels:
        ax = fig.add_subplot(1, 2, pos)
        ax.axhline(0, color=GRID, lw=1)
        ax.axvline(0, color=GRID, lw=1)
        ends = []
        for name, f, s in entries:
            ys = values(name, f, s)
            ax.plot(zs, ys, color=ACT_COLOURS[name], lw=ACT_WIDTH[name])
            ends.append((ys[-1], name, ACT_COLOURS[name]))
        if pos == 2:
            ax.axhline(0.25, color=QUIET, lw=1.1, ls=":")
            ax.annotate("sigmoid never gets steeper than 0.25", (-4.9, 0.27),
                        fontsize=9, color=QUIET, va="bottom")
        ax.set_xlim(-5, 6.6)
        ax.set_ylim(*ylim)
        ax.set_xlabel("z")
        _label_series(ax, zs[-1], ends)
        title(ax, head, sub)

    fig.suptitle("Four activations, and the numbers a deep stack multiplies",
                 x=0.02, y=0.99, ha="left", fontsize=13, fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def depth_figure(profiles, shown_depths):
    """profiles: {scale label: {activation: [gradient norm at each depth]}}."""
    theme()
    labels = list(profiles)
    fig = figure(figsize=(12.4, 5.0))

    floor = min(v for rows in profiles.values() for norms in rows.values()
                for v in norms if v > 0)
    ceiling = max(v for rows in profiles.values() for norms in rows.values()
                  for v in norms)

    for pos, label in enumerate(labels, 1):
        ax = fig.add_subplot(1, len(labels), pos)
        rows = profiles[label]
        depth = max(len(n) for n in rows.values())
        ends = []
        for name, norms in rows.items():
            ax.plot(range(1, len(norms) + 1), norms,
                    color=ACT_COLOURS[name], lw=ACT_WIDTH[name])
            ends.append((norms[-1], name, ACT_COLOURS[name]))
        ax.set_yscale("log")
        ax.set_xlim(1, depth * 1.22)
        ax.set_ylim(floor * 0.3, ceiling * 3)
        ax.set_xticks(list(shown_depths))
        ax.set_xlabel("layers the gradient came back through")
        if pos == 1:
            ax.set_ylabel("gradient norm")
        _label_series(ax, depth, ends)
        title(ax, label.split(",")[0], label.split(",", 1)[1].strip())

    fig.suptitle("The same four activations, two weight scales",
                 x=0.02, y=0.99, ha="left", fontsize=13, fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return fig


# ------------------------------------------------------------------ exercise 3


def sigmoid_figure(sigmoid_fn, slope_fn):
    """Why the step function had to go. Needs nothing written."""
    theme()
    fig = figure(figsize=(11.5, 4.8))
    zs = np.linspace(-8, 8, 400)

    ax = fig.add_subplot(1, 2, 1)
    ax.plot(zs, [1 if z > 0 else 0 for z in zs], color=QUIET, lw=2,
            label="step: the old one")
    ax.plot(zs, [sigmoid_fn(z) for z in zs], color=FIRES, lw=2.4,
            label="sigmoid: the new one")
    ax.axhline(0.5, color=GRID, lw=1, zorder=0)
    ax.axvline(0, color=GRID, lw=1, zorder=0)
    ax.set_xlabel("z   (the weighted sum)")
    ax.set_ylabel("what the unit outputs")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    title(ax, "Step vs sigmoid",
          "the step is flat either side of a cliff; the sigmoid is a ramp")

    ax2 = fig.add_subplot(1, 2, 2)
    outs = [sigmoid_fn(z) for z in zs]
    ax2.plot(zs, [slope_fn(a) for a in outs], color=FIRES, lw=2.4)
    ax2.fill_between(zs, [slope_fn(a) for a in outs], color=FIRES, alpha=0.12)
    ax2.set_xlabel("z   (the weighted sum)")
    ax2.set_ylabel("how steep the sigmoid is there")
    ax2.annotate("steepest here, output near 0.5:\nthe unit is unsure and moves a lot",
                 xy=(0, 0.25), xytext=(1.0, 0.205), fontsize=9, color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK_SOFT, lw=1))
    ax2.annotate("flat out here, output near 1:\nconfident, and barely moves",
                 xy=(5.2, slope_fn(sigmoid_fn(5.2))), xytext=(0.6, 0.075),
                 fontsize=9, color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK_SOFT, lw=1))
    title(ax2, "The slope is the whole point",
          "a step function's slope is zero everywhere, so it can never say "
          "'which way is better'")
    fig.tight_layout()
    return fig


def network_region_figure(predict_fn, solved):
    """What two layers carve out that one neuron never could."""
    theme()
    fig = figure(figsize=(7.4, 6.4))
    ax = fig.add_subplot(1, 1, 1)
    lo, hi = -0.25, 1.25

    gx, gy = np.meshgrid(np.linspace(lo, hi, 200), np.linspace(lo, hi, 200))
    zz = np.array([[predict_fn((a, b)) for a, b in zip(rx, ry)]
                   for rx, ry in zip(gx, gy)])

    cs = ax.contourf(gx, gy, zz, levels=np.linspace(0, 1, 21), cmap="Blues",
                     alpha=0.9)
    ax.contour(gx, gy, zz, levels=[0.5], colors=[INK], linewidths=2.2)
    bar = fig.colorbar(cs, ax=ax, pad=0.02, ticks=[0, 0.25, 0.5, 0.75, 1])
    bar.set_label("what the network outputs", color=INK_SOFT, fontsize=9)
    bar.ax.tick_params(labelsize=8)

    for (x1, x2), want in [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]:
        ax.scatter([x1], [x2], s=190, zorder=5,
                   color=FIRES if want else QUIET,
                   edgecolor=SURFACE, linewidth=2.5)
        ax.annotate(str(want), (x1, x2), color=SURFACE, fontsize=10,
                    fontweight="bold", ha="center", va="center", zorder=6)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    title(ax, "Two layers, one XOR" if solved else "Two layers, not there yet",
          "the black line is the boundary -- compare it with the straight ones "
          "in exercise 1")
    fig.tight_layout()
    return fig
