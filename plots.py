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


def boundary_family_figure(families, boundary_fn):
    """Every gate at once, on ONE set of axes.

    The old figure put five gates in five panels and asked you to compare them
    from memory. Overlaid, the two claims in the caption are single glances:
    lines sharing a colour share their weights, and they are parallel.
    """
    theme()
    lo, hi = -0.6, 1.6
    fig = figure(figsize=(11.8, 6.4))
    ax = fig.add_subplot(1, 2, 1)

    for label, weights, bias, colour, dash in families:
        xs = np.array([lo, hi])
        ys = np.array([boundary_fn(x, weights, bias) for x in xs])
        ax.plot(xs, ys, color=colour, lw=2.4, ls=dash, zorder=3,
                label=f"{label}   w={list(weights)}  b={bias:+g}")

    _corners(ax)
    for x1, x2 in CORNERS:
        ax.annotate(f"({x1},{x2})", (x1, x2), xytext=(0, -18),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color=INK_SOFT)
    _square(ax, lo, hi)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True, facecolor=SURFACE,
              edgecolor=GRID, framealpha=0.96).set_zorder(6)
    title(ax, "Four neurons, one picture",
          "colour = which weights. same colour means the same weights")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    ax2.text(0, 0.97,
             "The two BLUE lines are AND and OR.\n"
             "They have identical weights and differ\n"
             "only in the bias, and they came out\n"
             "parallel. That is what a bias does: it\n"
             "SLIDES the line without turning it.\n\n"
             "The green and orange lines have different\n"
             "weights, and they point in different\n"
             "directions. That is what weights do: they\n"
             "TILT the line.\n\n"
             "Two knobs, two motions, and between them\n"
             "they can put a straight line anywhere on\n"
             "this square. That is the entire expressive\n"
             "range of one neuron -- which is also the\n"
             "setup for the next figure, where four\n"
             "points defeat all of it.",
             va="top", fontsize=10.5, color=INK, linespacing=1.6)
    fig.tight_layout()
    return fig


def xor_figure():
    """Why no single line works for XOR. Needs none of your functions."""
    theme()
    fig = figure(figsize=(9.2, 4.6))
    XOR = [((0, 0), 0), ((0, 1), 1), ((1, 0), 1), ((1, 1), 0)]

    ax = fig.add_subplot(1, 2, 1)
    for slope, intercept in [(-1, 0.5), (-1, 1.5), (1, 0.4), (-0.2, 0.9),
                             (2.5, -0.6)]:
        xs = np.linspace(-0.6, 1.6, 2)
        ax.plot(xs, slope * xs + intercept, color=INK_SOFT, lw=1, alpha=0.45)
    for (x1, x2), want in XOR:
        ax.scatter([x1], [x2], s=170, zorder=4,
                   color=FIRES if want else QUIET,
                   edgecolor=SURFACE, linewidth=2)
        ax.annotate(str(want), (x1, x2), color=SURFACE, fontsize=9,
                    fontweight="bold", ha="center", va="center", zorder=5)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.6, 1.6)
    ax.set_aspect("equal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    title(ax, "XOR", "try any straight line you like")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.axis("off")
    ax2.text(0, 0.95,
             "The two blue points sit on opposite corners.\n"
             "The two orange points sit on the other two.\n\n"
             "Every line above gets at least one point wrong,\n"
             "and so does every line you could draw.\n\n"
             "A single neuron IS a single line. That is the\n"
             "whole reason it cannot do XOR, and the whole\n"
             "reason networks have more than one layer.\n\n"
             "Two layers bend the space first, then draw the\n"
             "line in the bent space. That is exercise 3.",
             va="top", fontsize=10.5, color=INK, linespacing=1.6)
    fig.tight_layout()
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
