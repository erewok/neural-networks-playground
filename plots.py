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


def gate_panel(ax, name, weights, bias, cases, neuron_fn, boundary_fn):
    """One gate: the region the neuron fires in, the fence, and the four inputs."""
    lo, hi = -0.6, 1.6

    # Shade the two regions by asking the neuron about a grid of points.
    gx, gy = np.meshgrid(np.linspace(lo, hi, 220), np.linspace(lo, hi, 220))
    zz = np.array([[neuron_fn((a, b), weights, bias) for a, b in zip(rx, ry)]
                   for rx, ry in zip(gx, gy)], dtype=float)
    ax.contourf(gx, gy, zz, levels=[-0.5, 0.5, 1.5],
                colors=[QUIET, FIRES], alpha=0.10)

    # The fence itself: where the weighted sum lands exactly on zero.
    xs = np.linspace(lo, hi, 2)
    ys = [boundary_fn(x, weights, bias) for x in xs]
    ax.plot(xs, ys, color=INK_SOFT, lw=2, zorder=3, label="decision boundary")

    # The four inputs, coloured by the answer they are supposed to get.
    for (x1, x2), want in cases:
        ax.scatter([x1], [x2], s=150, zorder=4,
                   color=FIRES if want else QUIET,
                   edgecolor=SURFACE, linewidth=2)
        ax.annotate(str(want), (x1, x2), color=SURFACE, fontsize=9,
                    fontweight="bold", ha="center", va="center", zorder=5)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    title(ax, name, f"weights={list(weights)}  bias={bias}")


def gates_figure(gates, neuron_fn, boundary_fn):
    """gates: list of (name, weights, bias, cases)."""
    theme()
    n = len(gates)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig = figure(figsize=(4.2 * cols, 4.3 * rows))
    for i, (name, weights, bias, cases) in enumerate(gates, 1):
        ax = fig.add_subplot(rows, cols, i)
        gate_panel(ax, name, weights, bias, cases, neuron_fn, boundary_fn)
    # The leftover panel earns its place as a caption rather than sitting blank.
    if n < rows * cols:
        ax = fig.add_subplot(rows, cols, n + 1)
        ax.axis("off")
        ax.text(0, 0.92,
                "Same code in all five panels.\n"
                "Only the numbers changed.\n\n"
                "The weights TILT the line.\n"
                "AND and OR have identical\n"
                "weights, so their lines are\n"
                "parallel -- only the bias\n"
                "differs, and the bias SLIDES\n"
                "the line without turning it.\n\n"
                "Flip both weights negative and\n"
                "the shaded side swaps: that is\n"
                "all NAND and NOR are.",
                va="top", fontsize=10, color=INK, linespacing=1.55)

    fig.suptitle("One neuron draws one straight line",
                 x=0.02, y=0.995, ha="left", fontsize=13, fontweight="bold",
                 color=INK)
    fig.text(0.02, 0.958,
             "Blue = the neuron fires (1).   Orange = it stays quiet (0).   "
             "The line is where the weighted sum comes out to exactly zero.",
             fontsize=9.5, color=INK_SOFT, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.945])
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


# ------------------------------------------------------------------ exercise 4

# Sequential blue, light -> dark, for "which epoch is this". Discrete ordered
# marks, so the lightest step is 250 -- anything lighter recedes into the paper.
EPOCH_RAMP = ["#86b6ef", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
              "#1c5cab", "#0d366b"]


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
