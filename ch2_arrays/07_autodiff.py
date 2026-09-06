"""
Stage 7: automatic differentiation.

03 asked you to derive, by hand, how the loss responded to each weight in a
two-layer network. 06 asked for the same thing for one linear layer in matrix
form. Both derivations only covered the exact arrangement of operations in
front of you. Change the network and you derive again.

This file builds the thing that stops that. You write a rule for each single
operation, once. Everything after that is bookkeeping over a graph.

HOW THE GRAPH GETS BUILT

There is a Node class below, and add(), mul() and sigmoid() that take Nodes and
return a new Node. Each new Node computes its value immediately and keeps a
reference to the Nodes it came from. An expression is therefore ordinary
Python:

    def f(x, y):
        return mul(add(x, y), sigmoid(x))

Calling f() on two leaf Nodes computes the answer and leaves behind a record of
how the answer was reached. PyTorch works this way. Running the forward pass is
what builds the graph.

WHAT YOU WRITE

    grad_add(a, b, out, up)     how add's output responds to each of its inputs
    grad_mul(a, b, out, up)     the same for mul
    grad_sigmoid(a, out, up)    the same for sigmoid
    toposort(node)              an order to visit the graph in
    backward(node)              fill in .grad on every node

The three rules are local. grad_mul cannot see the rest of the graph and does
not know what happens to its output. It is handed `up`, which is how fast the
final answer rises when this node's output rises, and it answers the same
question about each of its own inputs.

`up` is where the chain rule happens. Nothing else in this file combines
derivatives.

THE RULES ARE MEASURABLE

`just run 7` starts by taking one operation on its own, nudging one input, and
dividing the change in the output by the size of the nudge. Those printed
numbers are what the three rule functions have to return when `up` is 1. Read
that section before writing anything.

WHY THE ORDER MATTERS

A node whose output is used in two places collects two separate contributions
to its gradient. Both have to arrive before that node's own rule runs, or the
rule runs on a number that is still incomplete.

toposort is what arranges that. Every node comes after the nodes it was built
from, so walking the list backwards reaches every use of a node before reaching
the node itself.

The run demonstrates this. It takes a graph with a shared node, runs the
backward pass twice over two different orders, and prints both answers next to
the measured one. That happens before you write toposort.

PROPERTIES

  1. A leaf appearing twice in an expression has one gradient, and it is the
     sum of what both appearances contribute.
  2. toposort lists every reachable node once, including the shared ones.
  3. The output node's own gradient is 1.
  4. A leaf the output does not depend on has gradient 0.
  5. grad_add's answer does not depend on a or b. grad_mul's does.
  6. grad_sigmoid can be written from `out` alone, without looking at `a`.
     The forward pass already computed what it needs.

LAST SECTION

03's network gets rebuilt out of add, mul and sigmoid. Its thirteen gradients
come out of backward() with nothing derived, get checked against measurement,
and then train it on XOR.

`just plot 7` draws a small graph with the local gradient on every edge.
"""

import math
import random
import sys

from harness import (CROSS, TICK, Cell, ExerciseTable, approx_nested, console,
                     exact, fmt_grid, not_written, section, summary)


# ------------------------------------------------------------------ the graph
# Working code. Nothing in this block needs fixing.


def _sig(z):
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


class Node:
    """One value, and a record of where it came from.

    data     the number, computed when the node was made
    op       "leaf", "add", "mul" or "sigmoid"
    parents  the Nodes this one was built from. Empty for a leaf.
    grad     how fast the final answer rises when this node's value rises.
             Starts at 0. backward() is what fills it in.
    """

    __slots__ = ("data", "op", "parents", "grad", "label")

    def __init__(self, data, op="leaf", parents=(), label=""):
        self.data = float(data)
        self.op = op
        self.parents = tuple(parents)
        self.grad = 0.0
        self.label = label

    def __repr__(self):
        return f"{self.label}={self.data:.4f}"


_COUNT = [0]


def _next_label(prefix):
    _COUNT[0] += 1
    return f"{prefix}{_COUNT[0]}"


def leaf(value, label=None):
    return Node(value, "leaf", (), label or _next_label("x"))


def add(a, b):
    return Node(a.data + b.data, "add", (a, b), _next_label("+"))


def mul(a, b):
    return Node(a.data * b.data, "mul", (a, b), _next_label("*"))


def sigmoid(a):
    return Node(_sig(a.data), "sigmoid", (a,), _next_label("s"))


def named(node, label):
    """Rename a node so the tables and figures can refer to it."""
    node.label = label
    return node


def zero_grads(node):
    """Set .grad back to 0 on every node reachable from this one."""
    node.grad = 0.0
    for p in node.parents:
        zero_grads(p)


def reachable(node):
    """The nodes this one was built from, itself included. No order promised."""
    seen = {id(node): node}
    stack = [node]
    while stack:
        n = stack.pop()
        for p in n.parents:
            if id(p) not in seen:
                seen[id(p)] = p
                stack.append(p)
    return list(seen.values())


# ------------------------------------------------------------ the measurement
# Working code. An expression here is a Python function over leaf Nodes, so it
# can be rebuilt at whatever input values we like.


def measure(build, values, i, h=1e-5):
    """How fast build()'s output moves when input i moves. No calculus in it."""
    up = list(values)
    up[i] += h
    dn = list(values)
    dn[i] -= h
    return (build(*[leaf(v) for v in up]).data -
            build(*[leaf(v) for v in dn]).data) / (2 * h)


def measure_all(build, values, h=1e-5):
    return [measure(build, values, i, h) for i in range(len(values))]


# --------------------------------------------------------------- the originals
# Left here on purpose. Don't edit these -- the harness runs them beside yours.


def grad_add_broken(a, b, out, up):
    return [up * a, up * b]


def grad_mul_broken(a, b, out, up):
    return [up * a, up * b]


def grad_sigmoid_broken(a, out, up):
    return [up * out]


def toposort_broken(node):
    order = [node]
    for p in node.parents:
        order.extend(toposort_broken(p))
    return order


def backward_broken(node):
    zero_grads(node)

    def go(n, up):
        n.grad = up
        if n.op != "leaf":
            for p, g in zip(n.parents, local_grads(n, up)):
                go(p, g)

    go(node, 1.0)


# ---------------------------------------------------------------------- yours
# Five functions. The three rules first: several sections below run on them.
# The rules return a list, one gradient per input, in parent order.


def grad_add(a, b, out, up):
    """add(a, b) produced `out`, and raising `out` raises the answer at `up`.

    How fast does the answer rise when a rises? When b rises? Two numbers.
    """
    raise NotImplementedError


def grad_mul(a, b, out, up):
    """mul(a, b) produced `out`, and raising `out` raises the answer at `up`.

    Two numbers, the same question as grad_add.
    """
    raise NotImplementedError


def grad_sigmoid(a, out, up):
    """sigmoid(a) produced `out`, and raising `out` raises the answer at `up`.

    One number. `a` is passed in, and you may not need it.
    """
    raise NotImplementedError


def toposort(node):
    """Every node reachable from `node`, in an order to walk forwards in.

    A node never appears before any node it was built from, so `node` itself
    comes last. Each node appears once, however many places it is used in.
    """
    raise NotImplementedError


def backward(node):
    """Fill in .grad on every node in the graph, treating `node` as the answer.

    Start from 0 everywhere. `node`'s own grad is 1, since raising the answer
    by one raises the answer by one. Then work through the rest so that no
    node's rule is applied until every use of that node has contributed to its
    grad.

    local_grads(n, up) below runs the right rule for whatever op n is.
    Returns nothing. The gradients are left on the nodes.
    """
    raise NotImplementedError


# ---------------------------------------------------------------- test harness
# Dispatch and drawing. Nothing here needs fixing.


def local_grads(node, up):
    """Run your rule for this node's op. One gradient per parent, in order."""
    vals = [p.data for p in node.parents]
    if node.op == "add":
        return grad_add(vals[0], vals[1], node.data, up)
    if node.op == "mul":
        return grad_mul(vals[0], vals[1], node.data, up)
    if node.op == "sigmoid":
        return grad_sigmoid(vals[0], node.data, up)
    raise ValueError(f"no rule for {node.op}")


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


def short(v):
    return f"{round(v, 4):g}"


def compare(title, original, mine, argnames, cases, note="", places=4):
    """cases: list of (args_tuple, expected)."""
    written = is_written(mine, cases[0][0])
    t = ExerciseTable(
        title=title, note=note, input_header="arguments",
        yours_written=written, fmt=fmt_grid(places),
        compare=approx_nested,
    )
    for args, want in cases:
        label = " ".join(f"{n}={short(v)}" for n, v in zip(argnames, args))
        t.add(label, want, attempt(original, args),
              attempt(mine, args) if written else None)
    return t.render()


def claim(text, holds):
    console.print(f"  [{'bold green' if holds else 'bold red'}]"
                  f"{TICK if holds else CROSS}[/] {text}")
    return holds


def labels(nodes):
    return " ".join(n.label for n in nodes)


# ----------------------------------------------- one operation, measured alone


def run_local_measurement():
    """What the three rules have to return when up is 1."""
    console.print("[dim]Each row below builds one operation as the whole "
                  "expression, nudges one input, and divides the change in the "
                  "output by the size of the nudge.[/dim]")
    console.print()

    console.print("  [bold]add(a, b)[/bold]")
    for a, b in [(3.0, 4.0), (-2.0, 0.5), (10.0, 10.0)]:
        g = measure_all(lambda p, q: add(p, q), [a, b])
        console.print(f"    a={a:>6.2f}  b={b:>6.2f}  out={a + b:>8.4f}   "
                      f"d/da {g[0]:+.4f}   d/db {g[1]:+.4f}")

    console.print()
    console.print("  [bold]mul(a, b)[/bold]")
    for a, b in [(3.0, 4.0), (-2.0, 0.5), (0.0, 7.0), (5.0, 5.0)]:
        g = measure_all(lambda p, q: mul(p, q), [a, b])
        console.print(f"    a={a:>6.2f}  b={b:>6.2f}  out={a * b:>8.4f}   "
                      f"d/da {g[0]:+.4f}   d/db {g[1]:+.4f}")

    console.print()
    console.print("  [bold]sigmoid(a)[/bold]")
    for a in [0.0, 0.7, 2.0, -2.0, 4.0]:
        g = measure(lambda p: sigmoid(p), [a], 0)
        console.print(f"    a={a:>6.2f}              out={_sig(a):>8.4f}   "
                      f"d/da {g:+.4f}")

    console.print()
    console.print("[dim]`up` is absent from all of those numbers. Each "
                  "operation was the whole expression there, so its own output "
                  "was the answer and up was 1.[/dim]")


# -------------------------------------------------- the same graph, two orders


def shared_expr(x):
    """a gets used twice: once on its own, once inside a product."""
    a = sigmoid(x)
    return add(a, mul(x, a))


def backward_in_this_order(order, node):
    """Working code. Runs the backward pass over whatever order it is handed."""
    zero_grads(node)
    node.grad = 1.0
    for n in order:
        if n.op == "leaf":
            continue
        for p, g in zip(n.parents, local_grads(n, n.grad)):
            p.grad += g


def build_shared():
    out = named(shared_expr(leaf(0.5)), "out")
    a, b = out.parents
    named(a, "a")
    named(b, "b")
    named(b.parents[0], "x")
    return out, a, b, b.parents[0]


def run_order_demo():
    out, a, b, x = build_shared()
    console.print()
    console.print("  [cyan]x[/cyan] = 0.5,  [cyan]a[/cyan] = sigmoid(x),  "
                  "[cyan]b[/cyan] = mul(x, a),  [cyan]out[/cyan] = add(a, b)")
    console.print("  [dim]a is a parent of b and of out. x is a parent of a "
                  "and of b.[/dim]")
    console.print()

    want = measure(shared_expr, [0.5], 0)
    console.print(f"  measured d out/d x   [bold]{want:+.6f}[/bold]   "
                  f"[dim]nudge x, watch out[/dim]")

    for name, order in [("out, b, a, x", [out, b, a, x]),
                        ("out, a, b, x", [out, a, b, x])]:
        backward_in_this_order(order, out)
        ok = abs(x.grad - want) < 1e-6
        mark = f"[bold green]{TICK}[/]" if ok else f"[bold red]{CROSS}[/]"
        console.print(f"  order {name:<14}  x.grad {x.grad:+.6f}   {mark}")

    console.print()
    console.print("  [dim]The second order applies a's rule while a.grad still "
                  "holds only the\n  contribution from out. b's contribution "
                  "arrives after that and is\n  never used.[/dim]")


# ----------------------------------------------------- toposort, as a sequence
# Each graph here has exactly one valid order, so there is one string to
# match. Branching graphs get checked further down instead.


def topo_case_1():
    return named(leaf(1.0), "x")


def topo_case_2():
    x = named(leaf(1.0), "x")
    return named(mul(x, x), "t")


def topo_case_3():
    x = named(leaf(1.0), "x")
    a = named(sigmoid(x), "a")
    return named(mul(x, a), "b")


def topo_case_4():
    x = named(leaf(1.0), "x")
    a = named(add(x, x), "a")
    b = named(sigmoid(a), "b")
    return named(mul(a, b), "c")


def topo_case_5():
    x = named(leaf(1.0), "x")
    a = named(sigmoid(x), "a")
    b = named(sigmoid(a), "b")
    c = named(add(a, b), "c")
    return named(mul(c, x), "d")


TOPO_CASES = [
    ("x", topo_case_1, "x"),
    ("t = x*x", topo_case_2, "x t"),
    ("a=s(x) b=x*a", topo_case_3, "x a b"),
    ("a=x+x b=s(a) c=a*b", topo_case_4, "x a b c"),
    ("a=s(x) b=s(a) c=a+b d=c*x", topo_case_5, "x a b c d"),
]


def run_toposort_table():
    written = is_written(toposort, (topo_case_1(),))
    t = ExerciseTable(
        title="toposort(node)",
        note="the labels in order. every graph here has exactly one valid "
             "order, so there is one string to match. rows 2 and 4 use a node "
             "twice and still list it once",
        input_header="expression", yours_written=written,
        fmt=str, compare=exact,
    )
    for label, make, want in TOPO_CASES:
        t.add(label, want,
              attempt(lambda m: labels(toposort_broken(m())), (make,)),
              attempt(lambda m: labels(toposort(m())), (make,))
              if written else None)
    return t.render()


def run_toposort_claims():
    """Branching graphs, where more than one order is valid."""

    def valid(node):
        order = toposort(node)
        nodes = reachable(node)
        if {id(n) for n in order} != {id(n) for n in nodes}:
            return False
        if len(order) != len(nodes):
            return False
        where = {id(n): i for i, n in enumerate(order)}
        return all(where[id(p)] < where[id(n)]
                   for n in order for p in n.parents)

    rng = random.Random(7)
    graphs = []
    for _ in range(200):
        n_leaves = rng.randint(1, 4)
        build = build_from(random_recipe(rng, n_leaves, rng.randint(1, 7)))
        graphs.append(build(*[leaf(rng.uniform(-1, 1))
                              for _ in range(n_leaves)]))

    console.print()
    good = claim("200 random graphs, and in every one each node came after all "
                 "of its parents", all(valid(g) for g in graphs))
    good &= claim("no node listed twice and none left out",
                  all(len(toposort(g)) == len(reachable(g)) for g in graphs))
    out = build_shared()[0]
    good &= claim(f"the shared graph came out as: {labels(toposort(out))}",
                  labels(toposort(out)) == "x a b out")
    return good


# ---------------------------------------------------- backward, on expressions
# `want` is the measured gradient, not a constant typed in here.


def leaf_grads(bw, build, values):
    leaves = [leaf(v) for v in values]
    out = build(*leaves)
    bw(out)
    return [n.grad for n in leaves]


BACKWARD_CASES = [
    ("x+y", lambda x, y: add(x, y), (3.0, 4.0)),
    ("x*y", lambda x, y: mul(x, y), (3.0, 4.0)),
    ("x*x", lambda x: mul(x, x), (3.0,)),
    ("s(x)", lambda x: sigmoid(x), (0.0,)),
    ("(x+y)*x", lambda x, y: mul(add(x, y), x), (2.0, 3.0)),
    ("x*s(x)", lambda x: mul(x, sigmoid(x)), (1.0,)),
    ("a=s(x); a+x*a", shared_expr, (0.5,)),
    ("x*x, y unused", lambda x, y: mul(x, x), (2.0, 9.0)),
]


def run_backward_table():
    written = is_written(backward, (leaf(1.0),))
    t = ExerciseTable(
        title="backward(node), read off the leaves",
        note="one gradient per leaf. `want` is measured by nudging that leaf "
             "rather than typed in here. rows 3, 5, 7 and 8 use a leaf in more "
             "than one place",
        input_header="expression", yours_written=written,
        fmt=fmt_grid(3), compare=approx_nested,
    )
    for label, build, values in BACKWARD_CASES:
        want = measure_all(build, values)
        shown = f"{label} at {','.join(short(v) for v in values)}"
        t.add(shown, want,
              attempt(leaf_grads, (backward_broken, build, values)),
              attempt(leaf_grads, (backward, build, values))
              if written else None)
    return t.render()


# ------------------------------------------------ random expressions, measured


def random_recipe(rng, n_leaves, n_ops):
    """A list of operations, so the same graph can be rebuilt at new values."""
    recipe = []
    for k in range(n_ops):
        size = n_leaves + k
        kind = rng.choice(["add", "mul", "sigmoid", "sigmoid"])
        if kind == "sigmoid":
            recipe.append(("sigmoid", rng.randrange(size)))
        else:
            recipe.append((kind, rng.randrange(size), rng.randrange(size)))
    return recipe


def build_from(recipe):
    def build(*leaves):
        pool = list(leaves)
        for item in recipe:
            if item[0] == "sigmoid":
                pool.append(sigmoid(pool[item[1]]))
            elif item[0] == "add":
                pool.append(add(pool[item[1]], pool[item[2]]))
            else:
                pool.append(mul(pool[item[1]], pool[item[2]]))
        return pool[-1]
    return build


def run_against_measurement(trials=300):
    rng = random.Random(707)
    worst = 0.0
    failed = None
    for _ in range(trials):
        n_leaves = rng.randint(1, 4)
        build = build_from(random_recipe(rng, n_leaves, rng.randint(1, 7)))
        values = [rng.uniform(-1.5, 1.5) for _ in range(n_leaves)]
        try:
            got = leaf_grads(backward, build, values)
        except Exception as exc:
            failed = f"{type(exc).__name__}: {exc}"
            break
        want = measure_all(build, values)
        worst = max(worst, max(abs(g - w) for g, w in zip(got, want)))

    console.print()
    if failed:
        console.print(f"  [bold red]{CROSS}[/] backward raised: "
                      f"[red]{failed}[/red]")
        return False
    ok = claim(f"{trials} random expressions, up to 7 operations over up to 4 "
               f"leaves, and no gradient differed from its measurement by more "
               f"than {worst:.3g}", worst < 1e-5)
    if ok:
        console.print("\n  [dim]Three local rules and an ordering. None of it "
                      "was derived for any particular expression.[/dim]")
    return ok


# ------------------------------------------------------- 03's network, rebuilt

XOR = [((0.0, 0.0), 0.0), ((0.0, 1.0), 1.0),
       ((1.0, 0.0), 1.0), ((1.0, 1.0), 0.0)]
N_HIDDEN = 3
N_PARAMS = N_HIDDEN * 3 + N_HIDDEN + 1   # 3 per hidden unit, then the output


def _net_output(p, x1, x2):
    """The forward pass, out of add, mul and sigmoid. Same shape as 03."""
    hs = []
    for j in range(N_HIDDEN):
        z = add(add(mul(x1, p[3 * j]), mul(x2, p[3 * j + 1])), p[3 * j + 2])
        hs.append(sigmoid(z))
    z = p[N_HIDDEN * 4]                              # the output bias
    for j in range(N_HIDDEN):
        z = add(z, mul(hs[j], p[N_HIDDEN * 3 + j]))
    return sigmoid(z)


def network_loss(x1v, x2v, tv):
    """A builder over the 13 parameters as leaves, returning 0.5*(pred - t)^2.

    Subtraction is a mul by a constant -1, so add, mul and sigmoid are the
    only operations in here.
    """
    def build(*p):
        pred = _net_output(p, leaf(x1v, "x1"), leaf(x2v, "x2"))
        diff = add(pred, mul(leaf(tv, "t"), leaf(-1.0, "neg")))
        return mul(leaf(0.5, "half"), mul(diff, diff))
    return build


def network_predict(params, x1v, x2v):
    return _net_output([leaf(v) for v in params],
                       leaf(x1v), leaf(x2v)).data


def network_grads(params, x1v, x2v, tv):
    """The thirteen gradients for one example, straight out of backward()."""
    p = [leaf(v) for v in params]
    out = network_loss(x1v, x2v, tv)(*p)
    backward(out)
    return out.data, [n.grad for n in p]


def train_xor(steps=1500, rate=2.0, seed=0):
    rng = random.Random(seed)
    params = [rng.uniform(-1.0, 1.0) for _ in range(N_PARAMS)]
    history = []
    for _ in range(steps):
        total = 0.0
        grads = [0.0] * N_PARAMS
        for (a, b), t in XOR:
            L, gs = network_grads(params, a, b, t)
            total += L
            for i, g in enumerate(gs):
                grads[i] += g
        history.append(total / len(XOR))
        params = [w - rate * g / len(XOR) for w, g in zip(params, grads)]
    return params, history


def run_network():
    rng = random.Random(3)
    params = [rng.uniform(-1.0, 1.0) for _ in range(N_PARAMS)]
    (a, b), t = XOR[1]
    _, got = network_grads(params, a, b, t)
    want = measure_all(network_loss(a, b, t), params)

    console.print()
    console.print(f"  [dim]{N_PARAMS} parameters, on the XOR example "
                  f"({a:.0f}, {b:.0f}) -> {t:.0f}[/dim]")
    for i, (g, w) in enumerate(zip(got, want)):
        console.print(f"  [cyan]p{i:<2}[/cyan]  backward {g:+.6f}   "
                      f"measured {w:+.6f}")
    worst = max(abs(g - w) for g, w in zip(got, want))
    good = claim(f"no parameter's gradient differed from its measurement by "
                 f"more than {worst:.3g}", worst < 1e-6)
    console.print("  [dim]03 derived these by hand for this one network "
                  "shape.[/dim]")

    console.print()
    params, history = train_xor()
    console.print(f"  [dim]mean loss, step 0:[/dim] {history[0]:.6f}"
                  f"   [dim]step {len(history) - 1}:[/dim] {history[-1]:.6f}")
    solved = True
    for (x1, x2), target in XOR:
        y = network_predict(params, x1, x2)
        hit = (y > 0.5) == (target > 0.5)
        solved = solved and hit
        mark = f"[bold green]{TICK}[/]" if hit else f"[bold red]{CROSS}[/]"
        console.print(f"    x=({x1:.0f}, {x2:.0f})  want {target:.0f}  "
                      f"got {y:.3f}   {mark}")
    good = claim("XOR, trained through the same three rules", solved) and good
    return good


# ---------------------------------------------------------------------- plots


def graph_spec(out):
    """A plain description of the graph, for plots.py to draw."""
    order = toposort(out)
    depth = {}
    for n in order:
        depth[id(n)] = 0 if not n.parents else \
            1 + max(depth[id(p)] for p in n.parents)
    backward(out)
    return [
        {
            "id": id(n),
            "label": n.label,
            "op": n.op,
            "value": n.data,
            "grad": n.grad,
            "depth": depth[id(n)],
            "parents": [] if n.op == "leaf" else
                       [(id(p), g) for p, g in
                        zip(n.parents, local_grads(n, 1.0))],
        }
        for n in order
    ]


def plot():
    import plots

    if not everything_written():
        console.print("\n[yellow]The graph figure needs all five functions "
                      "written first.[/yellow]")
        return
    x = named(leaf(0.5), "x")
    y = named(leaf(1.0), "y")
    out = named(mul(named(add(x, y), "s"), named(sigmoid(x), "g")), "out")
    plots.graph_figure(graph_spec(out))
    plots.xor_autodiff_figure(train_xor()[1])
    plots.done("07")


ALL = [
    (grad_add, (1.0, 1.0, 2.0, 1.0)),
    (grad_mul, (1.0, 1.0, 1.0, 1.0)),
    (grad_sigmoid, (0.0, 0.5, 1.0)),
    (toposort, (leaf(1.0),)),
    (backward, (leaf(1.0),)),
]


def everything_written():
    return all(is_written(fn, probe) for fn, probe in ALL)


if __name__ == "__main__":
    section("one operation at a time, measured")
    run_local_measurement()

    section("the three local rules")
    results = []

    r_add = compare(
        "grad_add(a, b, out, up)", grad_add_broken, grad_add,
        ["a", "b", "out", "up"],
        [
            ((2.0, 3.0, 5.0, 1.0), [1.0, 1.0]),
            ((2.0, 3.0, 5.0, 4.0), [4.0, 4.0]),
            ((-7.0, 0.5, -6.5, 2.0), [2.0, 2.0]),
            ((2.0, 3.0, 5.0, 0.0), [0.0, 0.0]),
            ((2.0, 3.0, 5.0, -3.0), [-3.0, -3.0]),
            ((1.0, 1.0, 2.0, 1.0), [1.0, 1.0]),
        ],
        "a and b do not appear in the answer, only up does. the original "
        "agrees on the two rows where that cannot show", places=2)

    r_mul = compare(
        "grad_mul(a, b, out, up)", grad_mul_broken, grad_mul,
        ["a", "b", "out", "up"],
        [
            ((3.0, 4.0, 12.0, 1.0), [4.0, 3.0]),
            ((3.0, 4.0, 12.0, 2.0), [8.0, 6.0]),
            ((5.0, 0.0, 0.0, 1.0), [0.0, 5.0]),
            ((-2.0, 6.0, -12.0, 1.0), [6.0, -2.0]),
            ((0.5, -4.0, -2.0, 10.0), [-40.0, 5.0]),
            ((3.0, 3.0, 9.0, 1.0), [3.0, 3.0]),
        ],
        "the two entries swap places against the arguments. the last row has "
        "a == b, where a swap cannot be seen", places=2)

    r_sig = compare(
        "grad_sigmoid(a, out, up)", grad_sigmoid_broken, grad_sigmoid,
        ["a", "out", "up"],
        [
            ((0.0, 0.5, 1.0), [0.25]),
            ((0.0, 0.5, 4.0), [1.0]),
            ((0.7, 0.6681877721681662, 1.0), [0.22171287329310904]),
            ((2.0, 0.8807970779778823, 1.0), [0.10499358540350662]),
            ((-2.0, 0.11920292202211755, 1.0), [0.1049935854035065]),
            ((4.0, 0.9820137900379085, 2.0), [0.035325412426582214]),
        ],
        "the same numbers the measured block printed, scaled by up. rows 4 and "
        "5 sit either side of 0, and row 6 is a saturated unit")

    results += [r_add, r_mul, r_sig]
    rules_ok = all(r[1] == r[2] for r in (r_add, r_mul, r_sig))

    section("the same graph walked in two orders")
    if rules_ok:
        run_order_demo()
    else:
        not_written("the ordering demonstration")
        console.print("[dim]It runs your three rules, so those come "
                      "first.[/dim]")

    section("an order to walk the graph in")
    results.append(run_toposort_table())

    section("the backward pass")
    if rules_ok:
        results.append(run_backward_table())
    else:
        not_written("the backward table")
        console.print("[dim]It runs your three rules as well.[/dim]")

    summary(results)

    if everything_written():
        section("toposort on branching graphs")
        run_toposort_claims()

        section("backward against measurement")
        run_against_measurement()

        section("03's network, rebuilt out of add, mul and sigmoid")
        run_network()
    else:
        not_written("the random-expression check and 03's network")

    if "plot" in sys.argv:
        plot()
