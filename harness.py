"""Shared terminal output for the exercises.

There is nothing to fix in this file -- it only draws tables. Every exercise
renders the same way: the untouched original on one side, your version on the
other, scored row by row against what the answer should be.

Colour scheme, so it reads at a glance:

    cyan      inputs / arguments -- what went in
    bold      want -- the answer that row is asking for
    yellow x  the original getting a row wrong. Expected. Not your problem.
    green  v  correct
    red    x  YOUR version getting a row wrong. This is the one to look at.
    dim  --   a function you have not written yet
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()

TICK = "✓"
CROSS = "✗"


def approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) < tol


def exact(a: Any, b: Any) -> bool:
    return a == b


def fmt_int(v: Any) -> str:
    return str(int(v))


def fmt_float(places: int = 2) -> Callable[[Any], str]:
    return lambda v: f"{float(v):+.{places}f}"


@dataclass
class Cell:
    """One side's answer for one row.

    value   the thing compared against `want`
    detail  optional intermediate shown before it, e.g. the pre-activation z
    """

    value: Any
    detail: str = ""


@dataclass
class ExerciseTable:
    """A table comparing the original and your version, row by row."""

    title: str
    note: str = ""
    input_header: str = "inputs"
    why_header: Optional[str] = None
    yours_written: bool = False
    fmt: Callable[[Any], str] = str
    compare: Callable[[Any, Any], bool] = exact
    rows: list = field(default_factory=list)

    def add(self, label: str, want: Any, original: Cell,
            yours: Optional[Cell] = None, why: str = "") -> None:
        self.rows.append((label, want, original, yours, why))

    # -- rendering ------------------------------------------------------

    def _cell(self, cell: Optional[Cell], want: Any, mine: bool) -> Text:
        if cell is None:
            return Text("--", style="dim")
        ok = self.compare(cell.value, want)
        if ok:
            mark_style = "bold green" if mine else "green"
        else:
            mark_style = "bold red" if mine else "yellow"
        out = Text()
        if cell.detail:
            out.append(f"{cell.detail}  ", style="dim")
        out.append(self.fmt(cell.value), style="bold" if mine else "none")
        out.append(f"  {TICK if ok else CROSS}", style=mark_style)
        return out

    def render(self) -> tuple:
        # Title and note are printed as their own lines rather than as rich's
        # table title/caption, which would wrap them into the table's width.
        console.print()
        console.print(Text(self.title, style="bold cyan"))
        if self.note:
            console.print(Text(self.note, style="dim italic"))

        table = Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False,
                      padding=(0, 1))
        table.add_column(self.input_header, style="cyan", no_wrap=True)
        table.add_column("want", style="bold", justify="right", no_wrap=True)
        table.add_column("original", justify="left", no_wrap=True)
        table.add_column("yours", justify="left", no_wrap=True)
        if self.why_header:
            table.add_column(self.why_header, style="dim italic")

        ok_orig = ok_mine = 0
        for label, want, original, yours, why in self.rows:
            if self.compare(original.value, want):
                ok_orig += 1
            if self.yours_written and yours is not None and \
                    self.compare(yours.value, want):
                ok_mine += 1
            cells = [
                label,
                self.fmt(want),
                self._cell(original, want, mine=False),
                self._cell(yours if self.yours_written else None, want, mine=True),
            ]
            if self.why_header:
                cells.append(why)
            table.add_row(*cells)

        console.print(table)
        return (ok_orig, ok_mine if self.yours_written else None, len(self.rows))


def score_line(label: str, got: int, total: int, mine: bool) -> Text:
    out = Text()
    out.append(f"{label:<10}", style="bold" if mine else "none")
    if got == total:
        out.append(f"{got}/{total} correct", style="bold green" if mine else "green")
    else:
        style = "bold red" if mine else "yellow"
        out.append(f"{got}/{total} correct", style=style)
        out.append(f"   {total - got} failing", style=style)
    return out


def summary(results: list) -> None:
    """results: list of (ok_original, ok_yours_or_None, n_rows) from render().

    Scores whatever you have written so far, so working through the functions
    one at a time still gives you a number that moves.
    """
    total = sum(n for _, _, n in results)
    ok_orig = sum(o for o, _, _ in results)
    done = [(o, m, n) for o, m, n in results if m is not None]

    console.print()
    console.rule("[bold]score[/bold]", style="dim")
    console.print(score_line("original", ok_orig, total, mine=False))

    if not done:
        console.print(Text("yours     not written yet", style="dim"))
        return

    scored = sum(n for _, _, n in done)
    ok_mine = sum(m for _, m, _ in done)
    line = score_line("yours", ok_mine, scored, mine=True)
    if len(done) < len(results):
        line.append(f"   ({len(done)} of {len(results)} functions written)",
                    style="dim")
    console.print(line)
    if len(done) == len(results) and ok_mine == total:
        console.print(f"\n  [bold green]{TICK} every row correct[/bold green]")


def section(title: str) -> None:
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]", style="cyan")


def outcome(name: str, solved: bool, should_solve: bool, detail: str = "",
            why: str = "") -> bool:
    """One line for 'did this task get learned', where some tasks SHOULD fail."""
    as_wanted = solved == should_solve
    out = Text()
    out.append(f"  {name:<5}", style="bold cyan")
    out.append(f"{'solved' if solved else 'not solved':<11}",
               style="green" if solved else "yellow")
    if detail:
        out.append(f"{detail}  ", style="dim")
    if why:
        out.append(f"({why})  ", style="dim italic")
    out.append(f"{TICK} as expected" if as_wanted else f"{CROSS} not what we want",
               style="bold green" if as_wanted else "bold red")
    console.print(out)
    return as_wanted


def prediction_row(x: tuple, want: Any, got: Any, raw: Optional[float] = None) -> bool:
    ok = got == want
    out = Text("    ")
    out.append(f"x={x}", style="cyan")
    out.append("  want ")
    out.append(str(want), style="bold")
    out.append("  got ")
    if raw is not None:
        out.append(f"{raw:.3f} -> ", style="dim")
    out.append(str(got), style="bold")
    out.append(f"  {TICK if ok else CROSS}", style="bold green" if ok else "bold red")
    console.print(out)
    return ok


def not_written(what: str) -> None:
    console.print(Text(f"\n{what}: not written yet", style="dim"))
