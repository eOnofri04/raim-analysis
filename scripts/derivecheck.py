#!/usr/bin/env python3
"""Diff two trees of generated artefacts with a numeric tolerance, and say what moved.

A byte comparison works on the machine the artefacts were exported from and
nowhere else: re-deriving them on a second machine, from the same verdicts under
the same seeds, leaves about half the files differing in the ninth to twelfth
decimal, lbfgs and the correlation sums landing on different last bits under a
different BLAS. Nothing has moved, and a check that cannot say so over half its
inputs is a check that gets ignored.

So the comparison is tolerance-aware. A file is

    IDENTICAL   byte for byte, past any banner --skip-lines drops;
    CLOSE       structurally equal, every number within THRESHOLD;
    DIFFERS     some number outside it, or any non-numeric difference at all.

Only DIFFERS fails, and the three are counted separately: byte-identity is still
worth knowing about, it is just no longer the pass condition.

The tolerance is for the last bits of a float and nothing else. Integers, strings,
booleans, key sets and list lengths are compared exactly -- an `n` that moved, a
condition that flipped from claimcheck to def_on, or a model that dropped out of a
roster is a real change at any threshold, and none of them is a rounding artefact.

Zone 4 is compared the same way, which is what `--patterns` and `--skip-lines` are
for: the generated tables are text carrying the same numbers, and they open with a
provenance banner naming their generator rather than reporting anything.

    derivecheck.py --new /tmp/derive-check                       # vs ../derived
    derivecheck.py --new /tmp/derive-check --threshold 1e-10     # tighter
    derivecheck.py --new /tmp/derive-check -v                    # every CLOSE file
    derivecheck.py --new /tmp/t --ref tables --patterns '*.tex' --skip-lines 1
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Tight enough to catch anything the arithmetic did not perturb, loose enough to
# pass what it did: the observed cross-machine drift tops out around 1e-9, and the
# smallest quantity the paper reports at all -- a third-decimal kappa -- is five
# orders of magnitude coarser.
DEFAULT_THRESHOLD = 1e-8

# How many differences to print per file before summarising the rest. A file whose
# every number moved says the same thing in its first four lines as in its four
# hundredth.
MAX_NOTES = 4

# Numbers as the derived .md summaries write them: sign, digits, optional decimal
# point, optional exponent.
NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")


def deviation(a: float, b: float) -> float:
    """|a - b| in units of the allowance: absolute below 1, relative above it.

    Kappa, balanced accuracy and every interval bound live in [-1, 1], where
    scaling by magnitude would make the tolerance meaninglessly tight near zero;
    the stacker's coefficients and the pooled row counts run larger, where a fixed
    absolute tolerance would be meaninglessly loose. max(1, |a|, |b|) is absolute
    for the former and relative for the latter, which is what lets one THRESHOLD
    cover the whole tree.
    """
    if a == b:
        return 0.0
    # json.load accepts the non-standard NaN that json.dump writes, and the
    # diversity sweeps carry it (a correlation against a constant column). Two
    # NaNs agree here; reporting them as a difference would fail eight files on
    # every run, none of them for a reason.
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if math.isinf(a) or math.isinf(b) or math.isnan(a) or math.isnan(b):
        return math.inf
    return abs(a - b) / max(1.0, abs(a), abs(b))


def _at(where: str, key: object) -> str:
    return f"{where}.{key}" if where else str(key)


def _short(x: object, width: int = 48) -> str:
    s = repr(x)
    return s if len(s) <= width else s[: width - 1] + "…"


class Comparison:
    """One file's verdict: the worst numeric deviation, and the hard differences.

    `hard` is the pass/fail bit. It is set by anything a tolerance must not
    excuse, which is everything except a float within THRESHOLD. `structural`
    narrows that: it marks a difference no threshold could ever excuse, which is
    what separates "this run needs a looser tolerance" from "this run needs
    looking at".
    """

    def __init__(self, tol: float) -> None:
        self.tol = tol
        self.worst = 0.0
        self.worst_at = ""
        self.notes: list[str] = []
        self.suppressed = 0
        self.hard = False
        self.structural = False

    def fail(self, where: str, msg: str, numeric: bool = False) -> None:
        self.hard = True
        self.structural |= not numeric
        if len(self.notes) < MAX_NOTES:
            self.notes.append(f"{where or '<root>'}: {msg}")
        else:
            self.suppressed += 1

    def number(self, where: str, a: float, b: float) -> None:
        d = deviation(float(a), float(b))
        if d > self.worst:
            self.worst, self.worst_at = d, where
        if d > self.tol:
            self.fail(where, f"{a!r} vs {b!r}   deviation {d:.2e} > {self.tol:.0e}",
                      numeric=True)


def walk_json(c: Comparison, where: str, a: object, b: object) -> None:
    """Compare two parsed JSON values, recording differences into `c`."""
    if isinstance(a, dict) and isinstance(b, dict):
        # Key ORDER is not content: json.dump writes insertion order, so a
        # generator that assembles the same mapping in a different sequence has
        # changed no number. Key SETS are content -- a field that appeared or
        # vanished is a change to what the artefact records.
        for k in a:
            if k not in b:
                c.fail(_at(where, k), "present in the new tree only")
        for k in b:
            if k not in a:
                c.fail(_at(where, k), "present in the reference only")
        for k in a:
            if k in b:
                walk_json(c, _at(where, k), a[k], b[k])
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            c.fail(where, f"list of {len(a)} vs {len(b)}")
            return
        for i, (u, v) in enumerate(zip(a, b)):
            walk_json(c, f"{where}[{i}]", u, v)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            c.fail(where, f"{a!r} vs {b!r}")
    elif isinstance(a, int) and isinstance(b, int):
        # Denominators, fold and seed indices, K, n. Exact or nothing: an n of 222
        # against 221 is a different measurement, not a rounding artefact.
        if a != b:
            c.fail(where, f"{a} vs {b}   (integer)")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        c.number(where, a, b)
    elif type(a) is not type(b):
        c.fail(where, f"{type(a).__name__} vs {type(b).__name__}")
    elif a != b:
        c.fail(where, f"{_short(a)} vs {_short(b)}")


def walk_text(c: Comparison, new: str, ref: str, skip: int = 0) -> None:
    """Compare two text files line by line, numbers to tolerance.

    Zone 3's `_transfer/lodo_summary.md`, and zone 4's generated tables and
    reports, all take this path. Their numbers are printed to three or four
    decimals and so mostly survive the drift byte for byte -- but they come from
    the same fits as everything else, a rounding boundary can always fall the other
    way, and there is no reason for them to be the artefacts a tightened threshold
    cannot speak about.
    """
    a_lines, b_lines = new.splitlines()[skip:], ref.splitlines()[skip:]
    if len(a_lines) != len(b_lines):
        c.fail("", f"{len(a_lines)} lines vs {len(b_lines)}")
    for i, (x, y) in enumerate(zip(a_lines, b_lines), start=skip + 1):
        if x == y:
            continue
        # The prose between the numbers must match exactly; the numbers themselves
        # go to tolerance. If the prose moved, the line is a real difference and
        # pairing its numbers up would be meaningless anyway.
        if NUMBER.split(x) != NUMBER.split(y):
            c.fail(f"line {i}", f"{_short(x, 64)} vs {_short(y, 64)}")
            continue
        for j, (u, v) in enumerate(zip(NUMBER.findall(x), NUMBER.findall(y)), start=1):
            if u != v:
                c.number(f"line {i}, number {j}", float(u), float(v))


def compare_file(new: Path, ref: Path, tol: float, skip: int = 0) -> Comparison:
    c = Comparison(tol)
    if new.suffix == ".json":
        try:
            a, b = json.loads(new.read_text()), json.loads(ref.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            c.fail("", f"unreadable: {e}")
            return c
        walk_json(c, "", a, b)
    else:
        try:
            walk_text(c, new.read_text(), ref.read_text(), skip)
        except UnicodeDecodeError:
            # A binary artefact whose bytes differ -- the fast path established that
            # much. There is no tolerance to apply and no field to name, so say so
            # plainly rather than traceback: point --patterns at a directory holding a
            # PDF or a PNG and this is the honest answer, and figcheck.py is the tool
            # for the figures, where a byte difference means nothing at all.
            c.fail("", f"binary content differs ({new.stat().st_size} vs "
                       f"{ref.stat().st_size} bytes)")
    return c


def payload(path: Path, skip: int) -> bytes:
    """The file's bytes, less any skipped leading lines.

    The fast path has to skip what the slow path skips, or a table whose only
    difference is its banner would be reported as "within tolerance" -- true, but a
    worse answer than "identical", which is what the reader is entitled to.
    """
    b = path.read_bytes()
    return b if not skip else b"".join(b.splitlines(keepends=True)[skip:])


def inventory(root: Path, patterns: tuple[str, ...]) -> set[str]:
    return {
        str(p.relative_to(root))
        for pat in patterns
        for p in root.rglob(pat)
        if p.is_file()
    }


def profile(devs: list[float]) -> str:
    """Decade histogram of the per-file worst deviation.

    This is what THRESHOLD should be chosen from: a run whose mass sits at 1e-12
    and a run whose mass sits at 1e-6 both report "within tolerance" at 1e-8, and
    only the first of them is arithmetic.
    """
    buckets: Counter[str] = Counter()
    for d in devs:
        if d == 0:
            buckets["exact"] += 1
        elif math.isinf(d):
            buckets["inf"] += 1
        else:
            # Nearest decade rather than an upper bound: the drift clusters *on* the
            # decades (1e-12 for the pooled fits, 1e-9 for the worst of them), and
            # rounding up would file a deviation of 1.0000002e-12 under 1e-11.
            buckets[f"~1e{round(math.log10(d)):d}"] += 1

    def order(k: str) -> float:
        return {"exact": -math.inf, "inf": math.inf}.get(k, float(k[3:]))

    return "  ".join(f"{k}: {n}" for k, n in sorted(buckets.items(), key=lambda kv: order(kv[0])))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Diff a regenerated zone-3 tree against the committed one, "
                    "tolerating float drift and nothing else.")
    ap.add_argument("--new", required=True, type=Path,
                    help="the freshly regenerated tree (DERIVE_OUT of a derive.sh run)")
    ap.add_argument("--ref", type=Path, default=HERE.parent / "derived",
                    help="the committed tree to compare against (default ../derived)")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help="largest tolerated deviation, absolute below 1 and "
                         f"relative above it (default {DEFAULT_THRESHOLD:.0e})")
    ap.add_argument("--exempt", nargs="*", default=[], metavar="REL",
                    help="paths not expected in both trees, on either side: a "
                         "hand-authored table, a report written from outside this "
                         "repository, or a table only one of the two trees carries")
    ap.add_argument("--patterns", nargs="*", default=["*.json", "*.md"], metavar="GLOB",
                    help="which files to compare (default *.json *.md)")
    ap.add_argument("--skip-lines", type=int, default=0, metavar="N",
                    help="ignore the first N lines of every non-JSON file: the "
                         "generated tables' provenance banner names their generator "
                         "and reports no number (zone 4 uses 1)")
    ap.add_argument("--top", type=int, default=5, metavar="N",
                    help="how many files to name per list: the closest to the "
                         "threshold, the underived, the uncommitted (default 5)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="name every within-tolerance file, not just the closest --top")
    args = ap.parse_args()

    if not args.new.is_dir():
        print(f"!! no such tree: {args.new}", file=sys.stderr)
        return 2
    if not args.ref.is_dir():
        print(f"!! no such reference tree: {args.ref}", file=sys.stderr)
        return 2

    patterns = tuple(args.patterns)
    produced, committed = inventory(args.new, patterns), inventory(args.ref, patterns)

    identical, close, differs = 0, [], []
    for rel in sorted(produced & committed):
        new, ref = args.new / rel, args.ref / rel
        skip = 0 if new.suffix == ".json" else args.skip_lines
        if payload(new, skip) == payload(ref, skip):
            identical += 1
            continue
        c = compare_file(new, ref, args.threshold, skip)
        (differs if c.hard else close).append((rel, c))

    # A file on one side only fails too, in both directions and for the same reason:
    # a committed artefact the run did not regenerate is unverified, and a
    # regenerated artefact nothing committed means the frozen set is incomplete.
    # Legitimate asymmetries are declared with --exempt rather than tolerated.
    only_new = sorted(rel for rel in produced - committed if rel not in args.exempt)
    only_ref = sorted(rel for rel in committed - produced if rel not in args.exempt)

    banner = f", past the first {args.skip_lines} line(s)" if args.skip_lines else ""
    print(f"==> comparing {len(produced & committed)} file(s) against {args.ref} "
          f"at threshold {args.threshold:.0e}{banner}")

    differs.sort(key=lambda rc: rc[1].worst, reverse=True)
    for rel, c in differs:
        print(f"  DIFFERS     {rel}")
        for note in c.notes:
            print(f"                {note}")
        if c.suppressed:
            print(f"                … and {c.suppressed} further difference(s)")
    # Capped: pointing this at a single-dataset derive run is an easy mistake, and it
    # puts every other committed file on one side only. Two hundred lines saying so
    # bury the DIFFERS lines above, which are the ones worth reading.
    for rel in only_new[: args.top]:
        print(f"  ONLY IN NEW {rel}  (in {args.new}, absent from {args.ref})")
    if len(only_new) > args.top:
        print(f"  ONLY IN NEW … and {len(only_new) - args.top} more")
    for rel in only_ref[: args.top]:
        print(f"  ONLY IN REF {rel}  (in {args.ref}, absent from {args.new})")
    if len(only_ref) > args.top:
        print(f"  ONLY IN REF … and {len(only_ref) - args.top} more "
              f"(a partial run? derive.sh skips the cross-dataset steps for a single dataset)")

    print(f"\n  {identical} identical, {len(close)} within tolerance, "
          f"{len(differs)} differ, {len(only_ref)} only in the reference, "
          f"{len(only_new)} only in the new tree")

    # What it would take to pass, when the only thing standing in the way is the
    # threshold. Stated as a fact about the run and not as a recommendation: a
    # figure well above the arithmetic drift means a number moved, and the answer to
    # that is to find out which, not to widen the gate until it fits.
    if (differs and not only_new and not only_ref
            and not any(c.structural for _, c in differs)):
        need = max(c.worst for _, c in differs)
        print(f"  every failure is numeric; the run would pass at any threshold "
              f"above {need:.2e} — check what moved before adopting one")

    if close:
        print(f"  deviation profile (within tolerance):  {profile([c.worst for _, c in close])}")
        ranked = sorted(close, key=lambda rc: rc[1].worst, reverse=True)
        shown = ranked if args.verbose else ranked[: args.top]
        label = ("every within-tolerance file, closest first" if args.verbose
                 else f"closest {len(shown)} of {len(ranked)} to the threshold")
        print(f"  {label}:")
        for rel, c in shown:
            print(f"    {c.worst:.2e}  {rel}  ({c.worst_at or 'formatting only'})")

    return 1 if differs or only_ref or only_new else 0


if __name__ == "__main__":
    sys.exit(main())
