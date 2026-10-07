#!/usr/bin/env python
"""Base-vs-instruct probe: does panel error-correlation reflect a shared answer-bias
induced by alignment, or merely shared task-difficulty?

For each dataset we hold the underlying models fixed and compare two matched 8-member
panels -- base (pre-instruction-tuning) and instruct -- under the v3 constrained-
decoding scorer. The statistic is the mean pairwise *error-correlation* (phi = Pearson
on per-item error indicators) within each panel; if instruction tuning installs a
shared bias, the instruct panel's errors should be MORE correlated. The paired
difference (instruct - base) is tested by a cluster-bootstrap over row_id.

--exclude drops the named members from BOTH panels, keeping them matched; it is the
robustness check of app:base that sets aside the instruct member collapsing to a
single class under the constrained scorer (--exclude falcon). The table is written
for the full panels only.

Reads verdicts/lp_probe/<ds>_lp{base,inst}_v3_panel.jsonl; emits a console report and
tables/tab_baseprobe.tex.

Usage:  python analyze_base_probe.py   [--B 2000] [--out <dir>] [--exclude falcon]
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from paper_tables import DATASETS, HEADER  # display names + tex header (source of truth)

# raim_lib sits one level up; put the repository root on the path so this
# script runs the same whether driven by make or invoked directly.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from raim_lib.verdicts import exists as verdict_exists
from raim_lib.verdicts import load as load_verdicts
import os as _os

HERE = Path(__file__).resolve().parent
# RUNS names the DERIVED tree (zone 3). Verdict files (.jsonl) are addressed
# through the same root and re-pointed to verdicts/ by raim_lib.verdicts, so
# nothing here can write into the measurement mirror by mistyping a path.
RUNS = Path(_os.environ.get("RAIM_DERIVED", HERE.parent / "derived"))
REV = "v3"
# probe order: core first, then secondary grounded, then ungrounded (as in DATASETS)
DS_ORDER = list(DATASETS.keys())


def load_panel(path, condition=None):
    """Read a probe panel at its own recorded condition.

    `condition=None` resolves the frame from the file rather than assuming one,
    since the four LLM-AggreFact sets are scored under claim-support and the rest
    under def_on. A panel is single-frame by construction (tools/assemble_panel.py gates on it), so
    resolving it here is a read of the data, not a guess about it.
    """
    rows = list(load_verdicts(path))
    if condition is None:
        seen = {r.get("condition") for r in rows}
        if len(seen) != 1:
            raise SystemExit(f"{path}: expected one condition, found {sorted(seen)}")
        condition = seen.pop()
    votes, gold, rowid = defaultdict(dict), {}, {}
    for r in rows:
        if r.get("condition") != condition:
            continue
        votes[r["model"]][r["uid"]] = r["pred"]
        gold[r["uid"]] = r["gold"]
        rowid[r["uid"]] = r["row_id"]
    return votes, gold, rowid


def err_matrix(votes, gold, uids):
    models = sorted(votes)
    E = np.full((len(models), len(uids)), np.nan)
    for i, m in enumerate(models):
        vm = votes[m]
        for j, u in enumerate(uids):
            v = vm.get(u)
            if v is not None:
                E[i, j] = float(v != gold[u])
    return E


def mean_phi(E, cols):
    """Mean pairwise error-correlation over the given item columns."""
    sub = E[:, cols]
    if np.isnan(sub).any():  # nan-safe pairwise fallback (not needed at full coverage)
        n = sub.shape[0]; vals = []
        for a in range(n):
            for b in range(a + 1, n):
                x, y = sub[a], sub[b]; mask = ~(np.isnan(x) | np.isnan(y))
                if mask.sum() < 5:
                    continue
                xa, yb = x[mask], y[mask]
                vals.append(0.0 if (xa.std() == 0 or yb.std() == 0)
                            else float(np.corrcoef(xa, yb)[0, 1]))
        return float(np.mean(vals)) if vals else float("nan")
    with np.errstate(invalid="ignore"):
        C = np.corrcoef(sub)
    iu = np.triu_indices(C.shape[0], 1)
    return float(np.nanmean(C[iu]))


def drop(votes, exclude):
    """The panel without the members whose id contains any of `exclude` (any case)."""
    return {m: v for m, v in votes.items()
            if not any(x.lower() in m.lower() for x in exclude)}


def analyse(ds, B, seed=0, exclude=()):
    vb, gb, rb = load_panel(RUNS / "lp_probe" / f"{ds}_lpbase_{REV}_panel.jsonl")
    vi, gi, ri = load_panel(RUNS / "lp_probe" / f"{ds}_lpinst_{REV}_panel.jsonl")
    if exclude:
        vb, vi = drop(vb, exclude), drop(vi, exclude)
    uids = sorted(set(gb) & set(gi))                      # shared items (paired test)
    Eb, Ei = err_matrix(vb, gb, uids), err_matrix(vi, gi, uids)
    cols_all = np.arange(len(uids))
    pt_b, pt_i = mean_phi(Eb, cols_all), mean_phi(Ei, cols_all)

    row_to_cols = defaultdict(list)
    for j, u in enumerate(uids):
        row_to_cols[rb[u]].append(j)
    rids = sorted(row_to_cols)
    rng = np.random.default_rng(seed)
    bs_b, bs_i, bs_d = [], [], []
    for _ in range(B):
        pick = rng.integers(0, len(rids), size=len(rids))
        cols = np.array([j for p in pick for j in row_to_cols[rids[p]]], dtype=int)
        b, i = mean_phi(Eb, cols), mean_phi(Ei, cols)
        bs_b.append(b); bs_i.append(i); bs_d.append(i - b)
    ci = lambda pt, s: [pt, float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
    return dict(n=len(uids), members=Eb.shape[0],
                base=ci(pt_b, bs_b), inst=ci(pt_i, bs_i),
                delta=ci(pt_i - pt_b, bs_d))


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default=str(HERE / "tables"))
    ap.add_argument("--exclude", nargs="*", default=[], metavar="MEMBER",
                    help="drop these members (id substrings) from both panels; "
                         "console report only, the table stays the full panels'")
    args = ap.parse_args()

    res = {}
    dropped = f", excluding {', '.join(args.exclude)}" if args.exclude else ""
    print(f"Base-vs-instruct mean pairwise error-correlation (each dataset at its own frame, v3, B={args.B}{dropped})\n")
    print(f"  {'dataset':<11} {'n':>5} {'base phi':>22} {'instruct phi':>22} "
          f"{'delta (inst-base)':>24}")
    for ds in DS_ORDER:
        if not verdict_exists(RUNS / "lp_probe" / f"{ds}_lpbase_{REV}_panel.jsonl"):
            print(f"  {DATASETS[ds][0]:<11}  (missing)"); continue
        r = analyse(ds, args.B, exclude=args.exclude); res[ds] = r
        sig = "*" if (r["delta"][1] > 0 or r["delta"][2] < 0) else " "
        print(f"  {DATASETS[ds][0]:<11} {r['n']:>5}  {fmt(r['base']):>20}  "
              f"{fmt(r['inst']):>20}  {fmt(r['delta']):>22}{sig}")

    sigpos = [DATASETS[d][0] for d in res if res[d]["delta"][1] > 0]
    signeg = [DATASETS[d][0] for d in res if res[d]["delta"][2] < 0]
    print(f"\n  instruct MORE correlated (delta>0, CI excl. 0): "
          f"{', '.join(sigpos) or 'none'}")
    print(f"  instruct LESS correlated (delta<0, CI excl. 0): "
          f"{', '.join(signeg) or 'none'}")

    if args.exclude:
        print("\n  (--exclude given: tab_baseprobe.tex is left as the full panels')")
        return

    # LaTeX table
    rows = []
    for ds in DS_ORDER:
        if ds not in res:
            continue
        r = res[ds]; d = r["delta"]
        star = "\\,*" if (d[1] > 0 or d[2] < 0) else ""
        cell = lambda t: f"${t[0]:+.3f}$\\,{{\\scriptsize[{t[1]:+.2f},\\,{t[2]:+.2f}]}}"
        rows.append(f"{DATASETS[ds][0]} & {r['n']} & {cell(r['base'])} & "
                    f"{cell(r['inst'])} & ${d[0]:+.3f}${star}\\,"
                    f"{{\\scriptsize[{d[1]:+.2f},\\,{d[2]:+.2f}]}} \\\\")
    body = "\n".join(rows)
    tex = HEADER + r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{5pt}
\begin{tabular}{l r l l l}
\toprule
Dataset & $n$ & base panel $\phi$ & instruct panel $\phi$ & $\Delta\phi$ (instruct $-$ base) \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\caption{Base-versus-instruct probe: mean pairwise error-correlation $\phi$ within the matched eight-member base and instruct panels (each dataset under its own scoring frame, \S\ref{ssec:data}; v3 constrained decoding, $95\%$ paired cluster-bootstrap CIs over the clustering unit of~\S\ref{ssec:metrics}).
A positive $\Delta\phi$ means instruction tuning makes the panel's errors \emph{more} correlated, consistent with a shared answer-bias rather than shared task-difficulty; $*$ marks a CI excluding zero.}
\label{tab:baseprobe}
\end{table*}
"""
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "tab_baseprobe.tex").write_text(tex)
    print(f"\nwrote {Path(args.out) / 'tab_baseprobe.tex'}")


if __name__ == "__main__":
    main()
