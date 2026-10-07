#!/usr/bin/env python
"""Paired cluster-bootstrap of the stacked panel against a frontier judge.

The panel-vs-single win rule (weighted.py) turns on the PAIRED difference of the
kappas resampled together by row_id; this script applies the same standard to
the panel-vs-frontier comparison, so both rest on one test rather than the panel
using the paired difference and the frontier marginal-interval overlap.

The stacked panel is rebuilt from the frozen aggregator (raim_lib.weighting.crossfit,
5-fold cross-fit, seed 0, each dataset's own condition -- def_on on the QA-form
sets, claimcheck on the four LLM-AggreFact sets), exactly as weighted.py builds
it; the frontier judge is read from verdicts/<ds>/judge_<tag>.jsonl. We resample
row_id clusters (B=2000, seed 0), recompute each side's kappa on the items
covered by BOTH, and read the 95% interval of the difference (panel - frontier)
via bootstrap._ci. Items where the frontier abstained are dropped pairwise (the
stacker never abstains). Intervals follow the [point, lo, hi] convention.

Reads  verdicts/<ds>/experiment.jsonl and verdicts/<ds>/judge_<tag>.jsonl;
writes derived/<ds>/panel_vs_<tag>.json.

Usage:
  python panel_vs_frontier.py                        # Sonnet, all eight datasets
  python panel_vs_frontier.py --frontier sonnet --datasets medhallu ragtruth
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

# raim_lib sits one level up; put the repository root on the path so this
# script runs the same whether driven by make or invoked directly.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from raim_lib.verdicts import load as load_verdicts
from raim_lib.weighting import crossfit
from raim_lib.bootstrap import _ci
import os as _os

HERE = Path(__file__).resolve().parent
# RUNS names the DERIVED tree (zone 3). Verdict files (.jsonl) are addressed
# through the same root and re-pointed to verdicts/ by raim_lib.verdicts, so
# nothing here can write into the measurement mirror by mistyping a path.
RUNS = Path(_os.environ.get("RAIM_DERIVED", HERE.parent / "derived"))
ALL_DATASETS = ["medhallu", "ragtruth", "aggrefact_xsum", "aggrefact_wice",
                "aggrefact_cnn", "aggrefact_expertqa", "truthfulqa", "factscore"]


def load_panel(ds, folds=5, seed=0):
    """Rebuild the frozen stacked panel exactly as weighted.py does."""
    rows = load_verdicts(RUNS / ds / "experiment.jsonl")
    conds = {r["condition"] for r in rows}
    cond = "def_on" if "def_on" in conds else sorted(conds)[0]
    rows = [r for r in rows if r["condition"] == cond]
    models = sorted({r["model"] for r in rows})
    votes = {m: {} for m in models}
    gold, rowid_of = {}, {}
    for r in rows:
        votes[r["model"]][r["uid"]] = r["pred"]
        gold[r["uid"]] = r["gold"]
        rowid_of[r["uid"]] = r["row_id"]
    uids = sorted(gold)
    preds, _ = crossfit(votes, gold, uids, rowid_of, folds, seed)
    return cond, preds["stacked"], gold, rowid_of, uids


def load_frontier(ds, tag):
    g, pred = {}, {}
    for d in load_verdicts(RUNS / ds / f"judge_{tag}.jsonl"):
        g[d["uid"]] = d["gold"]
        pred[d["uid"]] = d["pred"]
    return g, pred


def _bacc(yt, yp):
    """Balanced accuracy, i.e. the mean of the two per-class recalls.

    Written out rather than taken from sklearn so that the empty-class case is
    ours to define: inside the bootstrap a resample can in principle leave a class
    unrepresented, and sklearn would emit a warning and a nan. The caller already
    skips resamples with a single gold class, so both denominators are non-zero
    here; the guard is kept so a future caller cannot get a silent nan.
    """
    yt = np.asarray(yt)
    yp = np.asarray(yp)
    recalls = []
    for c in (0, 1):
        m = yt == c
        if m.sum() == 0:
            continue
        recalls.append(float((yp[m] == c).mean()))
    return float(np.mean(recalls)) if recalls else float("nan")


def pair_one(ds, tag, B=2000, seed=0):
    cond, panel, gold, rowid_of, uids = load_panel(ds, seed=seed)
    fg, fp = load_frontier(ds, tag)
    shared = [u for u in uids if fp.get(u) is not None and panel.get(u) is not None]
    assert all(gold[u] == fg[u] for u in shared), f"{ds}: gold mismatch with frontier"

    row_to_uids = defaultdict(list)
    for u in shared:
        row_to_uids[rowid_of[u]].append(u)
    rids = sorted(row_to_uids)
    R = len(rids)

    rng = np.random.default_rng(seed)
    kp, kf, diff = [], [], []
    bp, bf, bdiff = [], [], []
    for _ in range(B):
        pick = rng.integers(0, R, size=R)
        sel = [u for p in pick for u in row_to_uids[rids[p]]]
        yt = [gold[u] for u in sel]
        if len(set(yt)) < 2:
            continue
        yp = [panel[u] for u in sel]
        yf = [fp[u] for u in sel]
        a = cohen_kappa_score(yt, yp)
        b = cohen_kappa_score(yt, yf)
        kp.append(a)
        kf.append(b)
        diff.append(a - b)
        # Balanced accuracy on the same resample, so the companion metric of
        # \S metrics carries an interval from the same clusters as kappa rather
        # than being read off two marginal estimates. Both sides are full-coverage
        # on `shared` by construction (the stacker never abstains and frontier
        # abstentions were dropped pairwise above), so no abstention charge enters.
        ab, bb = _bacc(yt, yp), _bacc(yt, yf)
        bp.append(ab)
        bf.append(bb)
        bdiff.append(ab - bb)

    d = _ci(diff)
    db = _ci(bdiff)
    verdict = ("panel_beats_frontier" if d[1] > 0 else
               "frontier_beats_panel" if d[2] < 0 else "tie")
    out = dict(dataset=ds, condition=cond, frontier=tag, n_shared=len(shared),
               B=B, seed=seed, kappa_panel=list(_ci(kp)),
               kappa_frontier=list(_ci(kf)),
               diff_panel_minus_frontier=list(d),
               bacc_panel=list(_ci(bp)), bacc_frontier=list(_ci(bf)),
               diff_bacc_panel_minus_frontier=list(db), verdict=verdict)
    path = RUNS / ds / f"panel_vs_{tag}.json"
    json.dump(out, path.open("w"), indent=2)
    sig = " *" if verdict != "tie" else ""
    print(f"[{ds:<18}] panel-{tag} = {d[0]:+.3f} "
          f"[{d[1]:+.3f},{d[2]:+.3f}]{sig}  ({verdict})  "
          f"| bacc {db[0]:+.3f} [{db[1]:+.3f},{db[2]:+.3f}]  -> {path.name}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frontier", default="sonnet",
                    help="frontier judge tag, i.e. verdicts/<ds>/judge_<tag>.jsonl")
    ap.add_argument("--datasets", nargs="*", default=ALL_DATASETS)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for ds in args.datasets:
        pair_one(ds, args.frontier, B=args.B, seed=args.seed)


if __name__ == "__main__":
    main()
