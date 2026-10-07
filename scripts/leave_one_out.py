#!/usr/bin/env python
"""Leave-one-out: does a single dominant judge carry the panel?

Removes each member in turn and recomputes, over the remaining K-1 members,
  - the stacked kappa, and
  - the best_single_cv kappa,
and compares both to the full panel. If dropping the leader costs the panel
little, the judging signal is distributed; if the remaining panel cannot recover
it, the panel is essentially that member plus redundant others.

Both come from raim_lib.weighting.crossfit itself, run on the reduced panel, so
the full-panel figures are weighted.json's.

Reads the per-model votes from an experiment JSONL (condition def_on unless
--condition says otherwise). Pure CPU, no GPU. Cluster-bootstrap CIs by row_id.
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


def load(path, condition="def_on"):
    votes, gold, rowid = {}, {}, {}
    for r in load_verdicts(path):
        if r.get("condition") != condition:
            continue
        votes.setdefault(r["model"], {})[r["uid"]] = r["pred"]
        gold[r["uid"]] = r["gold"]
        rowid[r["uid"]] = r["row_id"]
    return votes, gold, rowid


def crossfit_preds(votes, gold, models, uids, rowid, folds=5, seed=0):
    """Return (stacked_pred_by_uid, best_single_cv_pred_by_uid) for `models`."""
    preds, _ = crossfit({m: votes[m] for m in models}, gold, uids, rowid,
                        folds=folds, seed=seed)
    return preds["stacked"], preds["best_single_cv"]


def kappa_ci(pred, gold, uids, rowid, B, seed):
    r2u = defaultdict(list)
    for u in uids:
        r2u[rowid[u]].append(u)
    rids = sorted(r2u)
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(B):
        sel = [u for p in rng.integers(0, len(rids), len(rids)) for u in r2u[rids[p]]]
        yt = [gold[u] for u in sel if pred.get(u) is not None]
        yp = [pred[u] for u in sel if pred.get(u) is not None]
        if len(set(yt)) >= 2:
            ks.append(cohen_kappa_score(yt, yp))
    return _ci(ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--condition", default="def_on")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    votes, gold, rowid = load(args.experiment, args.condition)
    uids = sorted(gold)
    full = sorted(votes)
    short = {m: m.split("/")[-1][:18] for m in full}

    def run(models):
        st, bc = crossfit_preds(votes, gold, models, uids, rowid, seed=args.seed)
        return (kappa_ci(st, gold, uids, rowid, args.B, args.seed),
                kappa_ci(bc, gold, uids, rowid, args.B, args.seed))

    name = Path(args.experiment).stem
    print(f"\n{name}: full panel K={len(full)}")
    fs, fb = run(full)
    print(f"  FULL      stacked={fs[0]:.3f} [{fs[1]:.3f},{fs[2]:.3f}]   "
          f"best_single_cv={fb[0]:.3f} [{fb[1]:.3f},{fb[2]:.3f}]\n")

    rows = []
    for drop in full:
        rest = [m for m in full if m != drop]
        s, b = run(rest)
        d_stacked = s[0] - fs[0]
        print(f"  drop {short[drop]:<18} stacked={s[0]:.3f} [{s[1]:.3f},{s[2]:.3f}]  "
              f"(Δ vs full {d_stacked:+.3f})   best_single_cv={b[0]:.3f}")
        rows.append(dict(dropped=drop, stacked=list(s), best_single_cv=list(b),
                         delta_stacked_vs_full=d_stacked))

    out = dict(experiment=name, condition=args.condition,
               full_stacked=list(fs), full_best_single_cv=list(fb),
               leave_one_out=rows)
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
