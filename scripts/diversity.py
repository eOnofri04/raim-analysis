#!/usr/bin/env python
"""Panel-size sweep under stacking.

Adds judges one at a time; at each K, cross-fits the stacker
(raim_lib.weighting.crossfit), bootstraps its kappa by row_id, and records the
mean / max pairwise *error-correlation* among the current members. Two outputs:

  - the K-curve (stacked kappa and mean error-correlation as K grows): does the
    next member help, plateau, or hurt?
  - the pairwise error-correlation matrix at K_max: which members actually
    decorrelate?

Inputs:
  --experiment verdicts/<ds>/experiment.jsonl     # the panel
  --candidates verdicts/<ds>/judge_<tag>.jsonl .. # optional extra members, one model each

Order modes:
  given  -- cli order (the panel first, then candidates as listed)
  acc    -- best individual balanced-accuracy first (greedy by competence)
  decorr -- start with most accurate, then greedily add the one that minimises
            mean error-correlation with the current set (greedy by independence)

derive.sh runs it with --order decorr at the protocol's B = 2000; those are the
intervals fig_budget(a) shades.
"""
from __future__ import annotations
import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score

# raim_lib sits one level up; put the repository root on the path so this
# script runs the same whether driven by make or invoked directly.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from raim_lib.verdicts import load as load_verdicts
from raim_lib.weighting import crossfit
from raim_lib.bootstrap import _ci


def load_votes(paths, condition="def_on"):
    votes, gold, rowid = {}, {}, {}
    for p in paths:
        for r in load_verdicts(p):
            if r.get("condition") != condition:
                continue
            votes.setdefault(r["model"], {})[r["uid"]] = r["pred"]
            gold[r["uid"]] = r["gold"]
            rowid[r["uid"]] = r["row_id"]
    return votes, gold, rowid


def err_indicator(votes_m, gold, uids):
    """1 if wrong, 0 if right, NaN if abstain."""
    e = np.full(len(uids), np.nan)
    for i, u in enumerate(uids):
        v = votes_m.get(u)
        if v is not None:
            e[i] = float(v != gold[u])
    return e


def pair_corr(a, b):
    """Pearson (= phi for binary) on error indicators; NaN-safe."""
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 5:
        return float("nan")
    aa, bb = a[m], b[m]
    if aa.std() == 0 or bb.std() == 0:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def model_acc(votes_m, gold, uids):
    yt, yp = [], []
    for u in uids:
        v = votes_m.get(u)
        if v is not None:
            yt.append(gold[u]); yp.append(v)
    if len(set(yt)) < 2:
        return 0.5
    return balanced_accuracy_score(yt, yp)


def order_models(votes, gold, uids, models, mode, errs):
    if mode == "given":
        return models
    accs = {m: model_acc(votes[m], gold, uids) for m in models}
    by_acc = sorted(models, key=lambda m: -accs[m])
    if mode == "acc":
        return by_acc
    if mode == "decorr":
        ordered = [by_acc[0]]
        remaining = [m for m in models if m != by_acc[0]]
        while remaining:
            scored = [(m, np.nanmean([pair_corr(errs[m], errs[o])
                                      for o in ordered])) for m in remaining]
            best = min(scored, key=lambda x: (x[1] if x[1] == x[1] else 1.0))[0]
            ordered.append(best); remaining.remove(best)
        return ordered
    raise ValueError(mode)


def kappa_ci(method, preds, gold, uids, rowid, B, seed):
    row_to_uids = defaultdict(list)
    for u in uids:
        row_to_uids[rowid[u]].append(u)
    rids = sorted(row_to_uids)
    rng = np.random.default_rng(seed)
    ks = []
    for _ in range(B):
        pick = rng.integers(0, len(rids), size=len(rids))
        sel = [u for p in pick for u in row_to_uids[rids[p]]]
        yt, yp = [], []
        for u in sel:
            p = preds[method].get(u)
            if p is not None:
                yt.append(gold[u]); yp.append(p)
        if len(set(yt)) >= 2:
            ks.append(cohen_kappa_score(yt, yp))
    return _ci(ks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True,
                    help="raw experiment JSONL (e.g. verdicts/medhallu_experiment.jsonl)")
    ap.add_argument("--candidates", nargs="*", default=[],
                    help="zero or more candidate JSONLs (from scripts/run_judge.py --tag)")
    ap.add_argument("--order", default="decorr",
                    choices=["given", "decorr", "acc"])
    ap.add_argument("--condition", default="def_on")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    votes, gold, rowid = load_votes([args.experiment] + args.candidates,
                                    args.condition)
    if not votes:
        raise SystemExit(f"no rows found at condition={args.condition!r}")
    uids = sorted(gold)
    models = list(votes)
    errs = {m: err_indicator(votes[m], gold, uids) for m in models}

    ordered = order_models(votes, gold, uids, models, args.order, errs)
    print(f"order ({args.order}, {len(ordered)} models):")
    for i, m in enumerate(ordered, 1):
        print(f"  {i:2d}. {m}  (bal_acc={model_acc(votes[m], gold, uids):.3f})")

    sweep = []
    for K in range(1, len(ordered) + 1):
        sub = ordered[:K]
        votes_sub = {m: votes[m] for m in sub}
        preds, _ = crossfit(votes_sub, gold, uids, rowid,
                            folds=args.folds, seed=args.seed)
        k_stacked = kappa_ci("stacked", preds, gold, uids, rowid,
                             args.B, args.seed)
        k_unweighted = kappa_ci("unweighted", preds, gold, uids, rowid,
                                args.B, args.seed) if K >= 2 else k_stacked
        if K >= 2:
            cors = [pair_corr(errs[a], errs[b])
                    for a, b in itertools.combinations(sub, 2)]
            mean_corr = float(np.nanmean(cors))
            max_corr = float(np.nanmax(cors))
        else:
            mean_corr = max_corr = float("nan")
        sweep.append(dict(K=K, members=sub, stacked_kappa=list(k_stacked),
                          unweighted_kappa=list(k_unweighted),
                          mean_corr=mean_corr, max_corr=max_corr))
        print(f"  K={K}: stacked κ={k_stacked[0]:.3f} "
              f"[{k_stacked[1]:.3f},{k_stacked[2]:.3f}]  "
              f"mean_corr={mean_corr:.3f}  max_corr={max_corr:.3f}")

    n = len(ordered)
    M = np.zeros((n, n))
    for i, a in enumerate(ordered):
        for j, b in enumerate(ordered):
            M[i, j] = 1.0 if i == j else pair_corr(errs[a], errs[b])

    out = dict(order=args.order, condition=args.condition,
               final_models=ordered, sweep=sweep, corr_matrix=M.tolist())
    if args.out:
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")
    _plot(sweep, ordered, M,
          args.title or Path(args.experiment).stem,
          str(Path(args.out).with_suffix(".png")) if args.out else "diversity.png")


def _plot(sweep, models, M, title, png):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    Ks = [r["K"] for r in sweep]
    ks = np.array([r["stacked_kappa"][0] for r in sweep])
    lo = np.array([r["stacked_kappa"][1] for r in sweep])
    hi = np.array([r["stacked_kappa"][2] for r in sweep])
    ax1.errorbar(Ks, ks, yerr=[ks - lo, hi - ks], fmt="o-",
                 capsize=4, label="stacked κ vs gold (95% CI)", color="#9467bd")
    uw = np.array([r["unweighted_kappa"][0] for r in sweep])
    ax1.plot(Ks, uw, "s--", color="#888", alpha=0.7, label="unweighted κ (ref)")
    ax1.set_xlabel("panel size K (added in shown order)")
    ax1.set_ylabel("Cohen's κ vs gold")
    ax1.set_xticks(Ks)
    ax1.set_title(f"{title} — panel-size sweep")
    ax1.legend(loc="lower right", fontsize=8)
    ax1b = ax1.twinx()
    ax1b.plot(Ks, [r["mean_corr"] for r in sweep], "d:",
              color="#d62728", alpha=0.7, label="mean error-corr")
    ax1b.set_ylabel("mean pairwise error-correlation")
    ax1b.legend(loc="upper left", fontsize=8)

    im = ax2.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    short = [m.split("/")[-1][:16] for m in models]
    ax2.set_xticks(range(len(models))); ax2.set_yticks(range(len(models)))
    ax2.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax2.set_yticklabels(short, fontsize=7)
    ax2.set_title("pairwise error-correlation (red = correlated, blue = anti-corr)")
    fig.colorbar(im, ax=ax2, fraction=0.046)
    fig.tight_layout()
    fig.savefig(png, dpi=150); plt.close(fig)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
