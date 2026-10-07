#!/usr/bin/env python
r"""Does the admissibility call, its thresholds frozen, predict the gain on panel
configurations it was never set on?

WHY THIS EXISTS. The rule of app:budget (admissible when s >= N/2 members clear
kappa >= 0.30 and phi <= 0.40) was declared with the eight full-panel results in
view, and eight datasets holding one clear win cannot validate it. What they can
supply is many more *configurations*: every sub-panel of the ten members is a
panel with its own competence spread and error-correlation on the same items, so
the rule, applied unchanged, can be scored against the gain each sub-panel
actually obtains. This separates the two coordinates from dataset identity,
which is the confound behind the eight-point reading. It is out-of-configuration
evidence, not out-of-task evidence: the items, and the members, are shared.

WHAT IS MEASURED. Per dataset, the full ten plus sub-panels of K = 5..9 members
(every subset when there are at most MAXPER, otherwise MAXPER drawn at --seed).
For each sub-panel: the two regime coordinates on the full sample, with
regime_budget's own estimators; the call, s >= ceil(K/2) and phi <= 0.40; and the
cross-fitted stacked kappa minus the CV-best single's and minus the frontier
judge's, with raim_lib.weighting.crossfit, the paper's estimator. Points are
full-sample, without per-configuration intervals; on the full panels they are the
point estimates behind tab_master, which prints the bootstrap resample mean and
differs from them by at most 0.003.

The summary reports, per dataset, the gain by call; the within-dataset contrast
(admitted minus rejected) on the datasets where the call varies; a dataset
fixed-effects regression of the gain on the call and on the continuous
coordinates, with a dataset-level bootstrap; and the pooled shares.

    admissibility_subpanels.py --out derived/_summary/admissibility_subpanels.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

# raim_lib sits one level up; put the repository root on the path so this
# script runs the same whether driven by make or invoked directly.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from raim_lib.verdicts import load as load_verdicts, DERIVED, VERDICTS
from raim_lib.weighting import crossfit

from paper_tables import DATASETS, pick_condition
from regime_budget import coordinates, PHI_FAVOURABLE, ROSTER

MAXPER = 60
KS = range(5, 10)
FRONTIER = "sonnet"


def kap(pred, gold, uids):
    """Cohen's kappa over the items a method committed on."""
    yt = [gold[u] for u in uids if pred.get(u) is not None]
    yp = [pred[u] for u in uids if pred.get(u) is not None]
    return float(cohen_kappa_score(yt, yp))


def load(ds):
    rows = load_verdicts(VERDICTS / ds / "experiment.jsonl")
    cond = pick_condition({r["condition"] for r in rows})
    votes, gold, rowid = {}, {}, {}
    for r in rows:
        if r["condition"] != cond:
            continue
        votes.setdefault(r["model"], {})[r["uid"]] = r["pred"]
        gold[r["uid"]] = r["gold"]
        rowid[r["uid"]] = r["row_id"]
    front = {r["uid"]: r["pred"]
             for r in load_verdicts(VERDICTS / ds / f"judge_{FRONTIER}.jsonl")
             if r.get("condition") == cond}
    return cond, votes, gold, rowid, front


def run_dataset(ds, rng):
    cond, votes, gold, rowid, front = load(ds)
    uids = sorted(gold)
    models = [m for m in ROSTER if m in votes]
    V = np.array([[votes[m].get(u) if votes[m].get(u) is not None else np.nan
                   for u in uids] for m in models], dtype=float)
    y = np.array([gold[u] for u in uids], dtype=float)
    k_front = kap(front, gold, uids)
    subsets = [tuple(range(len(models)))]
    for K in KS:
        allc = list(itertools.combinations(range(len(models)), K))
        if len(allc) > MAXPER:
            allc = [allc[i] for i in sorted(rng.choice(len(allc), MAXPER, replace=False))]
        subsets += allc
    rows = []
    for S in subsets:
        preds, _ = crossfit({models[i]: votes[models[i]] for i in S}, gold, uids, rowid, 5, 0)
        ks, kb = kap(preds["stacked"], gold, uids), kap(preds["best_single_cv"], gold, uids)
        c = coordinates(V[list(S)], y)
        K = len(S)
        rows.append(dict(dataset=ds, condition=cond, K=K, members=[models[i] for i in S],
                         phi=c["phi"], spread=c["spread"], spread_frac=c["spread"] / K,
                         admissible=bool(c["spread"] >= math.ceil(K / 2) and c["phi"] <= PHI_FAVOURABLE),
                         kappa_stacked=ks, kappa_best_cv=kb, kappa_frontier=k_front,
                         gain_best=ks - kb, gain_frontier=ks - k_front))
    return rows


def fe_ols(rows, cols):
    """OLS of gain_best on `cols` after removing each dataset's mean."""
    X, yv = [], []
    for ds in {r["dataset"] for r in rows}:
        sub = [r for r in rows if r["dataset"] == ds]
        A = np.array([[float(r[c]) for c in cols] for r in sub])
        b = np.array([r["gain_best"] for r in sub])
        X.append(A - A.mean(0))
        yv.append(b - b.mean())
    X, yv = np.vstack(X), np.concatenate(yv)
    if np.allclose(X, 0):
        return [float("nan")] * len(cols)
    return [float(v) for v in np.linalg.lstsq(X, yv, rcond=None)[0]]


def fe_boot(rows, cols, B, rng):
    """Dataset-level bootstrap: resample whole datasets, relabelled so a dataset
    drawn twice enters as two fixed effects."""
    names = list(dict.fromkeys(r["dataset"] for r in rows))
    est = []
    for _ in range(B):
        pick = rng.choice(names, len(names), replace=True)
        rr = [dict(r, dataset=f"{d}#{j}") for j, d in enumerate(pick)
              for r in rows if r["dataset"] == d]
        est.append(fe_ols(rr, cols))
    lo, hi = np.nanpercentile(np.array(est, dtype=float), [2.5, 97.5], axis=0)
    return [[float(a), float(b)] for a, b in zip(lo, hi)]


def summarise(rows, B, rng):
    mean = lambda xs, k: float(np.mean([x[k] for x in xs])) if xs else None
    per, contrast = {}, {}
    for ds in dict.fromkeys(r["dataset"] for r in rows):
        sub = [r for r in rows if r["dataset"] == ds]
        a = [r for r in sub if r["admissible"]]
        j = [r for r in sub if not r["admissible"]]
        per[ds] = dict(n=len(sub), n_admitted=len(a),
                       full_panel_admissible=[r for r in sub if r["K"] == len(ROSTER)][0]["admissible"],
                       gain_best_admitted=mean(a, "gain_best"), gain_best_rejected=mean(j, "gain_best"),
                       gain_frontier_admitted=mean(a, "gain_frontier"),
                       gain_frontier_rejected=mean(j, "gain_frontier"))
        if a and j:
            contrast[ds] = dict(gain_best=mean(a, "gain_best") - mean(j, "gain_best"),
                                gain_frontier=mean(a, "gain_frontier") - mean(j, "gain_frontier"))
    adm = [r for r in rows if r["admissible"]]
    rej = [r for r in rows if not r["admissible"]]
    fe = {}
    for name, cols in (("call", ["admissible"]), ("continuous", ["spread_frac", "phi"])):
        fe[name] = dict(columns=cols, coef=fe_ols(rows, cols), ci95=fe_boot(rows, cols, B, rng))
    return dict(
        per_dataset=per, within_dataset_contrast=contrast, fixed_effects=fe,
        pooled=dict(n_admitted=len(adm), n_rejected=len(rej),
                    share_gain_best_positive_admitted=float(np.mean([r["gain_best"] > 0 for r in adm])),
                    share_gain_best_positive_rejected=float(np.mean([r["gain_best"] > 0 for r in rej])),
                    share_frontier_below_010_admitted=float(np.mean([r["gain_frontier"] < -0.10 for r in adm])),
                    share_frontier_below_010_rejected=float(np.mean([r["gain_frontier"] < -0.10 for r in rej]))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    rows = []
    for ds in args.datasets:
        rows += run_dataset(ds, rng)
        full = [r for r in rows if r["dataset"] == ds and r["K"] == len(ROSTER)][0]
        print(f"{ds:20} sub-panels={sum(r['dataset'] == ds for r in rows):4d}  "
              f"full: stacked={full['kappa_stacked']:.3f} gain={full['gain_best']:+.3f} "
              f"admissible={full['admissible']}", flush=True)
    summary = summarise(rows, args.B, np.random.default_rng(args.seed))
    # One row per sub-panel as a compact array under a column header, members as
    # indices into ROSTER, which keeps the 1,888 rows small enough to commit.
    cols = ["dataset", "K", "members", "phi", "spread", "admissible",
            "kappa_stacked", "kappa_best_cv", "kappa_frontier"]
    compact = [[r["dataset"], r["K"], [ROSTER.index(m) for m in r["members"]],
                round(r["phi"], 6), r["spread"], r["admissible"],
                round(r["kappa_stacked"], 6), round(r["kappa_best_cv"], 6),
                round(r["kappa_frontier"], 6)] for r in rows]
    out = dict(seed=args.seed, B=args.B, maxper=MAXPER, K=list(KS), frontier=FRONTIER,
               phi_bar=PHI_FAVOURABLE, spread_rule="s >= ceil(K/2)", roster=ROSTER,
               summary=summary, subpanel_columns=cols)
    path = Path(args.out) if args.out else DERIVED / "_summary" / "admissibility_subpanels.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(out, indent=1)[:-2]
    rows_txt = ",\n".join("  " + json.dumps(c, separators=(",", ":")) for c in compact)
    path.write_text(body + ',\n "subpanels": [\n' + rows_txt + "\n ]\n}\n")
    print(f"\nwrote {path}")
    for ds, c in summary["within_dataset_contrast"].items():
        print(f"  within {ds:20} admitted - rejected: gain_best {c['gain_best']:+.3f}")
    for name, f in summary["fixed_effects"].items():
        print(f"  FE {name:10} " + ", ".join(f"{c} {v:+.4f} [{lo:+.4f}, {hi:+.4f}]"
              for c, v, (lo, hi) in zip(f["columns"], f["coef"], f["ci95"])))
    p = summary["pooled"]
    print(f"  pooled: gain>0 {p['share_gain_best_positive_admitted']:.2f} admitted vs "
          f"{p['share_gain_best_positive_rejected']:.2f} rejected; >0.10 behind the frontier "
          f"{p['share_frontier_below_010_admitted']:.2f} vs {p['share_frontier_below_010_rejected']:.2f}")


if __name__ == "__main__":
    main()
