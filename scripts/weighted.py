#!/usr/bin/env python
"""The aggregation ladder on one dataset: every scheme, with paired intervals.

Reads a panel verdict file (no new inference), learns the weights by leakage-free
K-fold cross-fit calibration (raim_lib.weighting.crossfit), adds the oracle,
best-on-average and average-member baselines, and cluster-bootstraps (by row_id)
each method's kappa and balanced accuracy together with the paired differences
between methods. The paper's decision rule reads two of them:

  - stacked  vs  best_single_cv   (does the panel beat picking the best member on
                                   the calibration folds? -- the win)
  - stacked  vs  oracle_best      (does it beat the best member chosen on all the
                                   data? -- the clean win)

Usage:
  python weighted.py --raw verdicts/medhallu/experiment.jsonl --B 2000 \
      --title "MedHallu (grounded)" --out derived/medhallu/weighted.json
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
from raim_lib.weighting import crossfit, METHODS
from raim_lib.bootstrap import _ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--condition", default=None, help="default: def_on if present")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", default=None, help="PNG path (default: alongside --out)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--judges", nargs="*", default=[],
                    help="judge JSON paths (from scripts/run_judge.py, or a "
                         "scripts/run_judge.sh leg) "
                         "to overlay as reference points on the comparison plot")
    ap.add_argument("--oracle", default=None,
                    help="single model (substring of its id, e.g. 'gemma') to use "
                         "as the ORACLE best single in the paired diffs; default: "
                         "auto-pick the model with the highest full-data kappa. The "
                         "oracle is the deployable best member, chosen on all data — "
                         "the honest target for 'does stacking beat the best single?'")
    ap.add_argument("--best-on-avg", default=None,
                    help="single model (substring of its id, e.g. 'gemma') used as the "
                         "BEST-ON-AVERAGE single: one global, label-based selection "
                         "(the member with the best MEAN kappa across the suite), fixed "
                         "and deployed unchanged on every dataset. Pass the SAME id for "
                         "all datasets. It is the deployment-honest single-judge baseline "
                         "(one selection, no per-dataset peeking); reported as a pseudo-"
                         "method 'best_on_avg_single' with paired diffs. If omitted, only "
                         "the label-free 'average_member' baseline is added.")
    args = ap.parse_args()

    rows = load_verdicts(args.raw)
    conds = {r["condition"] for r in rows}
    cond = args.condition or ("def_on" if "def_on" in conds else sorted(conds)[0])
    rows = [r for r in rows if r["condition"] == cond]
    label = args.title or Path(args.raw).stem

    models = sorted({r["model"] for r in rows})
    votes = {m: {} for m in models}
    gold, rowid_of = {}, {}
    for r in rows:
        votes[r["model"]][r["uid"]] = r["pred"]
        gold[r["uid"]] = r["gold"]
        rowid_of[r["uid"]] = r["row_id"]
    uids = sorted(gold)

    xfit_kw = {}

    preds, mean_w = crossfit(votes, gold, uids, rowid_of, args.folds, args.seed,
                             **xfit_kw)

    # ORACLE best single: the single member with the highest kappa on ALL the data
    # (vs best_single_cv, which is dev-chosen and can be depressed when the leader's
    # margin is small/noisy, e.g. XSum/CNN). Adding it as a pseudo-method lets the
    # paired cluster-bootstrap report stacked - oracle directly, so the "clean vs
    # oracle" claim regenerates from the verdicts.
    def _full_kappa(m):
        yt = [gold[u] for u in uids if votes[m][u] is not None]
        yp = [votes[m][u] for u in uids if votes[m][u] is not None]
        return cohen_kappa_score(yt, yp) if len(set(yt)) > 1 else float("-inf")
    if args.oracle:
        cand = [m for m in models if args.oracle.lower() in m.lower()]
        if len(cand) != 1:
            raise SystemExit(f"--oracle {args.oracle!r} matched {cand}; "
                             f"give a substring that selects exactly one of {models}")
        oracle = cand[0]
    else:
        oracle = max(models, key=_full_kappa)
    preds["oracle_best"] = {u: votes[oracle].get(u) for u in uids}
    report_methods = list(METHODS) + ["oracle_best"]

    # best-on-average single: a fixed, GLOBAL label-based selection (the member with
    # the best MEAN kappa across the suite), deployed unchanged on every dataset --
    # the deployment-honest single-judge baseline (one selection, no per-dataset
    # peeking). Specified by substring; emitted as a pseudo-method like oracle_best.
    boa = None
    if args.best_on_avg:
        cand = [m for m in models if args.best_on_avg.lower() in m.lower()]
        if len(cand) != 1:
            raise SystemExit(f"--best-on-avg {args.best_on_avg!r} matched {cand}; "
                             f"give a substring that selects exactly one of {models}")
        boa = cand[0]
        preds["best_on_avg_single"] = {u: votes[boa].get(u) for u in uids}
        report_methods.append("best_on_avg_single")

    # committed-n per method on the full data: the denominator each method's kappa
    # is actually scored over (kap() drops uids where preds[method] is None). A value
    # below n means the method abstained on some items -- e.g. best_single_cv /
    # oracle_best where the CV-chosen / oracle model failed to parse -- so its kappa
    # is a *different-denominator* contrast against the never-abstaining stacker
    # (which imputes a missing member to 0.5). At full coverage all are == n and the
    # paired diffs are genuinely paired.
    committed = {m: sum(preds[m].get(u) is not None for u in uids)
                 for m in report_methods}
    committed["average_member"] = len(uids)  # an aggregate over members, never abstains

    # cluster bootstrap over row_ids
    row_to_uids = defaultdict(list)
    for u in uids:
        row_to_uids[rowid_of[u]].append(u)
    row_ids = sorted(row_to_uids)
    R = len(row_ids)

    def kap(method, sel):
        yt, yp = [], []
        for u in sel:
            p = preds[method].get(u)
            if p is not None:
                yt.append(gold[u]); yp.append(p)
        if len(yt) < 2 or len(set(yt)) < 2:
            return float("nan")
        return cohen_kappa_score(yt, yp)

    def bacc(method, sel):
        # Balanced accuracy with non-answer (pred is None) counted as ERROR, on the
        # FULL per-class denominator. Complements kap(): kap excuses abstention (drops
        # None, per-method denominator); bacc charges it (None lands outside the
        # numerator but inside the class denominator). The two bracket the abstention
        # treatment, so a method that wins on both is robust to it.
        tp = fn_p = tn = fn_n = 0
        for u in sel:
            p = preds[method].get(u)
            if gold[u] == 1:
                if p == 1: tp += 1
                else:      fn_p += 1     # p in {0, None} -> miss on a positive
            else:
                if p == 0: tn += 1
                else:      fn_n += 1     # p in {1, None} -> miss on a negative
        recs = []
        if tp + fn_p: recs.append(tp / (tp + fn_p))
        if tn + fn_n: recs.append(tn / (tn + fn_n))
        return sum(recs) / len(recs) if recs else float("nan")

    # single-member kappa / bacc on a resample, straight from the raw votes -- used
    # only to build the label-free 'average_member' baseline (mean over members).
    def kap_votes(m, sel):
        yt, yp = [], []
        for u in sel:
            p = votes[m].get(u)
            if p is not None:
                yt.append(gold[u]); yp.append(p)
        if len(yt) < 2 or len(set(yt)) < 2:
            return float("nan")
        return cohen_kappa_score(yt, yp)

    def bacc_votes(m, sel):
        tp = fn_p = tn = fn_n = 0
        for u in sel:
            p = votes[m].get(u)
            if gold[u] == 1:
                if p == 1: tp += 1
                else:      fn_p += 1
            else:
                if p == 0: tn += 1
                else:      fn_n += 1
        recs = []
        if tp + fn_p: recs.append(tp / (tp + fn_p))
        if tn + fn_n: recs.append(tn / (tn + fn_n))
        return sum(recs) / len(recs) if recs else float("nan")

    rng = np.random.default_rng(args.seed)
    acc = defaultdict(list)
    pairs = [("acc_weighted", "unweighted"), ("logodds_weighted", "unweighted"),
             ("stacked", "unweighted"),
             ("acc_weighted", "best_single_cv"),
             ("logodds_weighted", "best_single_cv"),
             ("stacked", "best_single_cv"),
             ("best_single_cv", "unweighted"),
             ("stacked", "oracle_best"),
             ("unweighted", "oracle_best"),
             ("best_single_cv", "oracle_best")]
    if boa:
        pairs += [("stacked", "best_on_avg_single"),
                  ("unweighted", "best_on_avg_single"),
                  ("best_single_cv", "best_on_avg_single")]
    for _ in range(args.B):
        pick = rng.integers(0, R, size=R)
        sel = [u for p in pick for u in row_to_uids[row_ids[p]]]
        k = {m: kap(m, sel) for m in report_methods}
        ba = {m: bacc(m, sel) for m in report_methods}
        # label-free average-member baseline: mean kappa / bacc over the members
        am_k = float(np.nanmean([kap_votes(m, sel) for m in models]))
        am_ba = float(np.nanmean([bacc_votes(m, sel) for m in models]))
        acc[("k", "average_member")].append(am_k)
        acc[("ba", "average_member")].append(am_ba)
        for a in ("stacked", "unweighted"):
            acc[("d", a, "average_member")].append(k[a] - am_k)
            acc[("dba", a, "average_member")].append(ba[a] - am_ba)
        for m in report_methods:
            acc[("k", m)].append(k[m])
            acc[("ba", m)].append(ba[m])
        for a, b in pairs:
            acc[("d", a, b)].append(k[a] - k[b])
            acc[("dba", a, b)].append(ba[a] - ba[b])

    out = dict(condition=cond, folds=args.folds, B=args.B, R=R, n=len(uids),
               models=models, mean_weights=mean_w, oracle_model=oracle,
               best_on_avg_model=boa,
               committed=committed,
               kappa={m: _ci(acc[("k", m)]) for m in report_methods},
               diffs={f"{a}-{b}": _ci(acc[("d", a, b)]) for a, b in pairs},
               bacc={m: _ci(acc[("ba", m)]) for m in report_methods},
               bacc_diffs={f"{a}-{b}": _ci(acc[("dba", a, b)]) for a, b in pairs})
    # the label-free average-member baseline (not a per-item method, so injected here)
    out["kappa"]["average_member"] = _ci(acc[("k", "average_member")])
    out["bacc"]["average_member"] = _ci(acc[("ba", "average_member")])
    for a in ("stacked", "unweighted"):
        out["diffs"][f"{a}-average_member"] = _ci(acc[("d", a, "average_member")])
        out["bacc_diffs"][f"{a}-average_member"] = _ci(acc[("dba", a, "average_member")])

    def fmt(t):
        return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
    print(f"\n[{label}]  condition={cond}  folds={args.folds}  "
          f"clusters={R}  B={args.B}  oracle={oracle.split('/')[-1]}\n")
    am_pairs = [("stacked", "average_member"), ("unweighted", "average_member")]
    print("kappa vs gold (out-of-fold, 95% CI):")
    for m in sorted(out["kappa"], key=lambda m: -out["kappa"][m][0]):
        c = out["committed"].get(m, len(uids))
        flag = "" if c == len(uids) else f"  (abstained {len(uids) - c})"
        print(f"  {m:<18} {out['kappa'][m][0]:.3f} "
              f"[{out['kappa'][m][1]:.3f}, {out['kappa'][m][2]:.3f}]  "
              f"n={c}/{len(uids)}{flag}")
    print("\npaired deltas (CI excluding 0 => significant):")
    for a, b in pairs + am_pairs:
        d = out["diffs"][f"{a}-{b}"]
        sig = "  *" if (d[1] > 0 or d[2] < 0) else ""
        print(f"  {a} - {b:<18} {fmt(d)}{sig}")
    print("\nbalanced accuracy, non-answer = error (full denominator, 95% CI):")
    for m in sorted(out["bacc"], key=lambda m: -out["bacc"][m][0]):
        print(f"  {m:<18} {out['bacc'][m][0]:.3f} "
              f"[{out['bacc'][m][1]:.3f}, {out['bacc'][m][2]:.3f}]")
    print("\npaired bal-acc deltas (CI excluding 0 => significant):")
    for a, b in pairs + am_pairs:
        d = out["bacc_diffs"][f"{a}-{b}"]
        sig = "  *" if (d[1] > 0 or d[2] < 0) else ""
        print(f"  {a} - {b:<18} {fmt(d)}{sig}")
    print("\nmean learned weights (acc_weighted):")
    for m in models:
        print(f"  {m.split('/')[-1]:<28} {mean_w['acc_weighted'][m]:.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")

    png = args.plot or (str(Path(args.out).with_suffix(".png")) if args.out
                        else f"fig_weighted_{label}.png")
    judges = [json.load(open(p)) for p in args.judges]
    _plot(out, label, png, judges=judges)
    print(f"wrote {png}")


def _plot(out, label, png, judges=None):
    import matplotlib.pyplot as plt
    _PANEL_METHODS = ("unweighted", "acc_weighted", "logodds_weighted",
                      "stacked", "best_single_cv")
    # oracle_best lives in the JSON/console only; the figure shows the panel methods
    order = sorted((m for m in out["kappa"] if m in _PANEL_METHODS),
                   key=lambda m: out["kappa"][m][0])
    rows = list(order)
    judges = judges or []
    # judges go at the TOP of the figure (above the panel methods)
    judge_labels = [f"JUDGE: {j['judge'].split('/')[-1]}" for j in judges]
    all_rows = rows + judge_labels
    y = np.arange(len(all_rows))
    means = [out["kappa"][m][0] for m in rows] + [j["kappa"][0] for j in judges]
    err = [[out["kappa"][m][0] - out["kappa"][m][1] for m in rows]
           + [j["kappa"][0] - j["kappa"][1] for j in judges],
           [out["kappa"][m][2] - out["kappa"][m][0] for m in rows]
           + [j["kappa"][2] - j["kappa"][0] for j in judges]]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    colors = {"unweighted": "#888", "acc_weighted": "#1f77b4",
              "logodds_weighted": "#2ca02c", "stacked": "#9467bd",
              "best_single_cv": "#d62728"}
    # panel methods: circles
    ax.errorbar(means[:len(rows)], y[:len(rows)],
                xerr=[err[0][:len(rows)], err[1][:len(rows)]],
                fmt="o", capsize=4, ecolor="#bbb", mfc="k", mec="k")
    # judges: stars (orange), drawn on top
    if judges:
        ax.errorbar(means[len(rows):], y[len(rows):],
                    xerr=[err[0][len(rows):], err[1][len(rows):]],
                    fmt="*", markersize=14, capsize=4, ecolor="#ff7f0e",
                    mfc="#ff7f0e", mec="#cc6600", label="frontier judge")
    for yi, name in zip(y, all_rows):
        col = colors.get(name, "#cc6600" if name.startswith("JUDGE") else "k")
        ax.annotate(name, (means[int(yi)], yi),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=8, color=col)
    bsc = out["kappa"]["best_single_cv"]
    ax.axvline(bsc[0], ls="--", color="#d62728", alpha=0.7,
               label="best single (dev-chosen)")
    ax.axvspan(bsc[1], bsc[2], color="#d62728", alpha=0.10)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel("Cohen's kappa vs gold (out-of-fold)")
    ax.set_title(f"weighted, single and frontier judges ({label})\n"
                 f"95% cluster-bootstrap CI")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(png, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()