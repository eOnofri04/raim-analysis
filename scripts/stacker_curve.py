#!/usr/bin/env python
r"""How many gold labels does the stacker itself need?

The cost model rests on the claim that a one-time calibration set, rather than
inference, is the operative expense; this measures how many gold labels the
aggregator needs on its own: fit the stacker on N gold-labelled records and
evaluate out of fold, sweeping N.

PROTOCOL. Every budget draws its labels from the training clusters of the
reported stacker's own folds (those of raim_lib.weighting.crossfit), and every
fit is scored on that fold's held-out clusters, so the evaluation set is the
same at every budget. Scoring each budget on "whatever was not sampled" instead
would make the evaluation set shrink and change composition as N grows, and the
curve would confound learning with a moving denominator -- visibly so for the
label-free majority vote, which has nothing to learn.

Reported beside the stacker, on that same fixed test set:
  best_single   the best member chosen ON the N calibration labels
                -- the label-spending alternative a deployer would otherwise take,
                selected by the rule of weighting.crossfit's best_single_cv
                (balanced accuracy on the calibration items the member answers,
                ties to the first in sorted model order) and scored, like it, on
                the items it commits on; at the full pool it IS best_single_cv

The unweighted majority vote is NOT reported as a curve: it consumes no labels,
so it is a constant in N, and drawing it as a function of budget invited exactly
the misreading above. Its per-dataset value on the same test set is recorded once
as `unweighted_reference`.

At the full pool the script also records the mean of the per-fold kappas, and
its gap to the pooled estimate, since the per-fold mean is the estimator the
pooled one is chosen over (`perfold_gap`, `max_perfold_gap`).

Budgets are capped by the calibration pool, so the per-dataset curves end at
different N and no suite average is taken across them: averaging over whichever
datasets happen to admit a budget makes the mean move with the composition of
that set rather than with the budget. Results are therefore reported per dataset
in absolute kappa.

    stacker_curve.py            # -> derived/_summary/stacker_curve.json
    stacker_curve.py --tex      # + tables/tab_stackercurve.tex
"""
from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from raim_lib.verdicts import load as load_verdicts
from raim_lib.verdicts import DERIVED, VERDICTS
from raim_lib.weighting import _bacc

from paper_tables import DATASETS, pick_condition, TEXNAME

# Budgets are in CLUSTERS -- the unit a deployer actually labels (a matched
# question/answer pair on the QA-form sets, a grounding document on the
# claim-verification ones), and the unit the paper's folds are built on.
BUDGETS = (10, 25, 50, 100, 200, 400, 800)
FOLDS = 5
ROSTER = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
          "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-9b-it",
          "microsoft/Phi-4-mini-instruct", "01-ai/Yi-1.5-9B-Chat-16K",
          "CohereLabs/c4ai-command-r7b-12-2024", "zai-org/glm-4-9b-chat-hf",
          "ibm-granite/granite-3.3-8b-instruct", "tiiuae/Falcon3-7B-Instruct"]


def load_panel(ds):
    rows = load_verdicts(VERDICTS / ds / "experiment.jsonl")
    cond = pick_condition({r["condition"] for r in rows})
    votes, gold, rowid = {}, {}, {}
    for r in rows:
        if r.get("condition") != cond:
            continue
        votes.setdefault(r["model"], {})[r["uid"]] = r["pred"]
        gold[r["uid"]] = r["gold"]
        rowid[r["uid"]] = r["row_id"]
    uids = sorted(gold)
    models = [m for m in ROSTER if m in votes]
    # 0.5-impute a missing vote, exactly as the reported stacker does
    X = np.array([[0.5 if votes[m].get(u) is None else float(votes[m][u])
                   for m in models] for u in uids])
    y = np.array([gold[u] for u in uids], dtype=int)
    clusters = np.array([rowid[u] for u in uids])
    return X, y, clusters, models, cond


def kap(y, p):
    if len(y) < 2 or len(set(y)) < 2 or len(set(p)) < 2:
        return float("nan")
    return float(cohen_kappa_score(y, p))


def best_member(Xt, yt, models):
    """Column of the member weighting.crossfit would pick as best_single_cv on these
    calibration items: highest balanced accuracy over the items it answers (a 0.5
    in X is a missing vote), ties going to the first in sorted model order."""
    items = list(range(len(yt)))
    gold = dict(enumerate(yt.tolist()))
    acc = {}
    for i in range(Xt.shape[1]):
        col = Xt[:, i]
        acc[i] = _bacc({j: (None if col[j] == 0.5 else int(col[j])) for j in items},
                       gold, items)
    return max(sorted(range(len(models)), key=lambda i: models[i]), key=lambda i: acc[i])


TIE_TOL = 1e-9   # a decision margin this close to zero is a tie (see run)


def crossfit_folds(clusters, folds, seed):
    """The fold assignment of raim_lib.weighting.crossfit, reproduced exactly.

    Reusing it rather than inventing a split is what makes this curve converge
    to the number the paper reports: at a budget equal to the training pool the
    estimate IS the reported cross-fitted stacker, on the same folds and the
    same items, which is checked in `saturated_matches_crossfit`.
    """
    row_ids = sorted(set(clusters.tolist()))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(row_ids)
    fold_of = {int(r): i % folds for i, r in enumerate(perm)}
    return np.array([fold_of[int(c)] for c in clusters])


def run(ds, budgets, R, seed, folds=FOLDS):
    X, y, clusters, models, cond = load_panel(ds)
    fold = crossfit_folds(clusters, folds, seed)
    n_clusters = len(set(clusters.tolist()))

    # the label-free floor, scored out-of-fold exactly as everything else is;
    # `>=` sends a tie to hallucinated, as weighting.py's majority does (with 0.5
    # imputation the mean over ten equals the mean over the members that committed)
    unw_oof = np.empty(len(y), dtype=int)
    for f in range(folds):
        te = fold == f
        unw_oof[te] = (X[te].mean(axis=1) >= 0.5).astype(int)
    unw_ref = kap(y, unw_oof)

    pool_sizes = [int(len(set(clusters[fold != f].tolist()))) for f in range(folds)]
    train_clusters = int(np.min(pool_sizes))
    # The final point is the FULL training pool of each fold, not the smallest
    # pool used as a common budget: on datasets whose folds are uneven the latter
    # would fall one cluster short of the reported fit. `None` marks "use
    # everything this fold has", making the saturated point exactly the estimator
    # raim_lib.weighting.crossfit computes.
    grid = [b for b in budgets if b < train_clusters] + [None]

    out = {"dataset": ds, "condition": cond, "n": len(y),
           "n_clusters": n_clusters, "train_clusters": train_clusters,
           "folds": folds, "unweighted_reference": unw_ref, "budgets": []}

    for b in grid:
        # Replicate-major and POOLED, so each replicate yields one kappa computed
        # exactly as raim_lib.weighting.crossfit computes it: every fold predicts
        # its own test items, the predictions are pooled over folds, and kappa is
        # taken once over the whole set. Averaging per-fold kappas instead would be
        # a different and biased estimator, because kappa is not linear in the
        # confusion counts.
        saturated = b is None
        reps = 1 if saturated else R
        ks, kb, degen, perfold = [], [], 0, []
        for r in range(reps):
            ps = np.empty(len(y), dtype=int)
            pb = np.empty(len(y), dtype=int)   # -1 where the chosen member abstains
            bad = False
            for f in range(folds):
                te = fold == f
                pool_cl = np.array(sorted(set(clusters[~te].tolist())))
                rng = np.random.default_rng(
                    [seed, zlib.crc32(ds.encode()), 0 if b is None else b, f, r])
                pick = pool_cl if (saturated or b >= len(pool_cl)) else \
                    rng.choice(pool_cl, size=b, replace=False)
                cal = np.isin(clusters, pick) & (~te)
                Xt, yt = X[cal], y[cal]
                if len(set(yt.tolist())) < 2:
                    bad = True
                    break
                lr = LogisticRegression(max_iter=1000, C=1.0)
                lr.fit(Xt, yt)
                # At the smallest budgets a draw can leave the fit flat (every
                # coefficient zero, or a margin within rounding of zero), and the
                # class of such an item then follows the sign of solver noise, which
                # differs across platforms. A margin within TIE_TOL is therefore a
                # tie, sent to class 0 as LogisticRegression.predict sends an exact
                # one, so the curve is the same on every machine.
                ps[te] = (lr.decision_function(X[te]) > TIE_TOL).astype(int)
                if saturated:
                    perfold.append(kap(y[te], ps[te]))
                col = X[te][:, best_member(Xt, yt, models)]
                pb[te] = np.where(col == 0.5, -1, (col > 0.5).astype(int))
            if bad:
                degen += 1
                continue
            ks.append(kap(y, ps))
            kb.append(kap(y[pb >= 0], pb[pb >= 0]))
        g = lambda v: dict(mean=float(np.nanmean(v)), median=float(np.nanmedian(v)),
                           lo=float(np.nanpercentile(v, 25)),
                           hi=float(np.nanpercentile(v, 75)))
        out["budgets"].append(dict(budget=train_clusters if saturated else b,
                                   fits=len(ks), degenerate=degen,
                                   saturated=saturated,
                                   stacked=g(ks), best_single=g(kb)))
        if saturated and ks:
            out["perfold_mean_kappa"] = float(np.mean(perfold))
            out["perfold_gap"] = abs(out["perfold_mean_kappa"] - ks[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--budgets", nargs="*", type=int, default=list(BUDGETS))
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tex", action="store_true")
    ap.add_argument("--texdir", default=str(Path(__file__).resolve().parent / "tables"))
    args = ap.parse_args()

    per = [run(ds, args.budgets, args.R, args.seed) for ds in args.datasets]

    # Validation: at a budget equal to the training pool this estimator IS the
    # reported cross-fitted stacker. If the two disagree, the protocol has
    # drifted from raim_lib.weighting.crossfit and the curve means nothing.
    from paper_tables import load as _load
    for d in per:
        sat = [r for r in d["budgets"] if r["saturated"]][0]
        d["reported_crossfit_kappa"] = _load(d["dataset"])["weighted"]["kappa"]["stacked"][0]
        d["saturated_kappa"] = sat["stacked"]["mean"]
        d["saturation_gap"] = abs(d["saturated_kappa"] - d["reported_crossfit_kappa"])

    out = dict(seed=args.seed, R=args.R, budgets=args.budgets, folds=FOLDS,
               budget_unit="clusters", per_dataset=per,
               max_saturation_gap=max(d["saturation_gap"] for d in per),
               max_perfold_gap=max(d["perfold_gap"] for d in per))
    path = Path(args.out) if args.out else DERIVED / "_summary" / "stacker_curve.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}\n")

    print(f"{'dataset':20} {'trainCl':>8} {'unw':>7} {'xfit':>7} {'gap':>6}   "
          + "  ".join(f"{b:>7}" for b in args.budgets) + f"  {'full':>7}")
    for d in per:
        cells = {r["budget"]: r["stacked"]["mean"] for r in d["budgets"]}
        line = "  ".join(f"{cells[b]:7.3f}" if b in cells else f"{'--':>7}"
                         for b in args.budgets)
        print(f"{d['dataset']:20} {d['train_clusters']:8d} {d['unweighted_reference']:7.3f} "
              f"{d['reported_crossfit_kappa']:7.3f} {d['saturation_gap']:6.3f}   {line}"
              f"  {d['saturated_kappa']:7.3f}")
    print(f"\nmax |saturated - reported cross-fit| = {out['max_saturation_gap']:.4f}"
          "   (0 means the protocol reproduces the paper's estimator)")
    worst = max(per, key=lambda d: d["perfold_gap"])
    print(f"max |mean of per-fold kappa - pooled| = {out['max_perfold_gap']:.4f}"
          f"   (on {worst['dataset']})")
    if args.tex:
        write_tex(out, Path(args.texdir))


def write_tex(out, texdir):
    texdir.mkdir(parents=True, exist_ok=True)
    budgets = out["budgets"]
    ncol = 3 + len(budgets)
    L = [f"% AUTO-GENERATED by {Path(__file__).name} -- do not edit by hand.",
         r"\begin{table}[tp]", r"\centering\footnotesize",
         r"\caption{Gold-label learning curve of the aggregator: mean Cohen's \kp\ over $R="
         + str(out["R"]) + r"$ draws per budget, budgets being in labelled \emph{records}, "
         r"the clustering unit of~\S\ref{ssec:metrics}. "
         r"Folds are the " + {5: "five"}.get(out["folds"], str(out["folds"]))
         + r" of the reported stacker, the budget drawn from each fold's training clusters and the "
         r"out-of-fold predictions pooled before \kp\ is taken.}",
         r"\label{tab:stackercurve}",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{l rr " + "r" * len(budgets) + "r}", r"\toprule",
         r"Dataset & train & unw. & \multicolumn{" + str(len(budgets) + 1)
         + r"}{c}{\emph{labelled records the stacker is fitted on}} \\",
         r"\cmidrule(lr){4-" + str(ncol + 1) + "}",
         r" & & & " + " & ".join(str(b) for b in budgets) + r" & full \\",
         r"\midrule"]
    for d in out["per_dataset"]:
        cells = {r["budget"]: r["stacked"]["mean"] for r in d["budgets"]}
        row = " & ".join(f"${cells[b]:.3f}$" if b in cells else "---" for b in budgets)
        L.append(f"{TEXNAME[d['dataset']]} & {d['train_clusters']} & "
                 f"${d['unweighted_reference']:.3f}$ & {row} & "
                 f"${d['saturated_kappa']:.3f}$ \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\tablelegend{\emph{train}: smallest per-fold training pool, capping the grid; "
          r"\emph{unw.}: label-free majority vote on the same folds; "
          r"\emph{full}: the whole pool, reproducing the reported cross-fitted value to within "
          + f"{math.ceil(1000 * out['max_saturation_gap']) / 1000:.3f}"  # rounded up: a bound
          + r"; ---: budget beyond that pool.}",
          r"\end{table}"]
    (texdir / "tab_stackercurve.tex").write_text("\n".join(L) + "\n")
    print(f"wrote {texdir / 'tab_stackercurve.tex'}")


if __name__ == "__main__":
    main()
