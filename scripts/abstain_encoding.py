#!/usr/bin/env python
r"""Does encoding an abstention as a CATEGORY, rather than as the scalar 0.5,
move the stacked panel?

WHY THIS EXISTS. The protocol imputes a missing member vote to 0.5 and feeds it
to the meta-learner as a continuous feature, so the linear term beta_i * v_i
asserts that an abstention lies exactly midway in log-odds between "supported"
(0) and "hallucinated" (1). Abstentions are, however, categorical failures -- an
unparseable generation, a safety refusal, a context overflow -- rather than a
50/50 probabilistic midpoint, so the imputation could skew the fitted
coefficients. This measures whether it does.

THE ARMS differ in the feature map and in NOTHING else. They share the fold
assignment, the seed, the item set, the estimator and its penalty, the 0.5
decision threshold and the degenerate-fold fallback, because all three run
through the same raim_lib.weighting.crossfit via its `featurise` hook rather
than through a second copy of it.

  imputed      1 column  per member: v in {0, 0.5, 1}                (the paper)
  categorical  2 columns per member: 1[v=1], 1[v=abstain]; v=0 is the
                                     reference level -- the literal one-hot
  indicator    2 columns per member: the committed vote (abstention filled 0)
                                     and 1[v=abstain] -- the standard
                                     missing-indicator idiom

The last two span the same column space, being linear reparameterisations of
one 3-level factor per member, so they can differ only through the L2 penalty's
basis-dependence. Both are reported on purpose: if they agree, a departure
from the baseline is a property of the encoding; if they disagree with each
other, it is an artefact of the penalty and not evidence about abstentions.

WHAT BOUNDS THE ANSWER. Abstentions are rare (some 2.1% of votes across the
suite) and unevenly spread, so the encoding cannot bite where there is nothing
to encode; the per-dataset abstention count is reported beside every delta. The
categorical arms also carry twice the parameters (20 features against 10), so on
the smallest sets a worsening may be variance from the larger model rather than
evidence that 0.5 was right. Both readings are stated with the numbers.

THE SAFEGUARD. The `imputed` arm must reproduce the committed
derived/<ds>/weighted.json stacked kappa (to within 5e-4; it does so exactly).
It is checked on every dataset and a mismatch is fatal: if the harness cannot reproduce the paper, no
contrast computed in it is worth reading.

    abstain_encoding.py              # -> derived/_summary/abstain_encoding.json
    abstain_encoding.py --tex        # + tables/tab_abstain.tex (or --texdir)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from raim_lib.verdicts import load as load_verdicts
from raim_lib.verdicts import DERIVED, VERDICTS
from raim_lib.weighting import crossfit

from paper_tables import pick_condition, TEXNAME

DATASETS = ["medhallu", "ragtruth", "aggrefact_wice", "aggrefact_xsum",
            "aggrefact_cnn", "aggrefact_expertqa", "factscore", "truthfulqa"]
ARMS = ["imputed", "categorical", "indicator"]


# ---------------------------------------------------------------- feature maps
def feat_imputed(votes, models, uu):
    """The paper's map, reproduced here only so the arms read alike; crossfit
    takes this same path when `featurise` is left None."""
    return np.array([[0.5 if votes[m].get(u) is None else votes[m][u]
                      for m in models] for u in uu], float)


def feat_categorical(votes, models, uu):
    """One-hot over {0, 1, abstain} per member, with 0 as the reference level."""
    rows = []
    for u in uu:
        r = []
        for m in models:
            v = votes[m].get(u)
            r += [1.0 if v == 1 else 0.0, 1.0 if v is None else 0.0]
        rows.append(r)
    return np.array(rows, float)


def feat_indicator(votes, models, uu):
    """Committed vote with the abstention filled to 0, plus a missingness flag."""
    rows = []
    for u in uu:
        r = []
        for m in models:
            v = votes[m].get(u)
            r += [0.0 if v is None else float(v), 1.0 if v is None else 0.0]
        rows.append(r)
    return np.array(rows, float)


FEATURISERS = {"imputed": None,            # None -> crossfit's own path, verbatim
               "categorical": feat_categorical,
               "indicator": feat_indicator}


# ---------------------------------------------------------------------- scoring
def kap(preds, gold, sel):
    yt, yp = [], []
    for u in sel:
        p = preds.get(u)
        if p is not None:
            yt.append(gold[u]); yp.append(p)
    if len(yt) < 2 or len(set(yt)) < 2:
        return float("nan")
    return cohen_kappa_score(yt, yp)


def bacc(preds, gold, sel):
    """Balanced accuracy charging a non-answer as an error, as in weighted.py."""
    tp = fn_p = tn = fn_n = 0
    for u in sel:
        p = preds.get(u)
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


def _ci(samples):
    a = np.asarray([s for s in samples if s == s])
    if a.size == 0:
        return (float("nan"),) * 3
    return float(a.mean()), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def run(ds, folds, seed, B):
    rows = list(load_verdicts(VERDICTS / ds / "experiment.jsonl"))
    cond = pick_condition({r["condition"] for r in rows})
    votes, gold, rowid = {}, {}, {}
    for r in rows:
        if r.get("condition") != cond:
            continue
        votes.setdefault(r["model"], {})[r["uid"]] = r["pred"]
        gold[r["uid"]] = r["gold"]
        rowid[r["uid"]] = r["row_id"]
    uids = sorted(gold)
    models = sorted(votes)

    n_abstain = sum(votes[m].get(u) is None for m in models for u in uids)
    n_votes = len(models) * len(uids)

    # One fit per arm. Same folds and seed, so the three are paired item by item.
    preds = {}
    for arm in ARMS:
        p, _ = crossfit(votes, gold, uids, rowid, folds=folds, seed=seed,
                        featurise=FEATURISERS[arm])
        preds[arm] = p["stacked"]

    # Paired cluster bootstrap over row_ids, the frozen protocol of weighted.py.
    row_to_uids = defaultdict(list)
    for u in uids:
        row_to_uids[rowid[u]].append(u)
    row_ids = sorted(row_to_uids)
    R = len(row_ids)
    rng = np.random.default_rng(seed)
    acc = defaultdict(list)
    contrasts = [("categorical", "imputed"), ("indicator", "imputed"),
                 ("categorical", "indicator")]
    for _ in range(B):
        pick = rng.integers(0, R, size=R)
        sel = [u for i in pick for u in row_to_uids[row_ids[i]]]
        k = {a: kap(preds[a], gold, sel) for a in ARMS}
        b = {a: bacc(preds[a], gold, sel) for a in ARMS}
        for a in ARMS:
            acc[("k", a)].append(k[a]); acc[("ba", a)].append(b[a])
        for a, ref in contrasts:
            acc[("d", a, ref)].append(k[a] - k[ref])
            acc[("dba", a, ref)].append(b[a] - b[ref])

    # How often do the two arms disagree on an item at all?
    flips = {f"{a}-{ref}": sum(preds[a][u] != preds[ref][u] for u in uids)
             for a, ref in contrasts}

    return dict(
        dataset=ds, condition=cond, n=len(uids), clusters=R, B=B,
        folds=folds, seed=seed, models=models,
        n_abstentions=n_abstain, n_votes=n_votes,
        abstention_rate=n_abstain / n_votes,
        kappa={a: _ci(acc[("k", a)]) for a in ARMS},
        bacc={a: _ci(acc[("ba", a)]) for a in ARMS},
        diffs={f"{a}-{ref}": _ci(acc[("d", a, ref)]) for a, ref in contrasts},
        bacc_diffs={f"{a}-{ref}": _ci(acc[("dba", a, ref)]) for a, ref in contrasts},
        point_kappa={a: kap(preds[a], gold, uids) for a in ARMS},
        flipped_items=flips)


def write_tex(res, texdir):
    """Emit tab_abstain from the JSON, so the table cannot drift from the run."""
    texdir.mkdir(parents=True, exist_ok=True)
    L = [f"% AUTO-GENERATED by {Path(__file__).name} in scripts/ "
         f"-- do not edit by hand; regenerate with `make tables`.",
         r"\begin{table}[tp]", r"\centering\footnotesize",
         r"\caption{Abstention encoded as a category rather than as the scalar $0.5$.",
         r"The two arms differ in the meta-learner's feature map alone, sharing "
         r"folds, seed, items, estimator, penalty and decision threshold.",
         r"Intervals are $95\%$ paired cluster-bootstrap intervals over $B=2000$ resamples of "
         r"\texttt{row\_id} (\S\ref{ssec:metrics}).}",
         r"\label{tab:abstain}",
         r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{l r r r r r c r}", r"\toprule",
         r"& & \multicolumn{2}{c}{\emph{abstentions}} "
         r"& \multicolumn{2}{c}{\emph{stacked} \kp} & & \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
         r"Dataset & $n$ & count & rate & $0.5$ & categ. "
         r"& $\Delta\kp$ \emph{(categ.\ $-$ $0.5$)} & moved \\",
         r"\midrule"]
    for ds in DATASETS:
        r = res[ds]
        d = r["diffs"]["categorical-imputed"]
        L.append(
            f"{TEXNAME[ds]} & {r['n']} & {r['n_abstentions']} & "
            f"${100*r['abstention_rate']:.1f}\\%$ & "
            f"${r['point_kappa']['imputed']:.3f}$ & "
            f"${r['point_kappa']['categorical']:.3f}$ & "
            f"${d[0]:+.3f}$ $[{d[1]:+.3f}, {d[2]:+.3f}]$ & "
            f"{r['flipped_items']['categorical-imputed']} \\\\")
    tot_a = sum(res[d]["n_abstentions"] for d in DATASETS)
    tot_v = sum(res[d]["n_votes"] for d in DATASETS)
    tot_n = sum(res[d]["n"] for d in DATASETS)
    tot_m = sum(res[d]["flipped_items"]["categorical-imputed"] for d in DATASETS)
    # The suite rate is what app:coverage quotes ("some 2.1% of the ten-judge
    # votes"), and without this row a reader could only check it by summing the
    # eight above. The kappa and delta columns do not aggregate, so they are left
    # unset rather than filled with a meaningless mean.
    def _sep(x):
        return f"{x:,}".replace(",", "{,}")
    L += [r"\midrule",
          r"\emph{suite} & $" + _sep(tot_n) + r"$ & $" + _sep(tot_a) + r"$ & "
          + f"${100*tot_a/tot_v:.2f}\\%$" + r" & --- & --- & --- & " + str(tot_m) + r" \\",
          r"\bottomrule", r"\end{tabular}",
          r"\tablelegend{$0.5$: the protocol's encoding, one column per member "
          r"($v_i \in \{0, 0.5, 1\}$);",
          r"\emph{categ.}: the categorical arm, two indicators "
          r"per member ($\mathbf{1}[v_i = 1]$ and $\mathbf{1}[v_i = \mathrm{abstain}]$, "
          r"with $v_i = 0$ the reference level), so twenty coefficients in place of ten.",
          r"\emph{count} and \emph{rate} give the abstentions among that dataset's "
          r"ten-judge votes, and \emph{moved} the items on which the two arms return "
          r"different verdicts.",
          r"The \emph{suite} row totals the eight, its rate being " + _sep(tot_a)
          + r" of " + _sep(tot_v) + r" votes; the \kp\ and $\Delta\kp$ columns do "
          r"not aggregate and are left unset there.}",
          r"\end{table}"]
    (texdir / "tab_abstain.tex").write_text("\n".join(L) + "\n")
    print(f"wrote {texdir / 'tab_abstain.tex'}")


def check_reproduction(res):
    """The imputed arm must equal the committed weighted.json, or nothing here holds."""
    bad = []
    for ds, r in res.items():
        p = DERIVED / ds / "weighted.json"
        if not p.exists():
            bad.append(f"{ds}: no committed weighted.json to check against")
            continue
        ref = json.load(p.open())["kappa"]["stacked"][0]
        got = r["kappa"]["imputed"][0]
        if abs(ref - got) > 5e-4:
            bad.append(f"{ds}: imputed arm {got:.6f} != committed {ref:.6f}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tex", action="store_true")
    ap.add_argument("--texdir", default=str(Path(__file__).resolve().parent / "tables"),
                    help="where --tex writes tab_abstain.tex (default tables/; "
                         "`make check` redirects it)")
    a = ap.parse_args()

    res = {ds: run(ds, a.folds, a.seed, a.B) for ds in DATASETS}

    bad = check_reproduction(res)
    print("\n=== reproduction check: imputed arm vs committed weighted.json ===")
    if bad:
        for b in bad:
            print("  FAIL", b)
        raise SystemExit("reproduction check failed; refusing to report contrasts")
    print("  all eight datasets reproduce the committed stacked kappa\n")

    hdr = (f"{'dataset':20s} {'n':>5s} {'abst':>6s} {'rate':>6s} "
           f"{'imputed':>8s} {'categ.':>8s} {'indic.':>8s} "
           f"{'D(categ-imp)':>26s} {'flips':>6s}")
    print(hdr); print("-" * len(hdr))
    for ds in DATASETS:
        r = res[ds]
        d = r["diffs"]["categorical-imputed"]
        print(f"{ds:20s} {r['n']:5d} {r['n_abstentions']:6d} "
              f"{100*r['abstention_rate']:5.1f}% "
              f"{r['point_kappa']['imputed']:8.3f} "
              f"{r['point_kappa']['categorical']:8.3f} "
              f"{r['point_kappa']['indicator']:8.3f} "
              f"  {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] "
              f"{r['flipped_items']['categorical-imputed']:6d}")

    out = Path(a.out) if a.out else DERIVED / "_summary" / "abstain_encoding.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")
    if a.tex:
        write_tex(res, Path(a.texdir))


if __name__ == "__main__":
    main()
