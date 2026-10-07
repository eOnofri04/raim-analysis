#!/usr/bin/env python
r"""Item-level test of the competence/correlation account (tab_recovery).

The cross-dataset predictor (analyze_extras.py) relates the stacked-minus-best gap
to a (1-corr)*spread composite over eight datasets, so it has eight points. This
script tests the *mechanism* directly, at the item level.

Conditioning on the items where the dev-selected best single judge (the leader) is
WRONG, the panel can only recover the item from the OTHER members, so recovery is a
non-circular measure of distributed competence. The test is whether, within each
dataset, the panel's recovery probability rises with the fraction of members that
are correct on the item.

Emits tables/tab_recovery.tex and prints the headline statistics. From scripts/, in the README's environment:
    ../.venv/bin/python3 analyze_mechanism.py
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as ss
from sklearn.linear_model import LogisticRegression

# raim_lib sits one level up; put the repository root on the path so this
# script runs the same whether driven by make or invoked directly.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from raim_lib.verdicts import load as load_verdicts
from raim_lib.weighting import crossfit
from paper_tables import TEXNAME as DSNAME
import os as _os

DATASETS = ["medhallu", "ragtruth", "truthfulqa", "factscore",
            "aggrefact_wice", "aggrefact_xsum", "aggrefact_cnn", "aggrefact_expertqa"]
# RUNS names the DERIVED tree (zone 3). Verdict files (.jsonl) are addressed
# through the same root and re-pointed to verdicts/ by raim_lib.verdicts, so
# nothing here can write into the measurement mirror by mistyping a path.
RUNS = Path(_os.environ.get("RAIM_DERIVED",
                            Path(__file__).resolve().parent.parent / "derived"))


def _load(ds):
    rows = load_verdicts(RUNS / ds / "experiment.jsonl")
    # each dataset is scored under its own frame (def_on for the QA-form sets,
    # claimcheck for the LLM-AggreFact sets); mirrors weighted.py.
    conds = {r["condition"] for r in rows}
    cond = "def_on" if "def_on" in conds else sorted(conds)[0]
    votes = defaultdict(dict); gold = {}; rowid = {}
    for r in rows:
        if r["condition"] != cond:
            continue
        votes[r["model"]][r["uid"]] = r["pred"]
        gold[r["uid"]] = r["gold"]; rowid[r["uid"]] = r["row_id"]
    return votes, gold, rowid


def collect():
    """Per leader-wrong item: crowd = fraction of members correct, y = panel recovers.

    `ncorr` counts the members that were right on the item and `nvote` those that
    returned a verdict at all, so the table can be cut by the raw count (which a
    reader can hold in mind) as well as by the fraction the regression uses; the
    two differ only where a member abstained."""
    crowd, recover, dcol, ncorr, nvote = [], [], [], [], []
    per_ds = {}
    for di, ds in enumerate(DATASETS):
        votes, gold, rowid = _load(ds)
        uids = sorted(gold); models = sorted(votes)
        preds, _ = crossfit(votes, gold, uids, rowid, folds=5, seed=0)
        st, bs = preds["stacked"], preds["best_single_cv"]
        c_ds, y_ds = [], []
        for u in uids:
            mv = [votes[m][u] for m in models if votes[m].get(u) is not None]
            b, s = bs.get(u), st.get(u)
            if not mv or b is None or s is None or b == gold[u]:
                continue  # leader-wrong items only
            hits = sum(v == gold[u] for v in mv)
            c = hits / len(mv)
            crowd.append(c); recover.append(int(s == gold[u])); dcol.append(di)
            ncorr.append(hits); nvote.append(len(mv))
            c_ds.append(c); y_ds.append(int(s == gold[u]))
        per_ds[ds] = (np.array(c_ds), np.array(y_ds))
    return (np.array(crowd), np.array(recover), np.array(dcol), per_ds,
            np.array(ncorr), np.array(nvote))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "tables"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    crowd, y, dcol, per_ds, ncorr, nvote = collect()
    n = len(y)

    # pooled correlation
    r_pool, p_pool = ss.pointbiserialr(y, crowd)

    # logistic with dataset fixed effects (crowd + dataset dummies)
    D = np.zeros((n, len(DATASETS))); D[np.arange(n), dcol] = 1
    lr = LogisticRegression(max_iter=1000).fit(np.column_stack([crowd, D[:, 1:]]), y)
    coef = lr.coef_[0][0]

    # within-dataset signs
    within = [ss.pointbiserialr(yy, cc).correlation
              for cc, yy in per_ds.values() if len(yy) > 30 and len(set(yy)) > 1]
    n_pos = sum(w > 0 for w in within)

    # panel (a): one row per COUNT of members correct on the item, which is what a
    # reader can hold in mind and which shows the shape of the rise
    kmax = int(ncorr.max())
    krows = []
    for kk in range(kmax + 1):
        m = ncorr == kk
        if not m.sum():
            continue
        krows.append((kk, float(np.mean(nvote[m])), float(y[m].mean()), int(m.sum())))

    # panel (b): the same effect within each dataset, so "positive within all eight"
    # is auditable rather than asserted
    drows = []
    for ds, (cc, yy) in per_ds.items():
        r_ds = (ss.pointbiserialr(yy, cc).correlation
                if len(yy) > 30 and len(set(yy)) > 1 else float("nan"))
        drows.append((ds, len(yy), float(yy.mean()) if len(yy) else float("nan"), r_ds))

    print(f"item-level recovery test (leader-wrong items, n={n}):")
    print(f"  pooled corr(recover, crowd-correct) = {r_pool:+.3f}  p={p_pool:.1e}")
    print(f"  logistic coef(crowd) w/ dataset FE   = {coef:+.3f} (log-odds)")
    print(f"  positive within-dataset on {n_pos}/{len(within)} datasets (median r={np.median(within):+.2f})")
    for kk, nv, pr, cnt in krows:
        print(f"  {kk} members correct (of {nv:.1f} voting): P(recover)={pr:.3f}  (n={cnt})")
    for ds, cnt, pr, r_ds in drows:
        print(f"  {ds:20} n={cnt:5}  P(recover)={pr:.3f}  r={r_ds:+.3f}")

    kbody = "\n".join(f"{kk} & {pr:.3f} & {cnt} \\\\" for kk, _, pr, cnt in krows)
    dbody = "\n".join(
        f"{DSNAME[ds]} & {cnt} & {pr:.3f} & ${r_ds:+.2f}$ \\\\"
        for ds, cnt, pr, r_ds in drows)
    banner = (f"% AUTO-GENERATED by {Path(_sys.argv[0]).name} in scripts/ "
              f"-- do not edit by hand.\n")
    tex = (banner + r"""\begin{table}[t]
\centering\footnotesize
\caption{Item-level mechanism test on the """
    + f"${n:,}$".replace(",", "{,}") + r""" items on which the CV-best single judge is wrong.
Items are pooled over the eight datasets, so that any recovery must come from the \emph{other} members.
\textbf{(a)} Recovery probability against the number of members individually correct on the item, counting only those that returned a verdict;
\textbf{(b)} the same relation within each dataset.}
\label{tab:recovery}
\begin{minipage}[t]{0.44\linewidth}
\centering
\setlength{\tabcolsep}{5pt}
\begin{tabular}{c c c}
\toprule
members correct & $P(\text{recovers})$ & $n$ \\
\midrule
""" + kbody + r"""
\bottomrule
\end{tabular}
\par\smallskip
{\scriptsize (a) by members correct}
\end{minipage}\hfill
\begin{minipage}[t]{0.52\linewidth}
\centering
\setlength{\tabcolsep}{4pt}
\begin{tabular}{l c c c}
\toprule
Dataset & $n$ & $P(\text{recovers})$ & $r$ \\
\midrule
""" + dbody + r"""
\bottomrule
\end{tabular}
\par\smallskip
{\scriptsize (b) by dataset}
\end{minipage}
\tablelegend{$r$:~correlation between recovery and the fraction of members correct on the item.}
\end{table}
""")
    (out / "tab_recovery.tex").write_text(tex)
    print("wrote tab_recovery.tex")


if __name__ == "__main__":
    main()
