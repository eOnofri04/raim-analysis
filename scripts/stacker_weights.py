"""Record the stacker's own coefficients, per dataset and per member.

The run JSONs carry the stacker's PREDICTIONS but not its weights. This reads the
coefficients out of `raim_lib.weighting.crossfit` itself, via its opt-in
`coef_out` channel, and writes the per-fold mean for every member -- so the
numbers come from the fit that actually produced the paper's stacked verdicts,
rather than from a refit that would restate its fold construction, 0.5-imputation
and estimator settings in a second place.

Output: derived/_summary/stacker_weights.json
    {dataset: {"models": [...], "beta": {model: mean}, "beta_sd": {model: sd},
               "intercept": mean, "kappa": {model: member kappa vs gold}}}

Usage:  ../.venv/bin/python3 stacker_weights.py
"""
from __future__ import annotations
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
from paper_tables import DATASETS, RUNS

FOLDS, SEED = 5, 0
OUT = RUNS / "_summary" / "stacker_weights.json"


def load_panel(ds: str):
    """Votes, gold and row_id clusters for a dataset's analysed condition.

    Mirrors weighted.py's loader, including its condition choice (def_on where
    present, else the single condition the file carries -- claimcheck on the
    LLM-AggreFact sets)."""
    raw = RUNS / ds / "experiment.jsonl"
    rows = load_verdicts(raw)
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
    return models, votes, gold, rowid_of, sorted(gold), cond


def fold_coefficients(models, votes, gold, rowid_of, uids):
    """Per-fold coefficient vectors, from the reported stacker's own fits.

    crossfit is run for its side effect on `coef_out`; its predictions are the
    ones weighted.py already reports and are discarded here. Because it is the
    same call, the folds, the imputation and the estimator cannot diverge from
    the ones behind the paper's numbers."""
    coefs: list[dict] = []
    crossfit(votes, gold, uids, rowid_of, folds=FOLDS, seed=SEED, coef_out=coefs)
    if not coefs:
        return np.empty((0, len(models))), []
    # crossfit sorts its own model list; align to ours so the columns are ours.
    order = [coefs[0]["models"].index(m) for m in models]
    betas = np.array([[c["coef"][i] for i in order] for c in coefs])
    return betas, [c["intercept"] for c in coefs]


def member_kappa(votes_m, gold, uids):
    yt, yp = [], []
    for u in uids:
        p = votes_m.get(u)
        if p is not None:
            yt.append(gold[u]); yp.append(p)
    if len(set(yt)) < 2:
        return float("nan")
    return float(cohen_kappa_score(yt, yp))


def main():
    out = {}
    for ds in DATASETS:
        models, votes, gold, rowid_of, uids, cond = load_panel(ds)
        betas, intercepts = fold_coefficients(models, votes, gold, rowid_of, uids)
        out[ds] = {
            "condition": cond,
            "folds": len(intercepts),
            "models": models,
            "beta": {m: float(betas[:, i].mean()) for i, m in enumerate(models)},
            "beta_sd": {m: float(betas[:, i].std(ddof=0)) for i, m in enumerate(models)},
            "intercept": float(np.mean(intercepts)),
            "kappa": {m: member_kappa(votes[m], gold, uids) for m in models},
        }
        lead = max(out[ds]["beta"], key=lambda m: out[ds]["beta"][m])
        print(f"{DATASETS[ds][0]:11s} cond={cond:11s} "
              f"largest weight {lead.split('/')[-1]:34s} "
              f"beta={out[ds]['beta'][lead]:+.3f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
