#!/usr/bin/env python
"""Best-on-average single judge.

The deployment-honest single-judge baseline is one global, label-based selection --
the member with the highest MEAN Cohen's kappa across the suite -- deployed unchanged
on every dataset (this is the model passed to ``weighted.py --best-on-avg`` by
derive.sh). This script computes the single-member kappa ladder per dataset, each
under its own scoring frame, from the raw votes, and reports the per-tier mean and
the argmax. The selection ranks by the full-suite mean; the same member
(gemma-2-9b-it) also leads on the core four and on the six grounded sets, so the
choice does not depend on the tier.

It also makes visible the related claim that the *per-dataset* best single judge moves
across datasets (so no fixed judge can be picked a priori without labels), by marking
the argmax in each column.

Usage:
  python best_on_average.py                 # console report (+ tables/tab_singles.tex)
  python best_on_average.py --out <dir>
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np

# reuse the source-of-truth registry, kappa, and per-member ladder from paper_tables
from paper_tables import (DATASETS, CORE, GROUNDED, UNGROUNDED, member_kappa, k, HEADER)

# fixed display order: core, then secondary grounded, then ungrounded contrast
DS_ORDER = list(DATASETS.keys())
SHORT = {ds: DATASETS[ds][0] for ds in DS_ORDER}

TIERS = {
    "core grounded (4)": [d for d in DS_ORDER if DATASETS[d][1] == CORE],
    "all grounded (6)":  [d for d in DS_ORDER if DATASETS[d][1] in (CORE, GROUNDED)],
    "full suite (8)":    DS_ORDER,
}
# The selection is ranked over the full suite, as in weighted.py --best-on-avg: it is
# a baseline "deployed everywhere", so the pool it is chosen on is every dataset it
# is deployed on, not the primary denominator of the win count.
PRIMARY_TIER = "full suite (8)"


def short_model(m):
    return m.split("/")[-1]


def compute():
    """Return (models, kappa[model][ds], means[tier][model])."""
    mk = {ds: member_kappa(ds) for ds in DS_ORDER}
    models = sorted({m for ds in DS_ORDER for m in mk[ds]})
    kap = {m: {ds: mk[ds].get(m, float("nan")) for ds in DS_ORDER} for m in models}
    means = {t: {m: float(np.nanmean([kap[m][ds] for ds in tier_ds]))
                 for m in models}
             for t, tier_ds in TIERS.items()}
    return models, kap, means


def report(models, kap, means):
    print("Single-member Cohen's kappa (each dataset under its own frame), per dataset:\n")
    hdr = "  {:<26}".format("model") + "".join(f"{SHORT[ds][:8]:>9}" for ds in DS_ORDER)
    hdr += "".join(f"{t.split()[0][:6]+'.mean':>12}" for t in TIERS)
    print(hdr)
    # rank by the primary tier's mean
    order = sorted(models, key=lambda m: -means[PRIMARY_TIER][m])
    # per-dataset argmax (best single per dataset)
    argmax_ds = {ds: max(models, key=lambda m: (kap[m][ds] if kap[m][ds] == kap[m][ds]
                                                else -9)) for ds in DS_ORDER}
    for m in order:
        row = "  {:<26}".format(short_model(m))
        for ds in DS_ORDER:
            star = "*" if argmax_ds[ds] == m else " "
            row += f"{kap[m][ds]:>8.3f}{star}"
        for t in TIERS:
            row += f"{means[t][m]:>12.3f}"
        print(row)
    print("\nPer-dataset best single (argmax, '*' above):")
    for ds in DS_ORDER:
        print(f"  {SHORT[ds]:<11} {short_model(argmax_ds[ds]):<26} "
              f"kappa={kap[argmax_ds[ds]][ds]:.3f}")
    print("\nBest-on-average single (argmax of mean kappa), per tier:")
    for t in TIERS:
        best = max(models, key=lambda m: means[t][m])
        runner = sorted(models, key=lambda m: -means[t][m])[1]
        print(f"  {t:<20} -> {short_model(best):<22} mean={means[t][best]:.3f}  "
              f"(next: {short_model(runner)} {means[t][runner]:.3f})")




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "tables"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    models, kap, means = compute()
    report(models, kap, means)


if __name__ == "__main__":
    main()
