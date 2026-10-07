#!/usr/bin/env python
"""Cross-dataset transfer: the loaders the leave-one-dataset-out test shares.

`load` reads one verdict file's per-model votes under its own scoring frame, and
`feat` builds the 0.5-imputed meta-learner design matrix; transfer_lodo.py imports
both, so its transferred and in-domain stackers see identical features.
"""
from __future__ import annotations

import numpy as np

# raim_lib sits one level up; put the repository root on the path so this
# script runs the same whether driven by make or invoked directly.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from raim_lib.verdicts import load as load_verdicts


def _frame_rows(path, condition=None):
    """The rows of a single scoring frame: the caller's, else the file's own
    (def_on if present, else the sole frame, e.g. claimcheck)."""
    rows = load_verdicts(path)
    if condition is None:
        conds = {r.get("condition") for r in rows}
        condition = "def_on" if "def_on" in conds else sorted(conds)[0]
    return [r for r in rows if r.get("condition") == condition]


def load(path, condition=None):
    """Load per-model votes. condition=None auto-detects the file's own frame
    (def_on if present, else the sole frame, e.g. claimcheck) so a pooled/LODO fit
    over a mixed-frame suite scores each dataset under its appropriate frame."""
    votes, gold, rowid = {}, {}, {}
    for r in _frame_rows(path, condition):
        votes.setdefault(r["model"], {})[r["uid"]] = r["pred"]
        gold[r["uid"]] = r["gold"]
        rowid[r["uid"]] = r["row_id"]
    return votes, gold, rowid




def feat(src, models, uids):
    """Meta-learner design matrix, missing -> 0.5, columns in `models` order.
    `src` maps model -> uid -> value: the binary votes of load().
    """
    return np.array([[0.5 if src[m].get(u) is None else src[m][u]
                      for m in models] for u in uids], float)


