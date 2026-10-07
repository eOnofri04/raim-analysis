#!/usr/bin/env python
"""The interval convention every reported interval uses.

`_ci` turns a list of bootstrap resamples into `[point, lo, hi]`: the resample
mean and the 95% percentile interval, nan resamples dropped. Each generator that
reports an interval draws its own paired cluster-bootstrap resamples -- whole
`row_id`s with replacement, so that the two instances a source row yields
("<row>:pos" and "<row>:neg") are resampled together, at the B and seed
derive.sh passes -- and all of them summarise the resamples through this one
function. `raim-verdicts/raim/ci.py` restates it for the measurement box, and
the two are kept identical by hand.
"""
from __future__ import annotations

import numpy as np


def _ci(samples, lo=2.5, hi=97.5):
    a = np.asarray([s for s in samples if s == s])  # drop nan
    if a.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    return float(a.mean()), float(np.percentile(a, lo)), float(np.percentile(a, hi))
