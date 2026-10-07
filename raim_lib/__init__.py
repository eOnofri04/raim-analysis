"""Shared analysis library: the aggregator, the interval convention, and verdict loading.

What belongs here is what more than one generator would use, and what would be
dangerous to have two copies of:

  weighting.py  the cross-fitted stacker and the aggregation ladder -- the method
                itself, and the one module the project's rules say must never be
                reinvented.
  bootstrap.py  the percentile interval `_ci` over bootstrap resamples, in the
                project's `[point, lo, hi]` convention, which every generator
                reports its paired cluster-bootstrap intervals through.
  verdicts.py   reading verdict files, plain or gzipped, and the boundary between
                the verdict zone and the derived zone.
  mirror.py     the verification of the committed verdict mirror against its
                manifest, and the release digest that names it.

What does not belong here is anything that encodes presentation choices -- the
dataset registry, the display names, the table layouts. Those live beside the
generators that use them, in scripts/, so that changing a layout never means
editing the method.
"""
