#!/usr/bin/env python
r"""Where would the admissibility thresholds sit if they were chosen, rather than
declared?

WHY THIS EXISTS. The rule of app:budget (s >= N/2 members at kappa >= 0.30, and
phi <= 0.40) was declared with the eight full-panel results in view. The
sub-panels of admissibility_subpanels.py give 1,888 configurations on which the
two thresholds can instead be *selected* against an outcome fixed in advance,
which answers whether the declared pair is arbitrary. It does not validate the
rule on a new task; the leave-one-dataset-out leg below is the nearest this suite
comes to that.

DESIGN, fixed before the grid was run (2026-09-24).
- Tuned: the count fraction c (s >= ceil(c K)) over C_GRID, and the correlation
  bar t (phi <= t) over T_GRID. Fixed: the competence bar kappa >= 0.30, which
  the whole paper uses.
- Objective: Youden's J of the call against a binary outcome per sub-panel.
  A (main): SAFE, the stacked panel no more than SAFE_MARGIN kappa behind the
  frontier judge -- the rule's operative role as a substitution screen.
  B (fallback): GAIN, the stacked panel above its own CV-best single -- the
  stack-or-select decision app:budget states.
- Reported per objective: the maximum J, every grid point within PLATEAU of it,
  J at the declared pair, and the full-panel calls under the maximisers.
- Leave one dataset out: tune on the other seven datasets' sub-panels, then read
  the held-out dataset's full-panel call under every fold maximiser.

Added after the grid was read, as properties of the whole family rather than of
any optimum: the share of grid points that split the panels at which the call is
informative (J > 0), on the sub-panels and on the eight full panels, for margins
of 0.05, 0.10 and 0.15 kappa under A and for B; and the interval of competence
bars kappa_0 over which the eight full-panel calls are those of the declared
instance, the other two thresholds held at it.

    admissibility_tuning.py --out derived/_summary/admissibility_tuning.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from raim_lib.verdicts import DERIVED

C_GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
# 0.25 .. 0.90. Declared as 0.25 .. 0.60; widened after the first run put
# objective A's maximum on the 0.60 edge, so that the edge is not the answer.
T_GRID = [round(0.25 + 0.01 * i, 2) for i in range(66)]
DECLARED = (0.5, 0.40)
SAFE_MARGIN = 0.10
PLATEAU = 0.01
OBJECTIVES = {
    "A_safe": lambda r: r["kappa_stacked"] - r["kappa_frontier"] >= -SAFE_MARGIN,
    "B_gain": lambda r: r["kappa_stacked"] - r["kappa_best_cv"] > 0,
}


def load(path):
    d = json.loads(Path(path).read_text())
    cols = d["subpanel_columns"]
    return [dict(zip(cols, row)) for row in d["subpanels"]]


def call(r, c, t):
    return r["spread"] >= math.ceil(c * r["K"] - 1e-9) and r["phi"] <= t


def youden(rows, c, t, outcome):
    tp = fn = tn = fp = 0
    for r in rows:
        y, p = outcome(r), call(r, c, t)
        tp += y and p; fn += y and not p; tn += (not y) and (not p); fp += (not y) and p
    if tp + fn == 0 or tn + fp == 0:
        return float("nan")
    return tp / (tp + fn) + tn / (tn + fp) - 1


def grid(rows, outcome):
    return {(c, t): youden(rows, c, t, outcome) for c in C_GRID for t in T_GRID}


def maximisers(J, tol=0.0):
    best = max(v for v in J.values() if v == v)
    return best, sorted(k for k, v in J.items() if v == v and v >= best - tol - 1e-12)


def full_calls(rows, c, t):
    return {r["dataset"]: bool(call(r, c, t)) for r in rows if r["K"] == 10}


def informative_share(rows, outcome):
    """Share of grid points splitting `rows` at which Youden's J is positive."""
    J = [youden(rows, c, t, outcome) for c in C_GRID for t in T_GRID
         if 0 < sum(call(r, c, t) for r in rows) < len(rows)]
    J = [j for j in J if j == j]
    return dict(points=len(J), share_positive=float(np.mean(np.array(J) > 0)), J_min=float(min(J)))


def kappa0_interval(datasets, step=0.001):
    """Competence bars at which the eight full-panel calls equal the declared
    instance's, count and correlation bar held at it; with the members at the edges."""
    from regime_budget import load_panel, coordinates, member_kappas
    K, P = {}, {}
    for ds in datasets:
        V, y, _, _, _ = load_panel(ds)
        K[ds] = np.array(member_kappas(V, y)[0])
        P[ds] = coordinates(V, y)["phi"]
    n = 10
    calls = lambda k0: tuple(bool((K[d] >= k0).sum() >= math.ceil(DECLARED[0] * n) and P[d] <= DECLARED[1])
                             for d in datasets)
    ref = calls(0.30)
    grid = np.round(np.arange(0.0, 0.8 + step / 2, step), 6)
    ok = [k0 for k0 in grid if calls(k0) == ref]
    lo = hi = 0.30
    while calls(round(lo - step, 6)) == ref:
        lo = round(lo - step, 6)
    while calls(round(hi + step, 6)) == ref:
        hi = round(hi + step, 6)
    return dict(lo=float(lo), hi=float(hi), step=step, contiguous_points=int(sum(lo <= k <= hi for k in ok)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=str(DERIVED / "_summary" / "admissibility_subpanels.json"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = load(args.inp)
    datasets = list(dict.fromkeys(r["dataset"] for r in rows))
    declared_calls = full_calls(rows, *DECLARED)

    out = dict(design=dict(c_grid=C_GRID, t_grid=T_GRID, declared=DECLARED,
                           safe_margin=SAFE_MARGIN, plateau=PLATEAU,
                           competence_bar=0.30, n_subpanels=len(rows)),
               declared_full_calls=declared_calls, objectives={})
    for name, outcome in OBJECTIVES.items():
        J = grid(rows, outcome)
        best, argmax = maximisers(J)
        _, plateau = maximisers(J, PLATEAU)
        calls_at = {f"{c}/{t}": full_calls(rows, c, t) for c, t in argmax}
        lodo = {}
        for ho in datasets:
            train = [r for r in rows if r["dataset"] != ho]
            Jt = grid(train, outcome)
            b, am = maximisers(Jt)
            held = [r for r in rows if r["dataset"] == ho]
            ho_calls = sorted({bool(call([r for r in held if r["K"] == 10][0], c, t)) for c, t in am})
            lodo[ho] = dict(J_train=b, maximisers=[list(k) for k in am],
                            declared_in_maximisers=DECLARED in am,
                            heldout_full_calls=ho_calls,
                            heldout_matches_declared=ho_calls == [declared_calls[ho]])
        out["objectives"][name] = dict(
            share_positive=float(np.mean([outcome(r) for r in rows])),
            J_max=best, argmax=[list(k) for k in argmax],
            J_declared=J[DECLARED], declared_in_plateau=DECLARED in plateau,
            plateau=[list(k) for k in plateau],
            full_calls_at_argmax=calls_at,
            full_calls_differ_from_declared=sorted({ds for v in calls_at.values()
                                                   for ds, a in v.items() if a != declared_calls[ds]}),
            lodo=lodo,
            # why a flat optimum in phi_0 is flat: how many panels clearing the count
            # at each maximising c lie above the smallest maximising phi_0
            panels_above_flat_phi={f"{c}": sum(r["spread"] >= math.ceil(c * r["K"] - 1e-9)
                                               and r["phi"] > min(t for cc, t in argmax if cc == c)
                                               for r in rows)
                                   for c in sorted({c for c, _ in argmax})},
            lodo_heldout_matches=sum(v["heldout_matches_declared"] for v in lodo.values()))

    full = [r for r in rows if r["K"] == 10]
    fam = {}
    for m in (0.05, 0.10, 0.15):
        safe = lambda r, m=m: r["kappa_stacked"] - r["kappa_frontier"] >= -m
        fam[f"A_safe_margin_{m:.2f}"] = dict(subpanels=informative_share(rows, safe),
                                              full_panels=informative_share(full, safe))
    fam["B_gain"] = dict(subpanels=informative_share(rows, OBJECTIVES["B_gain"]),
                         full_panels=informative_share(full, OBJECTIVES["B_gain"]))
    out["family"] = fam
    # The grid pairs that split nothing, admitting all or none of a set: the
    # difference between the grid's size and the `points` counts above.
    def non_splitting(rs):
        none_ = [[c, t] for c in C_GRID for t in T_GRID if not any(call(r, c, t) for r in rs)]
        all_ = [[c, t] for c in C_GRID for t in T_GRID if all(call(r, c, t) for r in rs)]
        return dict(n_admit_none=len(none_), n_admit_all=len(all_),
                    admit_none_c=sorted({c for c, _ in none_}),
                    admit_none_phi_range=[min(t for _, t in none_), max(t for _, t in none_)] if none_ else None)
    out["grid"] = dict(n_pairs=len(C_GRID) * len(T_GRID), n_c=len(C_GRID), n_phi=len(T_GRID),
                       subpanels=non_splitting(rows), full_panels=non_splitting(full))
    # The cell the full suite leaves empty -- enough competent members AND errors
    # above the correlation bar -- is populated by sub-panels; compare the two
    # sides of the bar at fixed competence, within dataset.
    q = [r for r in rows if r["spread"] >= math.ceil(DECLARED[0] * r["K"] - 1e-9)]
    half = {}
    for ds in dict.fromkeys(r["dataset"] for r in q):
        sub = [r for r in q if r["dataset"] == ds]
        lo = [r for r in sub if r["phi"] <= DECLARED[1]]
        hi = [r for r in sub if r["phi"] > DECLARED[1]]
        g = lambda xs: float(np.mean([r["kappa_stacked"] - r["kappa_best_cv"] for r in xs])) if xs else None
        half[ds] = dict(n_below_bar=len(lo), n_above_bar=len(hi),
                        gain_best_below_bar=g(lo), gain_best_above_bar=g(hi))
    out["correlation_half"] = dict(n_competence_qualified=len(q),
                                   n_above_bar=sum(r["phi"] > DECLARED[1] for r in q), per_dataset=half)
    out["kappa0_same_calls"] = kappa0_interval(datasets)

    path = Path(args.out) if args.out else DERIVED / "_summary" / "admissibility_tuning.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {path}\n")
    print("declared full-panel calls:", {k: int(v) for k, v in declared_calls.items()})
    for name, o in out["objectives"].items():
        cs = sorted({c for c, _ in o["argmax"]})
        print(f"\n== {name}: share positive {o['share_positive']:.2f}")
        print(f"  J max {o['J_max']:.3f} at {len(o['argmax'])} point(s); c in {cs}, "
              f"t in [{min(t for _, t in o['argmax']):.2f}, {max(t for _, t in o['argmax']):.2f}]")
        print(f"  J at declared {DECLARED}: {o['J_declared']:.3f}; in {PLATEAU}-plateau: "
              f"{o['declared_in_plateau']} (plateau {len(o['plateau'])} points)")
        print(f"  full-panel calls differing from declared at some maximiser: "
              f"{o['full_calls_differ_from_declared'] or 'none'}")
        for ho, v in o["lodo"].items():
            cs = sorted({c for c, _ in v["maximisers"]}); ts = [t for _, t in v["maximisers"]]
            print(f"  LODO {ho:20} J_train {v['J_train']:.3f}  c {cs} t [{min(ts):.2f},{max(ts):.2f}]  "
                  f"held-out call {v['heldout_full_calls']} matches declared: {v['heldout_matches_declared']}")
        print(f"  LODO held-out calls matching the declared rule: {o['lodo_heldout_matches']}/{len(datasets)}")
    print("\nfamily (share of splitting grid points with J > 0):")
    for k, v in out["family"].items():
        print(f"  {k:20} sub-panels {v['subpanels']['share_positive']:.3f} of {v['subpanels']['points']} "
              f"(min J {v['subpanels']['J_min']:+.3f});  full panels {v['full_panels']['share_positive']:.3f} "
              f"of {v['full_panels']['points']} (min J {v['full_panels']['J_min']:+.3f})")
    k0 = out["kappa0_same_calls"]
    print(f"competence bar with the declared calls: [{k0['lo']:.3f}, {k0['hi']:.3f}]")


if __name__ == "__main__":
    main()
