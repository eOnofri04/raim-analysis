#!/usr/bin/env python
"""The paper's headline numbers, each printed beside the file and key it is read from.

The abstract, the contributions and the conclusion quote numbers that no single
table carries -- a median over eight datasets, a tally of paired differences, a
break-even volume -- so this prints them, read from the committed derived tier and
the cost model, together with where each comes from. It computes nothing that a
generator does not already record: it reads, aggregates across datasets where the
text does, and formats. Every other number in the text is either in the table or
figure beside it or in the derived JSONs of that section's generator.

    headline_numbers.py         # read-only; writes nothing
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

import cost_table as C
from analyze_extras import holm

HERE = Path(__file__).resolve().parent
DERIVED = HERE.parent / "derived"
SUITE = ["medhallu", "ragtruth", "truthfulqa", "factscore",
         "aggrefact_xsum", "aggrefact_wice", "aggrefact_cnn", "aggrefact_expertqa"]
CORE = ["medhallu", "aggrefact_wice", "ragtruth", "aggrefact_xsum"]


def load(rel: str) -> dict:
    return json.loads((DERIVED / rel).read_text())


def show(value: str, what: str, source: str) -> None:
    print(f"  {value:>18}  {what}")
    print(f"  {'':>18}  <- {source}")


def main() -> None:
    pvs = {ds: load(f"{ds}/panel_vs_sonnet.json") for ds in SUITE}

    print("Against the frontier judge (Sonnet), over the eight datasets")
    ratio = st.median(p["kappa_panel"][0] / p["kappa_frontier"][0] for p in pvs.values())
    show(f"{100 * ratio:.0f}%", "median share of the frontier's kappa the panel retains",
         "derived/<ds>/panel_vs_sonnet.json: kappa_panel[0] / kappa_frontier[0], median")
    gap = st.mean(p["diff_bacc_panel_minus_frontier"][0] for p in pvs.values())
    bp = st.mean(p["bacc_panel"][0] for p in pvs.values())
    bf = st.mean(p["bacc_frontier"][0] for p in pvs.values())
    show(f"{-100 * gap:.1f} points", f"balanced accuracy given up on average "
         f"({100 * bp:.1f} against {100 * bf:.1f})",
         "derived/<ds>/panel_vs_sonnet.json: diff_bacc_panel_minus_frontier[0], mean")
    for ds, p in pvs.items():
        d, lo, hi = p["diff_panel_minus_frontier"]
        form = ("clear improvement" if lo > 0 else "clear worsening" if hi < 0
                else "unresolved")
        b = 100 * p["diff_bacc_panel_minus_frontier"][0]
        show(f"{d:+.3f}", f"{ds}: panel minus frontier kappa, [{lo:+.3f}, {hi:+.3f}], "
             f"{form} ({b:+.1f} bacc points)",
             f"derived/{ds}/panel_vs_sonnet.json: diff_panel_minus_frontier, "
             f"diff_bacc_panel_minus_frontier")

    print("\nAgainst the panel's own best member, on the four core grounded datasets")
    wins = [ds for ds in CORE
            if load(f"{ds}/weighted.json")["diffs"]["stacked-best_single_cv"][1] > 0]
    show(f"{len(wins)}/4", f"wins over the CV-best single ({', '.join(wins) or 'none'})",
         "derived/<ds>/weighted.json: diffs['stacked-best_single_cv'], lower bound > 0")
    stats = load("_extras/baseline_stats.json")
    show(f"{stats['multiplicity']['p_holm']['medhallu']:.3f}",
         "MedHallu's Holm-adjusted p over the four core datasets",
         "derived/_extras/baseline_stats.json: multiplicity.p_holm.medhallu")
    raw = {ds: v["p_stacked_best"] for ds, v in stats["per_dataset"].items()}
    over8 = dict(zip(raw, holm(list(raw.values()))))
    show(f"{over8['medhallu']:.3f}", "the same over all eight datasets",
         "derived/_extras/baseline_stats.json: per_dataset[].p_stacked_best, "
         "Holm-adjusted by analyze_extras.holm")
    tw = sum(load(f"_transfer/{ds}_transfer_lodo.json")["verdict"]["transfer_win"]
             for ds in CORE)
    show(f"{tw}/4", "transfer wins, leaving one dataset out (hard-vote pathway)",
         "derived/_transfer/<ds>_transfer_lodo.json: verdict.transfer_win")

    print("\nThe calibration budget")
    sc = load("_summary/stacker_curve.json")
    at50 = [d for d in sc["per_dataset"] for b in d["budgets"] if b["budget"] == 50]
    beat = sum(b["stacked"]["mean"] > d["unweighted_reference"]
               for d in sc["per_dataset"] for b in d["budgets"] if b["budget"] == 50)
    show(f"{beat} of {len(at50)}", "datasets where fifty records beat the label-free majority",
         "derived/_summary/stacker_curve.json: per_dataset[].budgets[budget=50]"
         ".stacked.mean vs unweighted_reference")
    rb = load("_summary/regime_budget.json")["by_budget"]
    for n in ("50", "100"):
        show(f"{100 * rb[n]['mean_p_call_correct']:.1f}%",
             f"regime call reproduced from {n} labelled records",
             f"derived/_summary/regime_budget.json: by_budget.{n}.mean_p_call_correct")
    show(f"{100 * rb['100']['mean_p_leader']:.0f}%",
         "best member named correctly from 100 labelled records",
         "derived/_summary/regime_budget.json: by_budget.100.mean_p_leader")

    print("\nCost (cost_table.py's rate constants; `make reports` prints the rate table)")
    c = C.per_item_costs(C.RATES)
    panel, api = 1000 * c["panel"], 1000 * c["api"]
    show(f"${panel:.3f} / ${api:.2f}", "per 1,000 items, panel against the Sonnet API",
         "cost_table.per_item_costs: 'panel', 'api'")
    show(f"{api / panel:.0f}x ({100 * panel / api:.2f}%)", "price ratio",
         "the same two values")
    for n in (C.N_GOLD_SWEET, C.N_GOLD_PLATEAU):
        be = 1000.0 * c["human"] * n / (api - panel)
        show(f"{be:,.0f} items", f"break-even against the API for {n // 2} labelled records",
             "one-time labelling cost / per-item saving, as external_baselines_table.py")

    print("\nAgainst the literature: read from the tables themselves -- the mean column of "
          "tab_extbase (the panel against GPT-4o and the best leaderboard row) and "
          "tab_profile (the strongest detector we reran, on the six grounded sets)")


if __name__ == "__main__":
    main()
