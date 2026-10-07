#!/usr/bin/env python
"""The cost model: the rate constants and per-item costs on which every cost in the paper rests.

The rates live as constants below and can be overridden per invocation with
--rate NAME=VALUE; `--show-rates` prints the rate table in force and the per-item
costs it implies. external_baselines_table.py imports RATES, per_item_costs and the
calibration budgets, so a repricing reaches tab_profile and the break-even volumes
it prints.

The rates are a public reference rate for the GPU card, throughput measured with
raim-verdicts' diagnostics/bench_throughput.py, and list API pricing, all as of
mid-2026; they price tokens rather than provisioned hardware (app:cost).
"""
from __future__ import annotations
import argparse


# ---------------- adjust these to your actual rates ---------------------
# GPU: the panel was run on a proprietary server with no per-hour billing, so we
# report against a public reference rate for the equivalent card (NVIDIA A40) as
# a transparency proxy. Per-GPU on-demand rates observed 2026-06 span $0.22 (12mo
# reserved, RunPod) -- $2.13 (Exoscale); we take RunPod on-demand as a commodity
# midpoint. Adjust here to reprice the whole table.
GPU_USD_PER_HOUR = 0.44            # A40, RunPod on-demand per GPU (range 0.22-2.13)

# Output tokens per judge call, measured from the brief-reason run artefacts
# (~52 words -> ~68 tokens/item; the constrained-decoding lp scorer emits far
# fewer -- a single verdict token). Input tokens are the rubric + reference +
# candidate; long-context sets push this well above the value below, so it is a
# conservative representative estimate (the actual API spend was cross-checked
# against the metered usage, provenance/claude_api_cost_2026_05_30_to_2026_06_30.csv).
TOKENS_PER_ITEM_OUT = 68
TOKENS_PER_ITEM_IN  = 500

# Throughput as ITEMS/sec (full prefill+decode wall time per item), measured by
# raim-verdicts' diagnostics/bench_throughput.py at gpu-util 0.90, batch 128, in
# ~500 tok, with --fixed-output 68 so every model decodes the same realistic
# ~68-token verdict and the figures are comparable across models
# (provenance/throughput_summary_{panel_0p90,qwen}_fixed68.txt). The ten members
# span 23.7-40.2 items/s (mean/median ~32);
# the value below is their harmonic mean, so PANEL = PANEL_K * SMALL reproduces the
# true sum of per-member times (~0.32 s/item for the ten-member panel). The 32B AWQ
# is correctly ~2x slower than a small member, as expected for a larger model.
ITEMS_PER_S_SMALL_MEMBER = 31.2    # harmonic mean of the 10 members, 0p90 fixed-68
ITEMS_PER_S_OSS_FRONTIER = 14.82   # Qwen2.5-32B-Instruct-AWQ, 0p90 fixed-68

# API rates (Claude Sonnet 4.6 list, USD per Mtok). Confirmed against the actual
# metered spend (provenance/), billed at list price (no cache discount), i.e.
# standard Sonnet 4.6 rates.
API_USD_PER_IN_TOK  = 3.00 / 1_000_000
API_USD_PER_OUT_TOK = 15.00 / 1_000_000

# GPT-4o list pricing, for the one published row in tab_profile that is an API
# judge other than ours. It is a published QUALITY figure -- we do not run GPT-4o --
# so this prices the same item profile through its own list rate, and nothing else
# in the paper reads it.
GPT4O_USD_PER_IN_TOK  = 2.50 / 1_000_000
GPT4O_USD_PER_OUT_TOK = 10.00 / 1_000_000

# The price of a gold label. No item was hand-annotated in this work (gold came with
# the benchmarks), so this is what a deployer would pay: $30/hr at 2 min per item,
# i.e. $1 per label, which prices the calibration budgets below.
HUMAN_USD_PER_HOUR = 30.00
HUMAN_MIN_PER_ITEM = 2.0

# Panel parameters
PANEL_K = 10                       # the paper panel is ten judges (raim.panel.PANEL)
# A full 1,000-label calibration set: the counterfactual app:cost prices.
N_GOLD_FULL     = 1000

# The two budgets the aggregator's OWN gold learning curve identifies
# (stacker_curve.py, derived/_summary/stacker_curve.json). Both are quoted in LABELS and
# priced for the paired question-answering sets, where one labelled record is
# two items; on the claim-verification sets a record is one item and the same
# record count costs half as much, so pricing the paired case is the
# conservative choice.
#   SWEET   50 labelled records -- the budget at which the fitted stacker clears
#           the label-free majority vote on seven of the eight datasets
#   PLATEAU 100 labelled records -- within 5% of full-supervision kappa on six of
#           the eight
N_GOLD_SWEET   = 100               # labels = 50 paired records
N_GOLD_PLATEAU = 200               # labels = 100 paired records

# ------------------------------------------------------------------------


# The rates above are the defaults, not the truth: they are a public reference rate
# for the card, a measured throughput, and list API pricing on a particular date.
# `--rate NAME=VALUE` overrides any of them from the command line, so a repricing is
# an invocation with a record rather than an edit without one.
#
# These feed a claim in the paper -- the ten members at $0.039 per thousand items
# against $2.52 for Sonnet -- so a change here is a change to the manuscript, not
# merely to this report.
RATES = {
    "gpu_usd_per_hour":     GPU_USD_PER_HOUR,
    "items_per_s_member":   ITEMS_PER_S_SMALL_MEMBER,
    "items_per_s_oss":      ITEMS_PER_S_OSS_FRONTIER,
    "api_usd_per_in_tok":   API_USD_PER_IN_TOK,
    "api_usd_per_out_tok":  API_USD_PER_OUT_TOK,
    "gpt4o_usd_per_in_tok":  GPT4O_USD_PER_IN_TOK,
    "gpt4o_usd_per_out_tok": GPT4O_USD_PER_OUT_TOK,
    "tokens_per_item_in":   TOKENS_PER_ITEM_IN,
    "tokens_per_item_out":  TOKENS_PER_ITEM_OUT,
    "human_usd_per_hour":   HUMAN_USD_PER_HOUR,
    "human_min_per_item":   HUMAN_MIN_PER_ITEM,
    "panel_k":              PANEL_K,
}


def per_item_costs(r: dict) -> dict:
    """The five per-item inference costs, from a rate table."""
    small = (1.0 / r["items_per_s_member"]) / 3600.0 * r["gpu_usd_per_hour"]
    oss   = (1.0 / r["items_per_s_oss"])    / 3600.0 * r["gpu_usd_per_hour"]
    return dict(
        small=small,
        panel=small * r["panel_k"],
        oss=oss,
        api=(r["api_usd_per_in_tok"]  * r["tokens_per_item_in"] +
             r["api_usd_per_out_tok"] * r["tokens_per_item_out"]),
        api_gpt4o=(r["gpt4o_usd_per_in_tok"]  * r["tokens_per_item_in"] +
                   r["gpt4o_usd_per_out_tok"] * r["tokens_per_item_out"]),
        human=r["human_usd_per_hour"] * (r["human_min_per_item"] / 60.0),
    )




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", action="append", metavar="NAME=VALUE", default=[],
                    help="override a rate constant, e.g. --rate gpu_usd_per_hour=2.13 "
                         "(repeatable). Names: " + ", ".join(sorted(RATES)))
    ap.add_argument("--show-rates", action="store_true",
                    help="print the rate table in force and exit")
    args = ap.parse_args()

    # Apply --rate overrides before anything derived from them is computed.
    for spec in args.rate:
        if "=" not in spec:
            ap.error(f"--rate expects NAME=VALUE, got {spec!r}")
        name, _, value = spec.partition("=")
        if name not in RATES:
            ap.error(f"unknown rate {name!r}; known: {', '.join(sorted(RATES))}")
        try:
            RATES[name] = float(value)
        except ValueError:
            ap.error(f"--rate {name} expects a number, got {value!r}")

    for k in sorted(RATES):
        print(f"  {k:22} {RATES[k]}")
    c = per_item_costs(RATES)
    print()
    for k in ("small", "panel", "oss", "api", "human"):
        print(f"  {k:22} ${c[k]:.8f} per item  (${c[k]*1000:.4f} per 1000)")




if __name__ == "__main__":
    main()