# Provenance of the hand-transcribed constants

Two kinds of number enter the paper as constants rather than as computations from the verdicts: the rates behind every cost the paper states, and the published results set beside ours.
Nothing in the build reads the files here; they are the record those constants were transcribed from.

- `PUBLISHED_REFERENCE_VALUES.md` — every published reference value in `tab_extbase`, `tab_profile`, `tab_extf1` and `tab_extcontext`, with its source and the table, page, row and column where it can be read.

The remaining files concern the rate constants of `cost_table.py`, which price every cost stated in the paper (`tab_profile`, the *Cost* section of Appendix H, and the break-even volumes printed by `external_baselines_table.py`).

- `throughput_summary_panel_0p90_fixed68.txt` — the ten panel members, measured with `raim-verdicts`' `diagnostics/bench_throughput.py` at GPU utilisation 0.90, batch 128, about 500 input tokens and a fixed 68-token decode.
  `ITEMS_PER_S_SMALL_MEMBER` is the harmonic mean of its `items/s` column.
- `throughput_summary_qwen_fixed68.txt` — the same benchmark on Qwen2.5-32B-Instruct-AWQ, the source of `ITEMS_PER_S_OSS_FRONTIER`.
- `claude_api_cost_2026_05_30_to_2026_06_30.csv` — the metered API usage of the Sonnet judge over the measurement period, as exported from the provider's console.
  Its `cost_usd` and `list_price_usd` columns coincide on every row, which is the evidence that the spend was billed at list price, as `cost_table.py` assumes.

> **Note.** The two throughput dumps are verbatim, including their closing hint, which names rate constants that `cost_table.py` no longer uses (it reads items per second, not output tokens per second).
> The API export omits two columns of the original, the workspace and the API-key label, which identify the account and carry no cost information.
