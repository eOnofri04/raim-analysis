#!/usr/bin/env python
r"""External comparability: the published record, and what each route offers.

Two tables, both mixing published numbers with our own:

  tab_extbase.tex  -- the four LLM-AggreFact sets in leaderboard shape (systems as
                      rows, datasets as columns, balanced accuracy), the published
                      block above ours. These four are the only sets on which the
                      field and we score the same underlying rows, which is what
                      makes a shared column legitimate at all.
  tab_profile.tex  -- the route-level profile: what each way of getting a
                      faithfulness verdict costs, needs, and covers, beside what it
                      scores. Quality is balanced accuracy, the metric the leaderboard
                      reports and the only one that needs no conversion between the
                      published rows and ours (see the note above `_pub_agg4_bacc`).

The published cells are NOT derived from our run JSONs: they are read off the
LLM-AggreFact leaderboard and the cited papers, each row recording where its figure
is found. They live in the PUBLISHED table below as
one canonical list, so a correction to a number or a citation cannot silently drift
between the table and the prose that reads off it. Everything in our own block is
loaded from the artefacts, as everywhere else.

Cost and calibration constants are imported from cost_table.py rather than
restated, so a repricing there reaches this table too.

    ../.venv/bin/python3 external_baselines_table.py
    ../.venv/bin/python3 external_baselines_table.py --out <dir>
"""
from __future__ import annotations
import argparse
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "tables"

# The four sets drawn from the LLM-AggreFact test split, in leaderboard order.
AGG = ["aggrefact_cnn", "aggrefact_xsum", "aggrefact_wice", "aggrefact_expertqa"]
AGGNAME = {"aggrefact_cnn": r"\cnn", "aggrefact_xsum": r"\xsum",
           "aggrefact_wice": r"\wice", "aggrefact_expertqa": r"\expertqa"}

# --------------------------------------------------------------------------------
# the published record: balanced accuracy on the four LLM-AggreFact test sets,
# from the live leaderboard (https://llm-aggrefact.github.io/, retrieved
# 2026-07-08) and the cited papers. None of this comes from our artefacts.
#
#   (display name, size gloss, route family, {dataset: bacc}, source)
#
# `None` marks a set the method does not report or cannot be applied to; a route's
# mean is taken over what it reports and the denominator is printed with it.
#
# SOURCES ARE WHERE THE NUMBER IS, not who built the system: Bespoke, GPT-4o,
# Claude-3.5 and Qwen2.5-72B appear nowhere in the MiniCheck paper, SummaC-ZS's
# LLM-AggreFact scores are MiniCheck's re-run of that system rather than anything in
# Laban et al., and Granite Guardian 3.3's figure is on the leaderboard alone. A
# referee who opens a citation must find the figure there; where the figure exists
# only on the leaderboard the row says so.
# --------------------------------------------------------------------------------
LB = r"leaderboard"
SPECIALIST, GUARD, PROMPTED, PRELLM = "specialist", "guard", "prompted", "prellm"
PUBLISHED = [
    ("Bespoke-MiniCheck-7B", "7B",   SPECIALIST,
     {"aggrefact_cnn": .655, "aggrefact_xsum": .778, "aggrefact_wice": .830,
      "aggrefact_expertqa": .592}, LB),
    # the only row whose four values are in the MiniCheck paper itself (Table 2)
    ("MiniCheck-Flan-T5-L",  "0.8B", SPECIALIST,
     {"aggrefact_cnn": .699, "aggrefact_xsum": .743, "aggrefact_wice": .722,
      "aggrefact_expertqa": .590}, r"\citealp{Tang_Laban_Durrett_24}"),
    # FactCG's own Table 10 (without threshold tuning) gives CNN 70.2 against the
    # leaderboard's 70.1; we keep the leaderboard value, this table being in
    # leaderboard shape, and name both sources.
    ("FactCG-DeBERTa-L",     "0.4B", SPECIALIST,
     {"aggrefact_cnn": .701, "aggrefact_xsum": .739, "aggrefact_wice": .742,
      "aggrefact_expertqa": .591}, r"\citealp{Lei_Li_Li_etal_25}, " + LB),
    # From the published version (Findings of ACL 2026, Table 1, pp. 16918--16932),
    # which reports all four values itself; the arXiv v1 carries a different, earlier
    # model (four-set mean 68.6). Author-run in think mode, not on the leaderboard.
    (r"HalluGuard-4B$^{\dagger}$", "4B", SPECIALIST,
     {"aggrefact_cnn": .707, "aggrefact_xsum": .755, "aggrefact_wice": .805,
      "aggrefact_expertqa": .594}, r"\citealp{Bergeron_Buhnila_Francois_etal_26}"),
    ("Granite Guardian 3.3", "8B",   GUARD,
     {"aggrefact_cnn": .670, "aggrefact_xsum": .749, "aggrefact_wice": .766,
      "aggrefact_expertqa": .596}, LB),
    # All four values appear in FactCG's Table 10 (row `GPT-4o-2024-05-13`, without
    # threshold tuning), exactly as stored (verified against the paper, 2026-08-14).
    ("GPT-4o",               "---",  PROMPTED,
     {"aggrefact_cnn": .681, "aggrefact_xsum": .768, "aggrefact_wice": .785,
      "aggrefact_expertqa": .596}, r"\citealp{Lei_Li_Li_etal_25}, " + LB),
    ("Claude-3.5 Sonnet",    "---",  PROMPTED,
     {"aggrefact_cnn": .676, "aggrefact_xsum": .751, "aggrefact_wice": .777,
      "aggrefact_expertqa": .609}, LB),
    ("Qwen2.5-72B-Instruct", "72B",  PROMPTED,
     {"aggrefact_cnn": .636, "aggrefact_xsum": .730, "aggrefact_wice": .802,
      "aggrefact_expertqa": .601}, LB),
    # MiniCheck Table 2 gives SummaC-ZS on all four sets (WiCE 62.8, ExpertQA 55.2),
    # confirmed in FactCG Table 10.
    ("SummaC-ZS",            "0.4B", PRELLM,
     {"aggrefact_cnn": .511, "aggrefact_xsum": .615, "aggrefact_wice": .628,
      "aggrefact_expertqa": .552},
     r"\citealp{Laban_Schnabel_Bennett_etal_22}, scored by \citealp{Tang_Laban_Durrett_24}"),
]
# Rows are looked up by name, not position, so that adding a
# published row cannot silently re-point another table's figures.
PUB = {r[0]: r for r in PUBLISHED}

# --------------------------------------------------------------------------------
# The four sets OUTSIDE LLM-AggreFact. Published work reports on all of them, but
# never on our rows and rarely in our metric, so they cannot share a column with the
# block above -- what they can do is say where the field stands and why the number
# is not ours to match. Each row: the strongest published anchor, its value, the
# metric it is in, and the one structural difference that blocks the comparison.
# Sources are the cited papers.
#
#   dataset -> (anchor system, value, metric, blocking difference, citation)
# --------------------------------------------------------------------------------
# MedHallu and RAGTruth are not listed: both report hallucinated-class F1, which
# external_f1.py computes on our items, so they carry a metric-matched comparison in
# tab_extf1. What blocks a comparison on the two below is a difference of
# supervision or of evidence, which no metric conversion repairs. The value strings
# are what the papers say: TruthfulQA reports a 90--96% validation-accuracy RANGE
# and no point estimate, and FActScore's bound is "less than a 2% error rate".
CONTEXT_ONLY = [
    ("truthfulqa", r"\mdname{GPT-judge} (fine-tuned \mdname{GPT-3-6.7B})", r"$90$--$96$", "accuracy",
     r"judges model \emph{generations} after fine-tuning on in-domain human ratings; we score the dataset's own reference answers zero-shot",
     r"\citealp{Lin_Hilton_Evans_22}"),
    ("factscore", "retrieval-augmented estimator", "---", "system-level error",
     r"retrieves Wikipedia evidence by construction, approximating the human score to within $2\%$ at corpus level; we supply no evidence and score per claim",
     r"\citealp{Min_Krishna_Lyu_etal_23}"),
]


def _mean(d):
    v = [x for x in d.values() if x is not None]
    return st.mean(v), len(v)


def _fmt(x):
    """Balanced accuracy as xx.x, the convention throughout (kappa is 0.xxx)."""
    return "---" if x is None else f"${100*x:.1f}$"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT),
                    help="directory to write the tables into (default: tables/; "
                         "`make check` redirects it)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # our own numbers come from the artefacts, never from a Markdown record
    from paper_tables import load, RUNS, DATASETS
    from raim_lib.verdicts import resolve as zone

    def ours(key):
        """Per-set balanced accuracy for one of our own rows, over the four sets."""
        d = {}
        for ds in AGG:
            w = load(ds)["weighted"]
            d[ds] = w["bacc"][key][0]
        return d

    def judge(tag):
        d = {}
        for ds in AGG:
            p = zone(RUNS / ds / f"judge_{tag}.json")
            d[ds] = json.loads(p.read_text())["balanced_accuracy"][0]
        return d

    OURS = [
        ("average panel member",              "4--9B", "avg",   ours("average_member")),
        ("best single member (CV-selected)",  "4--9B", "single", ours("best_single_cv")),
        (r"\mdname{MiniCheck-Flan-T5-L} \emph{(our run)}", "0.8B", "repro", judge("pt_minicheck_ft5")),
        (r"\mdname{Bespoke-MiniCheck-7B} \emph{(our run)}$^{\S}$", "7B", "repro",
         judge("pt_bespoke_minicheck")),
        (r"\qwenbig{72B} \textsc{awq} \emph{(our run)}", "72B", "repro", judge("qwen72b_awq")),
        (r"\textbf{stacked cheap panel (ours)}", r"$10\times$4--9B", "panel", ours("stacked")),
        (r"\sonnet \emph{(our run)}",          "---",   "frontier", judge("sonnet")),
    ]

    # ---------------------------------------------------------------- tab_extbase
    def _md(name):
        """Set a published row's system name as a model identifier; the names in
        PUBLISHED stay plain because they double as lookup keys (PUB)."""
        base, sep, mark = name.partition("$^")
        return rf"\mdname{{{base}}}" + (sep + mark if sep else "")

    def block(rows, published):
        out_rows = []
        for name, size, fam, vals, *cite in rows:
            if published:
                name = _md(name)
            m, nd = _mean(vals)
            cells = " & ".join(_fmt(vals[ds]) for ds in AGG)
            mean = f"${100*m:.1f}$" + (f"$^{{({nd})}}$" if nd < len(AGG) else "")
            tail = f" & {{\\scriptsize {cite[0]}}}" if published else " & "
            out_rows.append(f"{name} & {size} & {cells} & {mean}{tail} \\\\")
        return out_rows

    body = ["\\multicolumn{8}{l}{\\emph{published, from the leaderboard and the cited papers}} \\\\"]
    body += block(PUBLISHED, True)
    body.append("\\midrule")
    body.append("\\multicolumn{8}{l}{\\emph{this work, one protocol and one frame per dataset across all eight sets}} \\\\")
    body += block(OURS, False)

    byname = {r[0]: r[3] for r in OURS}
    panel_mean, _ = _mean(byname[r"\textbf{stacked cheap panel (ours)}"])
    son_mean, _ = _mean(byname[r"\sonnet \emph{(our run)}"])
    bsp_mean, _ = _mean(byname[r"\mdname{Bespoke-MiniCheck-7B} \emph{(our run)}$^{\S}$"])
    best_pub = max(_mean(r[3])[0] for r in PUBLISHED if _mean(r[3])[1] == len(AGG))
    # the checkpoint the section mark concerns, which is not the column's maximum
    bsp_pub = _mean(PUB["Bespoke-MiniCheck-7B"][3])[0]
    # The partial-mean note is printed only if some row actually earns it; every
    # row currently reports all four sets.
    partial_note = (r"$^{(k)}$ a mean over the $k<4$ sets a system reports; "
                    if any(_mean(r[3])[1] < len(AGG) for r in PUBLISHED) else "")

    banner = (f"% AUTO-GENERATED by {Path(sys.argv[0]).name} in scripts/ "
              f"-- do not edit by hand.\n")
    tex = banner + r"""\begin{table*}[tp]
\centering\footnotesize
\caption{The four \aggrefact\ sets in leaderboard shape (balanced accuracy), the only sets on which we score the same rows as the published results.
Upper block published; lower block measured here under the one protocol and per-dataset frame of~\S\ref{ssec:panel}.}
\label{tab:extbase}
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l l cccc c l}
\toprule
System & size & """ + " & ".join(AGGNAME[ds] for ds in AGG) + r""" & mean & source \\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}}
\tablelegend{
\emph{source}:~where each published value is found, \emph{leaderboard} being the benchmark's public leaderboard (\url{https://llm-aggrefact.github.io/}, retrieved 2026-07-08), which carries systems post-dating the papers, and a citation being given only where the cited work itself reports the four values;
""" + partial_note + r"""$^{\dagger}$:~run by its own authors and reported in their paper, not on the leaderboard;
$^{\S}$:~a checkpoint we do not reproduce (""" + f"${100*bsp_mean:.1f}$ against ${100*bsp_pub:.1f}$" + r""" published), whose published row we read instead (Appendix~\ref{app:extbase}).}
\end{table*}
"""
    (out / "tab_extbase.tex").write_text(tex)
    print(f"wrote tab_extbase.tex (panel {panel_mean:.3f}, our Sonnet {son_mean:.3f}, "
          f"best published {best_pub:.3f})")

    # ---------------------------------------------------------------- tab_profile
    # SCOPE IS THE POINT OF THIS TABLE. The AggreFact-4 mean alone is the ONE slice
    # on which the purpose-trained specialists are at their strongest -- it is their
    # training distribution -- and would show the panel narrowly losing to systems
    # that, off that slice, either collapse or cannot be run at all. The three
    # quality columns report the same metric over widening scopes, and the n/a cells
    # are the finding rather than missing data.
    from raim_lib.verdicts import resolve as _zone
    GR6 = AGG + ["medhallu", "ragtruth"]
    ALL8 = list(DATASETS)

    def _scope_mean(fn, keys):
        """Mean over `keys`, or None if the route cannot be run on all of them."""
        vals = [fn(d) for d in keys]
        return st.mean(vals) if all(v is not None for v in vals) else None

    # Two different absences, marked differently.
    # NA  -- the route CANNOT be run at that scope: a claim-support detector needs
    #        a document, and the two ungrounded sets supply none. Structural, and
    #        part of the finding.
    # None -- we did not measure it there and no published value covers it.
    #        Contingent, and says nothing about the route.
    NA = "__NA__"

    def _kfmt(v, bold=False):
        if v is NA or v == NA:
            return r"\emph{n/a}"
        if v is None:
            return r"---"
        # A pre-formatted cell (a published value with our rerun beside it) passes
        # through untouched; only the plain floats are formatted here.
        if isinstance(v, str):
            return v
        return f"$\\mathbf{{{100*v:.1f}}}$" if bold else f"${100*v:.1f}$"

    def _pub_and_ours(pub, ours):
        """Published headline, our own rerun of the same checkpoint beside it."""
        return f"${100*pub:.1f}$ $({100*ours:.1f}^{{\\dagger}})$"

    def _ours_only(v):
        return f"${100*v:.1f}^{{\\dagger}}$"

    # This table reports BALANCED ACCURACY, not kappa. The quality block juxtaposes published values with ours, and bacc is the only
    # metric on which that juxtaposition needs no conversion: the leaderboard
    # reports bacc, and bacc is the mean of the two per-class recalls, so it does
    # not move with class prevalence and carries from the benchmark's own
    # imbalanced split to our balanced draw from it. Kappa does not: on a balanced
    # set kappa = 2*bacc-1 exactly, but on THEIR split the chance term is not 0.5,
    # so a published kappa is not 2*bacc-1, and printing one would put a derived
    # number under a citation that does not contain it. The kappa view of these rows
    # is given per dataset in tab_ptjudge (app:ptj).
    def _pub_agg4_bacc(row):
        """Published AggreFact-4 balanced accuracy, verbatim -- no conversion."""
        m, n = _mean(row[3])
        return m if n == len(AGG) else None

    def _ours_bacc(key):
        return lambda ds: load(ds)["weighted"]["bacc"][key][0]

    def _sonnet_bacc(ds):
        return json.loads(
            _zone(RUNS / ds / "judge_sonnet.json").read_text())["balanced_accuracy"][0]

    def _ptb(ds, j):
        pth = _zone(RUNS / ds / f"judge_pt_{j}.json")
        return json.loads(pth.read_text())["balanced_accuracy"][0] if pth.exists() else None

    import cost_table as C
    c = C.per_item_costs(C.RATES)
    per1k = {kk: v * 1000 for kk, v in c.items()}
    gold_total = c["human"] * C.N_GOLD_FULL
    # The measured operating range, from the aggregator's own gold curve: 50
    # labelled records to clear the label-free vote, 100 to sit within 5% of full
    # supervision.
    # The unit here is the LABEL, not the record: c["human"] is HUMAN_USD_PER_HOUR *
    # HUMAN_MIN_PER_ITEM/60 = $1.00 per item, and N_GOLD_SWEET/PLATEAU count labels
    # (100/200 labels = 50/100 paired records), so the one-time column's $100--$200 is
    # $1 per label and $2 per pair; the legend says "per label" for that reason.
    # Changing HUMAN_MIN_PER_ITEM rescales the break-even volumes quoted in the body,
    # app:cost, the abstract and the conclusion, so change them together.
    sweet_total = c["human"] * C.N_GOLD_SWEET
    plateau_total = c["human"] * C.N_GOLD_PLATEAU

    def suite_kappa(getter):
        return st.mean([getter(ds) for ds in DATASETS])

    k_stk = suite_kappa(lambda ds: load(ds)["weighted"]["kappa"]["stacked"][0])

    # Rows name the system rather than its family. The one-time column prices the
    # DEPLOYER's outlay: every published route here ships released weights, so
    # adopting one costs nothing one-time (the corpus was the authors' expense and
    # is not recoverable from their papers), whereas ours is a real deployer cost
    # and stays priced; the legend says so rather than hiding the asymmetry behind a
    # shared label.
    stk = _ours_bacc("stacked"); avg = _ours_bacc("average_member")
    cvb = _ours_bacc("best_single_cv")
    def _q32(ds):
        return json.loads(
            _zone(RUNS / ds / "judge_qwen32b_awq.json").read_text())["balanced_accuracy"][0]

    # (label, size, agg4, gr6, all8, $/1k, one-time, open, ungrounded)
    #
    # `ungr.` is NOT implied by the suite-8 cell: that cell distinguishes measured from unmeasured, whereas
    # this says whether the route could be run on ungrounded factuality at all --
    # which separates GPT-4o (could, we did not) from MiniCheck (cannot).
    # `size` earns its place by making the comparison legible: the sharpest
    # competitor is a 0.4--8B trained verifier, against ten 4--9B generalists.
    SPEC_1K = f"${per1k['small']:.3f}^{{\\P}}$"
    # PROVENANCE IS MARKED PER ROW. Two of the five comparators below are not
    # published values at all: the MiniCheck-Flan-T5-L and Prometheus-2 rows are OUR
    # OWN runs of those released checkpoints, which is precisely why they carry
    # grounded-6 and suite-8 cells that no publication covers, and why they carry a
    # dagger. The three genuinely published rows point at the leaderboard, where
    # their figures are.
    prow = [
        (r"\mdname{Bespoke-MiniCheck-7B}~{\scriptsize (leaderboard)}", "7B",
         _pub_agg4_bacc(PUB["Bespoke-MiniCheck-7B"]), None, NA,
         SPEC_1K, r"fine-tune", r"\yes", r"\no"),
        # The one comparator with BOTH a published AggreFact-4 figure (Tang et al.
        # Table 2, the only published row whose four values are in the paper itself)
        # and our own rerun through the official package: the cell carries the
        # published headline with our reproduction beside it, which is also the
        # harness calibration app:extbase rests on.
        (r"\mdname{MiniCheck-Flan-T5-L}~{\scriptsize\citep{Tang_Laban_Durrett_24}}", "0.8B",
         _pub_and_ours(_pub_agg4_bacc(PUB["MiniCheck-Flan-T5-L"]),
                       _scope_mean(lambda d: _ptb(d, "minicheck_ft5"), AGG)),
         _ours_only(_scope_mean(lambda d: _ptb(d, "minicheck_ft5"), GR6)), NA,
         SPEC_1K, r"fine-tune", r"\yes", r"\no"),
        # ungr. is \partialy on purpose. Granite
        # Guardian runs reference-free for a whole family of risks -- jailbreak,
        # profanity, toxicity, hate speech, IBM AI Risk Atlas dimensions, scored from
        # prompt and response alone (ibm-granite/granite-guardian README) -- so unlike
        # the MiniCheck rows it CAN be pointed at ungrounded data. What it cannot do
        # there is the faithfulness judgement: its only hallucination risks are context
        # relevance, groundedness and answer relevance, each defined over a retrieved
        # context (Padhi et al. 2024 sec 2.1.2). Route survives, judgement does
        # not -- hence partly, not no. Spelled out in app:ptj.
        # NO CITATION: arXiv 2412.07724 has only v1/v2 (Dec 2024) and documents
        # version 3.0, never 3.3, so the row stays leaderboard-only.
        (r"\mdname{Granite Guardian 3.3}~{\scriptsize (leaderboard)}", "8B",
         _pub_agg4_bacc(PUB["Granite Guardian 3.3"]), None, None,
         SPEC_1K, r"fine-tune", r"\yes", r"\partialy"),
        # Prometheus-2 has NO published LLM-AggreFact figure: it is not on the
        # leaderboard and its own paper never scores it there, being a rubric-grading
        # evaluator rather than a claim-support detector. Every cell in this row is
        # therefore ours, and all three carry the dagger. Do not fill a "published"
        # value here -- there is none to fill.
        (r"\mdname{Prometheus-2}~{\scriptsize\citep{Kim_Suk_Longpre_etal_24}}", "7B",
         _ours_only(_scope_mean(lambda d: _ptb(d, "prometheus2"), AGG)),
         _ours_only(_scope_mean(lambda d: _ptb(d, "prometheus2"), GR6)),
         _ours_only(_scope_mean(lambda d: _ptb(d, "prometheus2"), ALL8)),
         SPEC_1K, r"fine-tune", r"\yes", r"\yes"),
        (r"\mdname{GPT-4o}, prompted~{\scriptsize (leaderboard)}", "---",
         _pub_agg4_bacc(PUB["GPT-4o"]), None, None,
         f"${per1k['api_gpt4o']:.2f}$", r"\emph{none}", r"\no", r"\yes"),
        (None,) * 9,
        (r"average cheap open judge", r"$4$--$9$B",
         _scope_mean(avg, AGG), _scope_mean(avg, GR6), _scope_mean(avg, ALL8),
         f"${per1k['small']:.3f}$", r"\emph{none}", r"\yes", r"\yes"),
        (r"best single member (CV-selected)", r"$4$--$9$B",
         _scope_mean(cvb, AGG), _scope_mean(cvb, GR6), _scope_mean(cvb, ALL8),
         f"${per1k['small']:.3f}$", f"\\${sweet_total:,.0f}--\\${plateau_total:,.0f}", r"\yes", r"\yes"),
        (r"larger open judge (\qwenbig{32B})", "32B",
         _scope_mean(_q32, AGG), _scope_mean(_q32, GR6), _scope_mean(_q32, ALL8),
         f"${per1k['oss']:.3f}$", r"\emph{none}", r"\yes", r"\yes"),
        (r"\textbf{stacked cheap panel (this work)}", r"$10\times4$--$9$B",
         _scope_mean(stk, AGG), _scope_mean(stk, GR6), _scope_mean(stk, ALL8),
         f"${per1k['panel']:.3f}$", f"\\${sweet_total:,.0f}--\\${plateau_total:,.0f}",
         r"\yes", r"\yes"),
        (r"frontier \textsc{api} judge (\sonnet)", "---",
         _scope_mean(_sonnet_bacc, AGG), _scope_mean(_sonnet_bacc, GR6),
         _scope_mean(_sonnet_bacc, ALL8),
         f"${per1k['api']:.2f}$", r"\emph{none}", r"\no", r"\yes"),
    ]
    panel_row = prow[-2]
    prows = []
    for r in prow:
        if r[0] is None:
            prows.append(r"\midrule")
            prows.append(r"\multicolumn{9}{l}{\emph{measured here, one protocol across all eight sets}} \\")
            continue
        best = r is panel_row
        cells = [r[0], r[1]] + [_kfmt(v, best) for v in r[2:5]] + list(r[5:])
        prows.append(" & ".join(cells) + r" \\")

    be_sweet = 1000.0 * sweet_total / (per1k["api"] - per1k["panel"])
    be_plateau = 1000.0 * plateau_total / (per1k["api"] - per1k["panel"])
    be_gold = 1000.0 * gold_total / (per1k["api"] - per1k["panel"])
    ratio = per1k["api"] / per1k["panel"]

    tex = banner + r"""\begin{table*}[tp]
\centering\footnotesize
\caption{Quality, cost and applicability of each route. \bacc\ is evaluated over three nested groups: the four \aggrefact\ sets, the six grounded sets and the full suite.
Published cells (top) are from the literature (Tables~\ref{tab:extbase} and~\ref{tab:extf1}; \kp\ in Table~\ref{tab:ptjudge}).}
\label{tab:profile}
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l l ccc rl cc}
\toprule
 & & \multicolumn{3}{c}{\emph{quality} (\bacc) \emph{by scope}} & \multicolumn{2}{c}{\emph{cost} (\textsc{usd})} & \multicolumn{2}{c}{\emph{scope}} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Route & size & \aggrefact-4 & grounded 6 & suite 8 & per 1k & one-time & open & ungr. \\
\midrule
\multicolumn{9}{l}{\emph{purpose-trained and prompted comparators}} \\
""" + "\n".join(prows) + r"""
\bottomrule
\end{tabular}}
\tablelegend{
$^{\dagger}$:~our rerun under the system's released weights;
\emph{---}:~no data;
\emph{n/a}:~scope the judge cannot be run on;
\emph{ungr.}:~applicability to ungrounded factuality (\yes~yes, \partialy~partly, \no~no);
\emph{cost}:~inference per $1{,}000$ items (\textsc{api} or cloud \textsc{gpu} rates, Appendix~\ref{app:cost}) and one-time outlay on a new domain (\$$1$ per label);
$^{\P}$:~estimated by comparison.}
\end{table*}
"""
    (out / "tab_profile.tex").write_text(tex)
    print(f"wrote tab_profile.tex (panel {_mean(ours('stacked'))[0]:.3f} bacc / {k_stk:.3f} kappa; "
          f"{ratio:.0f}x cheaper inference)")
    # The break-even volumes the abstract, the body and app:cost quote: the one-time
    # calibration outlay over the per-item saving against Sonnet.
    print(f"  per 1,000 items: panel ${per1k['panel']:.3f}, Sonnet ${per1k['api']:.2f}")
    print(f"  break-even: {be_sweet:,.0f} items at {C.N_GOLD_SWEET} labels, "
          f"{be_plateau:,.0f} at {C.N_GOLD_PLATEAU}, {be_gold:,.0f} at a full "
          f"{C.N_GOLD_FULL}-label set")

    # ------------------------------------------------------------- tab_extcontext
    # The four sets with no shared denominator. Reported anyway: "no comparable
    # number exists" is a claim a reader should be able to check, and it is a
    # different claim from "we did not look".
    TEXNAME = {"medhallu": r"\medhallu", "ragtruth": r"\ragtruth",
               "truthfulqa": r"\truthfulqa", "factscore": r"\factscore"}
    crows = []
    for ds, sys_, val, metric, why, cite in CONTEXT_ONLY:
        w = load(ds)["weighted"]
        son = json.loads(zone(RUNS / ds / "judge_sonnet.json").read_text())
        crows.append(
            f"{TEXNAME[ds]} & {sys_}~{{\\scriptsize {cite}}} & {val} & {metric} & {why} & "
            f"${100*w['bacc']['stacked'][0]:.1f}$ & ${100*son['balanced_accuracy'][0]:.1f}$ \\\\")
    tex = banner + r"""\begin{table*}[tp]
\centering\scriptsize
\caption{Published anchors on the two benchmarks without a comparable published baseline: per set, the strongest anchor we found, the metric its authors report, and the structural difference that blocks the comparison.
\medhallu\ and \ragtruth, reported in hallucinated-class F1, are in Table~\ref{tab:extf1}.}
\label{tab:extcontext}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{l p{0.19\textwidth} c l p{0.30\textwidth} c c}
\toprule
 & \multicolumn{4}{c}{\emph{published anchor}} & \multicolumn{2}{c}{\emph{ours} (\bacc)} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-7}
Dataset & strongest reported system & value & metric & what blocks a direct comparison & panel & \sonnet \\
\midrule
""" + "\n".join(crows) + r"""
\bottomrule
\end{tabular}
\tablelegend{\emph{ours}:~balanced accuracy on our construction (Appendix~\ref{app:construction}), \emph{not} commensurable with the anchor beside it.}
\end{table*}
"""
    (out / "tab_extcontext.tex").write_text(tex)
    print("wrote tab_extcontext.tex")


if __name__ == "__main__":
    main()
