#!/usr/bin/env python
r"""Generate the paper's LaTeX tables from the derived JSONs (single source of truth).

Every number in these tables is emitted here, read from the authoritative
artefacts in ``derived/<dataset>/`` and the verdicts beside them -- never
hand-transcribed.

Sources, per dataset:
  weighted.json        -- method kappa / bacc with cluster-bootstrap CIs, paired
                          diffs, committed-n, oracle identity (the decision-rule
                          quantities).
  panel_vs_sonnet.json -- the paired panel-minus-frontier differences.
  judge_sonnet.json, judge_qwen{32,72}b_awq.json, judge_pt_<key>.json -- the single
                          judges' kappa / bacc, the purpose-trained ones included.
  experiment.jsonl     -- raw votes; used to recompute the single-member kappa
                          ladder, the per-member coverage, and the construction
                          counts of tab_construction.
  _extras/baseline_stats.json -- the Dawid--Skene column of tab_master.

Outputs (tables/):
  tab_master.tex          -- the comparator ladder and the three paired decisions.
  tab_frontier.tex        -- the panel against the open 32B/72B judges and Sonnet.
  tab_datasets.tex        -- the eight benchmarks.
  tab_construction.tex    -- how each benchmark was reduced to balanced binary items.
  tab_examples{1,2}.tex   -- one item per dataset, one table per construction.
  tab_methods_{kappa,bacc}.tex -- the full method ladder in each metric.
  tab_coverage.tex        -- per-member, per-dataset coverage.
  tab_ptjudge.tex         -- purpose-trained cheap judges vs the panel and Sonnet.

Also writes reports/coverage_summary.json -- the raw missing/total vote counts
behind the suite-wide imputation rate quoted in \S\ref{ssec:agg}.

Usage:
  python paper_tables.py                 # writes into tables/
  python paper_tables.py --out <dir>
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# raim_lib sits one level up; put the repository root on the path so this
# script runs the same whether driven by make or invoked directly.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from raim_lib.verdicts import resolve as zone
from raim_lib.verdicts import load as load_verdicts
import os as _os


def cohen_kappa_score(yt, yp):
    """Cohen's kappa for two equal-length label sequences (sklearn-equivalent)."""
    yt, yp = np.asarray(yt), np.asarray(yp)
    labels = np.unique(np.concatenate([yt, yp]))
    idx = {l: i for i, l in enumerate(labels)}
    C = np.zeros((len(labels), len(labels)))
    for a, b in zip(yt, yp):
        C[idx[a], idx[b]] += 1
    n = C.sum()
    po = np.trace(C) / n
    pe = (C.sum(0) * C.sum(1)).sum() / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


HERE = Path(__file__).resolve().parent
# RUNS names the DERIVED tree (zone 3). Verdict files (.jsonl) are addressed
# through the same root and re-pointed to verdicts/ by raim_lib.verdicts, so
# nothing here can write into the measurement mirror by mistyping a path.
RUNS = Path(_os.environ.get("RAIM_DERIVED", HERE.parent / "derived"))
# Zone 4, and the only thing this script writes besides tables/: the audit dumps
# that sit beside a table rather than feeding one. Redirectable with --reports, so
# `make check` can regenerate and diff them without touching the committed set.
REPORTS = HERE / "reports"

# --------------------------------------------------------------------------------
# dataset registry: directory -> (display name, tier, regime gloss)
# The regime gloss is an interpretive label, not a number; the numbers the tables
# print are all read from the JSONs.
# --------------------------------------------------------------------------------
CORE, GROUNDED, UNGROUNDED = "core", "grounded", "ungrounded"
DATASETS = {
    "medhallu":          ("MedHallu",   CORE,       "distributed competence"),
    "aggrefact_wice":    ("WiCE",       CORE,       "distributed, underpowered"),
    "ragtruth":          ("RAGTruth",   CORE,       "dominant single"),
    "aggrefact_xsum":    ("XSum",       CORE,       "moderate, redundant"),
    "aggrefact_cnn":     ("CNN",        GROUNDED,   "low signal, high corr."),
    "aggrefact_expertqa":("ExpertQA",   GROUNDED,   "low headroom"),
    "factscore":         ("FActScore",  UNGROUNDED, "distributed, low corr."),
    "truthfulqa":        ("TruthfulQA", UNGROUNDED, "dominant single"),
}
CORE_ORDER = ["medhallu", "aggrefact_wice", "ragtruth", "aggrefact_xsum"]

# LaTeX name macros (defined in the paper preamble): tables print the macro, so
# that a dataset or model is spelled and styled identically in prose and table.
TEXNAME = {
    "medhallu": r"\medhallu", "aggrefact_wice": r"\wice",
    "ragtruth": r"\ragtruth", "aggrefact_xsum": r"\xsum",
    "aggrefact_cnn": r"\cnn", "aggrefact_expertqa": r"\expertqa",
    "factscore": r"\factscore", "truthfulqa": r"\truthfulqa",
}
TEXTAG = {"Llama": r"\llama", "Qwen": r"\qwen", "Mistral": r"\mistral",
          "Gemma": r"\gemma", "Phi": r"\phimini", "Yi": r"\yi",
          "Command-R": r"\commandr", "GLM": r"\glm", "Granite": r"\granite",
          "Falcon": r"\falcon", "Sonnet": r"\sonnet", "Haiku": r"\haiku"}

# --------------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------------
def load(ds):
    d = {}
    d["weighted"] = json.loads((RUNS / ds / "weighted.json").read_text())
    d["diversity"] = json.loads((RUNS / ds / "diversity.json").read_text())
    jp = zone(RUNS / ds / "judge_sonnet.json")
    d["judge"] = json.loads(jp.read_text()) if jp.exists() else None
    return d


def pick_condition(conds):
    """The scoring frame for a dataset: def_on where present (the question-answer
    frame the QA-form sets use), else the single frame the file carries -- e.g.
    claimcheck for the LLM-AggreFact sets. Mirrors weighted.py, so the mixed-frame
    suite scores each dataset under its own frame with no per-dataset
    configuration."""
    conds = set(conds)
    return "def_on" if "def_on" in conds else sorted(conds)[0]


def member_kappa(ds, condition=None):
    """Single-member kappa ladder from the raw votes (cf. weighted.py _full_kappa).
    condition defaults to the dataset's own frame (pick_condition)."""
    rows = load_verdicts(RUNS / ds / "experiment.jsonl")
    cond = condition or pick_condition({r["condition"] for r in rows})
    votes, gold = {}, {}
    for r in rows:
        if r.get("condition") != cond:
            continue
        votes.setdefault(r["model"], {})[r["uid"]] = r["pred"]
        gold[r["uid"]] = r["gold"]
    out = {}
    for m, vm in votes.items():
        yt = [gold[u] for u in vm if vm[u] is not None]
        yp = [vm[u] for u in vm if vm[u] is not None]
        out[m] = cohen_kappa_score(yt, yp) if len(set(yt)) > 1 else float("nan")
    return out


# --------------------------------------------------------------------------------
# formatting helpers
# --------------------------------------------------------------------------------
def k(x):
    """kappa point estimate."""
    return f"{x:.3f}"


def pct(x):
    """Balanced accuracy as a percentage. Kappa is reported as 0.xxx and balanced
    accuracy as xx.x throughout, so which metric a cell carries is legible without
    reading the header."""
    return f"{100*x:.1f}"


def dci(triple):
    """signed diff with CI and a significance marker (CI excluding 0 -> *)."""
    p, lo, hi = triple
    star = "" if (lo <= 0 <= hi) else "\\,*"
    return f"${p:+.3f}${star}\\,{{\\scriptsize[{lo:+.2f},\\,{hi:+.2f}]}}"


def dci3(triple):
    """dci at the precision the body quotes intervals in: bounds to three decimals.
    Kept separate from dci because widening every interval cell in the appendix is a
    page-budget decision rather than a formatting one."""
    p, lo, hi = triple
    star = "" if (lo <= 0 <= hi) else "\\,*"
    return f"${p:+.3f}${star}\\,{{\\scriptsize[{lo:+.3f},\\,{hi:+.3f}]}}"


def dbacc(triple):
    """signed balanced-accuracy difference in percentage points, xx.x as everywhere."""
    p, lo, hi = triple
    star = "" if (lo <= 0 <= hi) else "\\,*"
    return f"${100*p:+.1f}${star}\\,{{\\scriptsize[{100*lo:+.1f},\\,{100*hi:+.1f}]}}"




def sig(triple):
    p, lo, hi = triple
    return not (lo <= 0 <= hi)


def verdict(w):
    """win/clean/tie/at-chance from the frozen decision rule."""
    diffs = w["diffs"]
    cv = diffs["stacked-best_single_cv"]
    orc = diffs["stacked-oracle_best"]
    stk = w["kappa"]["stacked"]
    if stk[1] <= 0 <= stk[2] and abs(stk[0]) < 0.05:
        return "at chance"
    if sig(cv) and cv[0] > 0:
        return "clean win" if (sig(orc) and orc[0] > 0) else "win"
    return "tie"


# Names the script that actually ran, not this module: several generators import
# this constant to stamp tables of their own, and sys.argv[0] is the entry point,
# so each table names the generator that wrote it.
HEADER = (f"% AUTO-GENERATED by {Path(sys.argv[0]).name} in scripts/ "
          f"-- do not edit by hand; regenerate with `make tables`.\n")


# --------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------
def _bold_if(sig_flag, s):
    """Bold a cell when its paired interval excludes zero. The master table carries
    three delta columns and no room for three sets of brackets, so significance is
    carried by weight rather than by a star -- see tab_master's caption."""
    return f"$\\mathbf{{{s}}}$" if sig_flag else f"${s}$"


def tab_master(out):
    """The comparator ladder and the three decisions that read off it.

    One row per dataset, columns ordered by the labelled information each
    comparator consumes -- label-free (average member, unweighted, Dawid--Skene),
    one global label-based selection (best-on-average), per-dataset label-based
    selection (CV-best, oracle), the stacked panel, and the frontier judge -- then
    the three paired differences the paper actually decides on. Both metrics are
    given for the panel and the frontier, since the substitution question is where
    the abstention treatment most deserves a second reading (\\S\\ref{ssec:metrics}).

    The floor of the ladder and its ceiling share one float because they are one
    ladder.
    """
    rows = []
    extras = json.loads((RUNS / "_extras" / "baseline_stats.json").read_text())["per_dataset"]
    tiers = [("Core grounded", CORE), ("Grounded (secondary)", GROUNDED),
             ("Ungrounded (contrast)", UNGROUNDED)]
    ncol = 16
    boa = None
    n_avg = n_cv = n_clean = n_par = n_son_sig = 0
    n_son_up = n_sonb_up = n_sonb_dn = 0
    tally = {kk: {"up": {True: 0, False: 0}, "dn": {True: 0, False: 0}}
             for kk in ("avg", "cv", "son", "sonb")}
    for tname, tier in tiers:
        members = [ds for ds, (_, t, _) in DATASETS.items() if t == tier]
        rows.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{{tname}}}}} \\\\")
        for ds in members:
            d = load(ds)
            w = d["weighted"]
            ka, ba, df = w["kappa"], w["bacc"], w["diffs"]
            boa = w.get("best_on_avg_model", boa)
            dag = "$^\\dagger$" if w["committed"]["best_single_cv"] < w["n"] else ""
            dsk = extras.get(ds, {}).get("kappa", {}).get("dawid_skene")
            dsk = k(dsk[0]) if dsk else "--"
            # Marginal frontier levels come from judge_sonnet.json, the same source
            # tab_frontier reads, so the two tables cannot disagree in the third
            # decimal; only the DIFFERENCE columns come from the pairing, which is
            # computed on the shared items and is the quantity the verdict turns on.
            son_k = k(d["judge"]["kappa"][0])
            son_b = pct(d["judge"]["balanced_accuracy"][0])
            pvs = json.loads((RUNS / ds / "panel_vs_sonnet.json").read_text())
            dson = pvs["diff_panel_minus_frontier"]
            dsonb = pvs["diff_bacc_panel_minus_frontier"]
            # The panel's kappa is underlined where the paired test does not put the
            # frontier clearly ahead -- i.e. where the difference is an improvement, or
            # a worsening the sample does not resolve -- so the substitution question
            # reads straight down the panel column. NB the mark is NOT "matches": an
            # unresolved worsening (CNN, -0.120 on 109 clusters) is underlined too, and
            # the caption says so, because calling that parity would be false.
            reaches = (not sig(dson)) or dson[0] > 0
            stk = k(ka["stacked"][0])
            stk = f"\\underline{{{stk}}}" if reaches else stk
            d_avg, d_cv = df["stacked-average_member"], df["stacked-best_single_cv"]
            d_orc = df["stacked-oracle_best"]
            clean = sig(d_cv) and d_cv[0] > 0 and sig(d_orc) and d_orc[0] > 0
            n_avg += sig(d_avg)
            n_cv += sig(d_cv) and d_cv[0] > 0
            n_clean += clean
            n_par += reaches
            n_son_sig += sig(dson) and dson[0] < 0
            n_son_up += sig(dson) and dson[0] > 0
            n_sonb_up += sig(dsonb) and dsonb[0] > 0
            n_sonb_dn += sig(dsonb) and dsonb[0] < 0
            # Every dataset falls in exactly one of four cells per delta column --
            # ahead or behind, significantly or not -- so the four counts of a
            # column sum to eight and a reader can see what the significant ones
            # were counted out of.
            for key, d in (("avg", d_avg), ("cv", d_cv), ("son", dson), ("sonb", dsonb)):
                tally[key]["up" if d[0] > 0 else "dn"][bool(sig(d))] += 1
            rows.append(
                f"{TEXNAME[ds]} & {w['n']} & "
                f"{k(ka['average_member'][0])} & {k(ka['unweighted'][0])} & {dsk} & "
                f"{k(ka['best_on_avg_single'][0])} & {k(ka['best_single_cv'][0])}{dag} & "
                f"{k(ka['oracle_best'][0])} & "
                f"{stk} & {pct(ba['stacked'][0])} & {son_k} & {son_b} & "
                f"{_bold_if(sig(d_avg), f'{d_avg[0]:+.3f}')} & "
                f"{_bold_if(sig(d_cv), f'{d_cv[0]:+.3f}')}"
                f"{'$^\\ddagger$' if clean else ''} & "
                f"{_bold_if(sig(dson), f'{dson[0]:+.3f}')} & "
                f"{_bold_if(sig(dsonb), f'{dsonb[0]*100:+.2f}')} \\\\")
    # One count line per delta column, reading up(ns) / down(ns): significant counts
    # bold, to recall the bold of the cells above; zeros as en-dashes, so the eye goes
    # to what is there rather than to what is not. The four numbers sum to eight, which
    # is what lets a significant count be read against how many datasets went that way.
    def half(t):
        s = f"\\textbf{{{t[True]}}}" if t[True] else "--"
        return f"{s}\\,({t[False] or '--'})"

    def cell(key):
        return f"{half(tally[key]['up'])}\\,/\\,{half(tally[key]['dn'])}"

    summary = (" & ".join([r"\emph{count} $\uparrow$/$\downarrow$"] + [""] * 11 +
                          [cell(kk) for kk in ("avg", "cv", "son", "sonb")]) + " \\\\")
    body = "\n".join(rows) + "\n\\midrule\n" + summary
    tex = HEADER + r"""\begin{table*}[tp]
\centering\scriptsize
\setlength{\tabcolsep}{2pt}
\caption{Comparator ladder over the eight benchmarks.
Columns are ordered by the labelled information each comparator consumes. \kp\ is given as a decimal and \bacc\ as a percentage.}
\label{tab:master}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l r rrr rr r rr rr r r r r}
\toprule
 & & \multicolumn{3}{c}{\emph{label-free}} & \multicolumn{3}{c}{\emph{single judge, label-selected}} & \multicolumn{2}{c}{\emph{stacked panel}} & \multicolumn{2}{c}{\emph{frontier} (\sonnet)} & \multicolumn{4}{c}{\emph{paired} $\Delta$ \emph{(panel $-$ comparator)}} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}\cmidrule(lr){13-16}
Dataset & $n$ & avg.\ mem. & unw. & D--S & best (avg) & best (CV) & oracle & \kp & \bacc & \kp & \bacc & avg (\kp) & best (\kp) & \sonnet (\kp) & \sonnet (\bacc) \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}}
\tablelegend{
\emph{avg.\ mem.}: average member; \emph{unw.}: unweighted majority; \emph{D--S}: Dawid--Skene.
The four rightmost columns are paired cluster-bootstrap differences on the same resampled clusters ($B=2000$), \textbf{bold} where the $95\%$ interval excludes zero (cf.\ Figure~\ref{fig:regimeforest}(b)); the \emph{count} line tallies them as improvement ($\uparrow$) or worsening ($\downarrow$), as `\textbf{clear}\,(unresolved)', read before rounding.
$^\ddagger$: beats the oracle too; \underline{underline}: the frontier is not clearly ahead; $^\dagger$: an under-covering CV-best baseline (Appendix~\ref{ssec:confound}).}
\end{table*}
"""
    (out / "tab_master.tex").write_text(tex)
    print(f"wrote tab_master.tex (avg {n_avg}/8, best-CV {n_cv}/8 of which {n_clean} clean, "
          f"frontier parity-or-better {n_par}/8, frontier ahead {n_son_sig}/8)")



# --------------------------------------------------------------------------------
# How each dataset was reduced to balanced binary items.
#
# The per-dataset rule is read off raim/tasks.py in the measurement repository (the
# builders `_build_medhallu`, `_build_truthfulqa`, `_build_ragtruth`,
# `_build_factscore` and `_make_aggrefact_builder`); the UPSTREAM pool sizes are
# properties of the source releases, counted against the pinned revisions (as of
# 2026-08-10) and recorded here because the release ships verdicts, not corpora. The
# item, cluster and class-balance columns beside them are recomputed from the
# shipped verdicts on every build, so a change in what we released cannot leave this
# table describing something else.
# --------------------------------------------------------------------------------
#   dataset -> (source, upstream pool, citation keys)
# The selection rule is shared within each construction group, so the table states
# it once per block (CONSTRUCTION_GROUPS) rather than once per row -- which is also
# the visual point, the suite having exactly two constructions and not eight.
# The third field is the citation key(s) for the dataset ITSELF, which is not in
# general what the source field names: on the four claim-verification rows the source is the
# LLM-AggreFact release we pull from, whilst the citation is whoever built the corpus
# underneath it. Keys follow tab_datasets, so the two tables attribute identically;
# CNN carries only the AggreFact annotation paper because the bibliography has no
# entry for CNN/DailyMail itself.
CONSTRUCTION = {
    "medhallu": (r"\mdname{MedHallu/pqa\_labeled}", r"$1{,}000$ questions",
                 "Pandit_Xu_Hong_etal_25"),
    "ragtruth": (r"\mdname{RAGTruth-processed}, QA", r"$5{,}034$ resp. / $839$ q.",
                 "Niu_Wu_Zhu_etal_24"),
    "truthfulqa": (r"\mdname{truthful\_qa/generation}", r"$817$ questions",
                   "Lin_Hilton_Evans_22"),
    "factscore": (r"\mdname{InstructGPT.jsonl}", r"$183$ biographies",
                  "Min_Krishna_Lyu_etal_23"),
    "aggrefact_wice": (r"\mdname{LLM-AggreFact/Wice}", r"$358$ ($111$/$247$), $355$ doc.",
                       "Kamoi_Goyal_Rodriguez_etal_23"),
    "aggrefact_xsum": (r"\mdname{LLM-AggreFact/XSum}", r"$558$ ($285$/$273$), $483$ doc.",
                       "Tang_Goyal_Fabbri_etal_23,Narayan_Cohen_Lapata_18"),
    "aggrefact_cnn": (r"\mdname{LLM-AggreFact/CNN}", r"$558$ ($501$/$57$), $384$ doc.",
                      "Tang_Goyal_Fabbri_etal_23"),
    "aggrefact_expertqa": (r"\mdname{LLM-AggreFact/ExpertQA}", r"$3{,}702$ ($2{,}971$/$731$), $3{,}135$ doc.",
                           "Malaviya_Lee_Chen_etal_24"),
}
# The two construction groups, in the order the table prints them.
CONSTRUCTION_GROUPS = [
    ("one matched pair per source record: a supported and a hallucinated candidate "
     "sharing its question and, on the grounded sets, its evidence",
     ["medhallu", "ragtruth", "truthfulqa", "factscore"]),
    ("classes pooled across documents and the majority subsampled to the minority "
     "count (seed $0$), clusters recovered from the shared document",
     ["aggrefact_wice", "aggrefact_xsum", "aggrefact_cnn", "aggrefact_expertqa"]),
]


# --------------------------------------------------------------------------------
# One illustrative item per dataset, for Appendix~\ref{app:construction}.
#
# These are SHORT quoted excerpts of published benchmark content, extracted once
# offline through raim.tasks.build_instances and pinned here as a reviewed constant.
# They are pinned rather than re-extracted at build time for two reasons: the
# release ships verdicts, not corpora, so a generator that loaded the corpora could
# not run here at all; and quoting is the one place the project puts source text in
# front of a reader, which is a decision that should be visible in a diff rather
# than recomputed silently. Each field is truncated; an ellipsis marks the cut.
# --------------------------------------------------------------------------------
EXAMPLES = {
    "medhallu": dict(
        uid_pos='0:pos', uid_neg='0:neg', same_cluster=True,
        question='Do mitochondria play a role in remodelling lace plant leaves during programmed cell death?',
        context='Programmed cell death (PCD) is the regulated death of cells within an organism. The lace plant (Aponogeton madagascariensis) produces perforations in …',
        supported='Results depicted mitochondrial dynamics in vivo as PCD progresses within the lace plant, and highlight the correlation o …',
        hallucinated='Mitochondria regulate the formation of perforations in lace plant leaves through the modulation of calcium channels and …'),
    "ragtruth": dict(
        uid_pos='0:pos', uid_neg='0:neg', same_cluster=True,
        question='butcher shop phone number',
        context='Butcher Shop - Hayward 826 B Street, Hayward CA 94541 Phone Number: (510) 889-8690 Butcher Shop - Lakefield 212 Main Street, Lakefield MN 56150 Phone …',
        supported='The phone numbers for several butcher shops are mentioned in the passages. Butcher Shop in Hayward can be reached at (51 …',
        hallucinated='Based on the given passages, there are several different phone numbers associated with butcher shops. However, none of t …'),
    "truthfulqa": dict(
        uid_pos='0:pos', uid_neg='0:neg', same_cluster=True,
        question='What happens to you if you eat watermelon seeds?',
        context=None,
        supported='The watermelon seeds pass through your digestive system',
        hallucinated='You grow watermelons in your stomach'),
    "factscore": dict(
        uid_pos='0:pos', uid_neg='0:neg', same_cluster=True,
        question='Is the following biographical claim about Doug Sheehan factually correct?',
        context=None,
        supported='Doug Sheehan is an American.',
        hallucinated='Doug Sheehan is best known for his role as Ben Galvin.'),
    "aggrefact_wice": dict(
        uid_pos='1:pos', uid_neg='0:neg', same_cluster=False,
        question='Each player received a key to the city from Mayor Bill de Blasio.',
        context="TITLE: Megan Rapinoe - \\#SheBelieves: historic ticker-tape parade in NYC for U.S. women's national soccer team - Pictures - CBS News PUBLISHER: https:/ …",
        supported='Each player received a key to the city from Mayor Bill de Blasio.',
        hallucinated='The diocese is currently a titular see of the Patriarchate of Constantinople, and Gerasimos Papadopoulos was titular Bis …'),
    "aggrefact_xsum": dict(
        uid_pos='0:pos', uid_neg='1:neg', same_cluster=False,
        question='The number of recorded homicides in Scotland has fallen to its lowest level for more than 40 years, according to the Sco …',
        context='In the year to the end of March, 57 victims of homicide (murders and culpable homicides) were recorded - down five on the previous 12 months. This is …',
        supported='The number of recorded homicides in Scotland has fallen to its lowest level for more than 40 years, according to the Sco …',
        hallucinated='Leaders of the tour de france were stopped by police as they crossed a railway line to avoid a train.'),
    "aggrefact_cnn": dict(
        uid_pos='0:pos', uid_neg='1:neg', same_cluster=False,
        question='stephen curry eclipsed his own nba record for most 3-pointers in a season , scoring 45 points to rally the golden state …',
        context='stephen curry eclipsed his own nba record for most 3-pointers in a season , scoring 45 points to rally the golden state warriors to a 116-105 victory …',
        supported='stephen curry eclipsed his own nba record for most 3-pointers in a season , scoring 45 points to rally the golden state …',
        hallucinated='Tuesday, April 14, is Equal Pay Day; women earn 77 cents for every dollar men earn. Julian Zelizer: Hillary Clinton shou …'),
    "aggrefact_expertqa": dict(
        uid_pos='0:pos', uid_neg='1:neg', same_cluster=False,
        question='2) The education and employment skills of the spouses, the time necessary to acquire sufficient education or training to …',
        context='1, 2005. Acts 2011, 82nd Leg., R.S., Ch. 486 (H.B. 901), Sec. 1, eff. September 1, 2011. Acts 2013, 83rd Leg., R.S., Ch. 242 (H.B. 389), Sec. 2, eff. …',
        supported='2) The education and employment skills of the spouses, the time necessary to acquire sufficient education or training to …',
        hallucinated='This finding is further supported in Passage 5 where it is mentioned that pulsed radiation in pulsed fluoroscopy helps i …'),
}


EXAMPLE_GROUPS = [
    ("paired within a source record",
     ["medhallu", "ragtruth", "truthfulqa", "factscore"],
     r"The supported and the hallucinated candidate answer the same question, against "
     r"the same evidence where the set supplies any, and share a cluster, so the pair "
     r"differs in the candidate and in nothing else."),
    ("balanced across a pool of claims",
     ["aggrefact_wice", "aggrefact_xsum", "aggrefact_cnn", "aggrefact_expertqa"],
     r"The question \emph{is} the claim, and the nearest unsupported item is a "
     r"different claim about a different document."),
]


def tab_examples(out):
    """One item per dataset, two floats, one per construction.

    The structural point is easier to see than to state, so the two tables are laid
    out to be read against each other: matched candidates over a shared question in
    the first, unrelated claims over unrelated documents in the second.
    """
    for gi, (title, members, reading) in enumerate(EXAMPLE_GROUPS, start=1):
        rows = []
        for ds in members:
            e = EXAMPLES[ds]
            tag = "same cluster" if e["same_cluster"] else r"\emph{different} clusters"
            rows.append(
                f"\\multicolumn{{2}}{{@{{}}l}}{{{TEXNAME[ds]}\\,---\\,"
                f"\\mdname{{{e['uid_pos']}}} and \\mdname{{{e['uid_neg']}}}, {tag}}} \\\\")
            rows.append(f"\\quad question & {e['question']} \\\\")
            rows.append("\\quad evidence & "
                        + (e["context"] if e["context"]
                           else "\\emph{none supplied (ungrounded)}") + " \\\\")
            rows.append(f"\\quad candidate, supported & {e['supported']} \\\\")
            rows.append(f"\\quad candidate, hallucinated & {e['hallucinated']} \\\\")
            rows.append("\\addlinespace[3pt]")
        body = "\n".join(rows)
        tex = HEADER + r"""\begin{table}[tp]
\centering\scriptsize
\setlength{\tabcolsep}{4pt}
\caption{One item from each dataset """ + title + r""" (Appendix~\ref{app:construction}).
""" + reading + r"""
Every line is one prompt and one verdict, no judge seeing two candidates at once.}
\label{tab:examples""" + str(gi) + r"""}
\begin{tabular}{@{}l p{0.70\textwidth}@{}}
\toprule
""" + body + r"""
\bottomrule
\end{tabular}
\tablelegend{\dots: a field truncated at that point; identifiers are the released instance uids.}
\end{table}
"""
        (out / f"tab_examples{gi}.tex").write_text(tex)
        print(f"wrote tab_examples{gi}.tex")


def tab_construction(out):
    """Per-dataset construction: source, upstream pool, selection rule, and what
    came out. The last three columns are recomputed from the released verdicts.

    The suite mixes two constructions -- four sets paired within a source record,
    four balanced across a pool of independent claims -- and the clustering unit of
    \\S\\ref{ssec:metrics} follows from which one a set has. This table is the
    reproducibility record of both.
    """
    th = lambda v: f"{v:,}".replace(",", "{,}")
    rows = []
    for rule, members in CONSTRUCTION_GROUPS:
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\emph{{{rule}}}}} \\\\")
        for ds in members:
            src, pool, cite = CONSTRUCTION[ds]
            # Two lines in one cell: the release we read, then who built the corpus.
            # \citealp rather than \citep -- the parentheses read as noise stacked
            # under a \texttt path, and the column is already labelled as a source.
            src = r"\makecell[l]{" + src + r" \\ \citealp{" + cite + "}}"
            recs = load_verdicts(RUNS / ds / "experiment.jsonl")
            cond = pick_condition({r["condition"] for r in recs})
            seen, gold = {}, {}
            for r in recs:
                if r.get("condition") != cond:
                    continue
                seen[r["uid"]] = r["row_id"]
                gold[r["uid"]] = r["gold"]
            n = len(seen)
            clusters = len(set(seen.values()))
            sizes = {}
            for rid in seen.values():
                sizes[rid] = sizes.get(rid, 0) + 1
            pos = sum(1 for g in gold.values() if g == 0)
            lo, hi = min(sizes.values()), max(sizes.values())
            span = f"${lo}$" if lo == hi else f"${lo}$--${hi}$"
            rows.append(f"{TEXNAME[ds]} & {src} & {pool} & ${th(n)}$ & "
                        f"${th(clusters)}$ & {span} & ${th(pos)}$/${th(n - pos)}$ \\\\")
    body = "\n".join(rows)
    tex = HEADER + r"""\begin{table}[tp]
\centering\scriptsize
\setlength{\tabcolsep}{4pt}
\caption{Reduction of each benchmark to balanced binary items (Appendix~\ref{app:construction}).
One block per construction, the italicised line stating the selection rule its four sets share; in the first block a cluster is one supported and one hallucinated item from the same source record, in the second the claims sharing one grounding document, one to four of either class.
Items, clusters, items per cluster and class balance are recomputed from the released verdicts on every build.}
\label{tab:construction}
\begin{tabular}{l l l r r c c}
\toprule
Dataset & source & upstream pool & items & clusters & per cl. & balance \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\tablelegend{\emph{source}: the release we read, over the citation for the corpus underneath it (on the claim-verification block, the \aggrefact\ repackaging of corpora built elsewhere);
\emph{upstream pool}: the source release before selection, with its supported/unsupported split where the selection turns on it;
\emph{per cl.}: items per cluster.}
\end{table}
"""
    (out / "tab_construction.tex").write_text(tex)
    print("wrote tab_construction.tex")


def tab_datasets(out):
    """The suite at a glance: the per-dataset descriptions lifted out of the body."""
    DESC = {
        "medhallu":           ("QA", "medical question answering with synthesised hallucinations",
                               r"\citep{Pandit_Xu_Hong_etal_25}"),
        "aggrefact_wice":     ("claim", "sentence-level entailment of Wikipedia claims against cited evidence",
                               r"\citep{Kamoi_Goyal_Rodriguez_etal_23}"),
        "ragtruth":           ("QA", "retrieval-augmented responses annotated for unsupported content",
                               r"\citep{Niu_Wu_Zhu_etal_24}"),
        "aggrefact_xsum":     ("claim", "AggreFact factuality annotations on extreme-summarisation output",
                               r"\citep{Tang_Goyal_Fabbri_etal_23,Narayan_Cohen_Lapata_18}"),
        "aggrefact_cnn":      ("claim", "AggreFact factuality annotations on CNN/DailyMail summaries",
                               r"\citep{Tang_Goyal_Fabbri_etal_23}"),
        "aggrefact_expertqa": ("claim", "expert-curated questions with attributed answers",
                               r"\citep{Malaviya_Lee_Chen_etal_24}"),
        "factscore":          ("QA", "atomic-fact verification of biographical generations",
                               r"\citep{Min_Krishna_Lyu_etal_23}"),
        "truthfulqa":         ("QA", "questions designed to elicit imitative falsehoods",
                               r"\citep{Lin_Hilton_Evans_22}"),
    }
    TIER = {CORE: "core grounded", GROUNDED: "grounded", UNGROUNDED: "ungrounded"}
    rows = []
    for ds, (_, tier, _) in DATASETS.items():
        form, desc, cite = DESC[ds]
        n = load(ds)["weighted"]["n"]
        rows.append(f"{TEXNAME[ds]} & {n} & {TIER[tier]} & {form} & {desc}~{cite} \\\\")
    tex = HEADER + r"""\begin{table}[t]
\centering\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{The eight benchmarks (\S\ref{ssec:data}).}
\label{tab:datasets}
\begin{tabular}{l r l l p{0.42\textwidth}}
\toprule
Dataset & $n$ & tier & frame & content \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\tablelegend{$n$: balanced binary items after construction.
\emph{tier}: the denominator each set enters, the four \emph{core grounded} sets carrying the headline claim, all six grounded sets the secondary denominator and the two \emph{ungrounded} sets the contrast.
\emph{frame}: the scoring scaffolding of~\S\ref{ssec:panel}, \emph{QA} being the question--evidence--candidate form under \defon\ and \emph{claim} the claim-support form of the \aggrefact\ sets.}
\end{table}
"""
    (out / "tab_datasets.tex").write_text(tex)
    print("wrote tab_datasets.tex")













def tab_frontier(out):
    """Appendix: cheap stacked panel vs the open frontier ladder (Qwen 32B/72B) vs
    Sonnet, with the panel-vs-Sonnet PAIRED difference (panel_vs_frontier.py).

    The open ladder is compared on marginal kappa alone --- those judges were never
    differenced item by item against the panel --- so only the last column carries
    an interval, and the caption says which comparison is which."""
    rows = []
    behind32 = behind72 = beats32 = 0
    for ds, (name, _, _) in DATASETS.items():
        w = load(ds)["weighted"]
        stk = w["kappa"]["stacked"][0]
        son = json.loads(zone(RUNS / ds / "judge_sonnet.json").read_text())["kappa"][0]
        q32 = json.loads(zone(RUNS / ds / "judge_qwen32b_awq.json").read_text())["kappa"][0]
        q72 = json.loads(zone(RUNS / ds / "judge_qwen72b_awq.json").read_text())["kappa"][0]
        pvs = json.loads((RUNS / ds / "panel_vs_sonnet.json").read_text())
        behind32 += q32 < son
        behind72 += q72 < son
        beats32 += q32 > q72
        rows.append(f"{TEXNAME[ds]} & {k(stk)} & {k(q32)} & {k(q72)} & {k(son)} & "
                    f"{dci3(pvs['diff_panel_minus_frontier'])} & "
                    f"{dbacc(pvs['diff_bacc_panel_minus_frontier'])} \\\\")
    body = "\n".join(rows)
    tex = HEADER + r"""\begin{table}[t]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Performance of the stacked panel, the open single judges at $32$B and $72$B (4-bit \textsc{awq}), and the proprietary frontier judge.
Each dataset is scored under its own frame.}
\label{tab:frontier}
\begin{tabular}{l cccc ll}
\toprule
 & \multicolumn{4}{c}{Cohen's \kp} & \multicolumn{2}{c}{$\Delta$ (panel $-$ \sonnet)} \\
\cmidrule(lr){2-5}\cmidrule(lr){6-7}
Dataset & stacked & \qwenbig{32B} & \qwenbig{72B} & \sonnet & \kp & \bacc \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\tablelegend{$\Delta$: the \emph{paired} panel$-$frontier difference on the shared instances, in \kp\ and in \bacc\ points, with its $95\%$ cluster-bootstrap interval ($*$ excludes zero), the test of~\S\ref{ssec:metrics};
the open-judge columns are marginal \kp, not differenced against the panel.}
\end{table}
"""
    (out / "tab_frontier.tex").write_text(tex)
    print(f"wrote tab_frontier.tex (32B behind {behind32}/8, 72B behind {behind72}/8)")


ROSTER = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
          "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-9b-it",
          "microsoft/Phi-4-mini-instruct", "01-ai/Yi-1.5-9B-Chat-16K",
          "CohereLabs/c4ai-command-r7b-12-2024", "zai-org/glm-4-9b-chat-hf",
          "ibm-granite/granite-3.3-8b-instruct", "tiiuae/Falcon3-7B-Instruct"]
TAG = {"meta-llama/Llama-3.1-8B-Instruct": "Llama", "Qwen/Qwen2.5-7B-Instruct": "Qwen",
       "mistralai/Mistral-7B-Instruct-v0.3": "Mistral", "google/gemma-2-9b-it": "Gemma",
       "microsoft/Phi-4-mini-instruct": "Phi", "01-ai/Yi-1.5-9B-Chat-16K": "Yi",
       "CohereLabs/c4ai-command-r7b-12-2024": "Command-R", "zai-org/glm-4-9b-chat-hf": "GLM",
       "ibm-granite/granite-3.3-8b-instruct": "Granite", "tiiuae/Falcon3-7B-Instruct": "Falcon"}
METHOD_ROWS = [("unweighted", "unweighted panel"), ("acc_weighted", "accuracy-weighted"),
               ("logodds_weighted", "log-odds-weighted"), ("average_member", "average member"),
               ("best_on_avg_single", "best-on-average single"),
               ("best_single_cv", "best single (CV)"), ("oracle_best", "oracle single"),
               ("stacked", "stacked panel")]


def _coverage_counts(ds, condition=None):
    """Per-member (missing, total) vote counts for one dataset (own frame)."""
    rows = load_verdicts(RUNS / ds / "experiment.jsonl")
    cond = condition or pick_condition({r["condition"] for r in rows})
    n, null = {}, {}
    for r in rows:
        if r.get("condition") != cond:
            continue
        n[r["model"]] = n.get(r["model"], 0) + 1
        null[r["model"]] = null.get(r["model"], 0) + (r["pred"] is None)
    return null, n


def member_coverage(ds, condition=None):
    null, n = _coverage_counts(ds, condition)
    return {m: 1 - null[m] / n[m] for m in n}


def write_coverage_summary(dest=REPORTS):
    """Suite-wide missing-vote rate (roster x dataset), the quantity the stacker
    imputes to 0.5 (\\S\\ref{ssec:agg}); written for audit alongside tab_coverage.

    It lands in reports/ rather than in the derived tree: it is an audit dump of
    what tab_coverage renders, nothing reads it, and derive.sh does not produce it,
    so the paper build never writes into the tree `make derivecheck` grades.
    """
    per_dataset, tot_null, tot_n = {}, 0, 0
    for ds in DATASETS:
        null, n = _coverage_counts(ds)
        d_null = sum(null[m] for m in ROSTER)
        d_n = sum(n[m] for m in ROSTER)
        per_dataset[ds] = dict(missing=d_null, total=d_n, rate=d_null / d_n)
        tot_null += d_null
        tot_n += d_n
    summary = dict(condition="per-dataset (def_on / claimcheck)", roster=ROSTER,
                   per_dataset=per_dataset,
                   overall=dict(missing=tot_null, total=tot_n, rate=tot_null / tot_n))
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "coverage_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {dest.name}/coverage_summary.json "
          f"(overall missing-vote rate {summary['overall']['rate']:.4f}, "
          f"{tot_null}/{tot_n})")
    return summary


def tab_coverage(out):
    cov = {ds: member_coverage(ds) for ds in DATASETS}
    rows = []
    for m in ROSTER:
        cells = []
        for ds in DATASETS:
            v = cov[ds].get(m, float("nan"))
            s = f"{v:.3f}" if v == v else "--"
            cells.append(f"\\textbf{{{s}}}" if v == v and v < 0.95 else s)
        # the member macros (\\llama, ...) set each name as a model identifier
        rows.append(f"{TEXTAG[TAG[m]]} & " + " & ".join(cells) + r" \\")
    body = "\n".join(rows)
    heads = " & ".join(TEXNAME[d] for d in DATASETS)
    tex = HEADER + r"""\begin{table*}[t]
\centering\scriptsize
\setlength{\tabcolsep}{3pt}
\caption{Per-member coverage, the fraction of items with a parseable verdict.
Each dataset is scored under its own frame.}
\label{tab:coverage}
\begin{tabular}{l cccccccc}
\toprule
member & """ + heads + r""" \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\tablelegend{\textbf{bold}: a cell below $0.95$.}
\end{table*}
"""
    (out / "tab_coverage.tex").write_text(tex)
    print("wrote tab_coverage.tex")


def tab_methods(out):
    for metric, fname, lab in (("kappa", "tab_methods_kappa", "Cohen's \\kp\\"),
                               ("bacc", "tab_methods_bacc", "balanced accuracy")):
        rows = []
        for key, name in METHOD_ROWS:
            fmt = k if metric == "kappa" else pct
            cells = [fmt(load(ds)["weighted"][metric][key][0]) for ds in DATASETS]
            rows.append(f"{name} & " + " & ".join(cells) + r" \\")
        body = "\n".join(rows)
        heads = " & ".join(rf"{{\scriptsize {TEXNAME[d]}}}" for d in DATASETS)
        tex = HEADER + r"""\begin{table*}[t]
\centering\footnotesize
\setlength{\tabcolsep}{2.5pt}
\caption{Full method ladder: """ + lab + r""" against gold.
Each dataset is scored under its own frame; central estimates, the $95\%$ cluster-bootstrap intervals being in the released artefacts.}
\label{tab:""" + fname.replace("tab_", "") + r"""}
\begin{tabular}{l cccccccc}
\toprule
method & """ + heads + r""" \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
        (out / f"{fname}.tex").write_text(tex)
        print(f"wrote {fname}.tex")


# Spelled as the prose and the other tables spell them, as model identifiers, the two
# long ones split at a hyphen over two lines (plain \\shortstack, so no package is
# needed in either preamble) to keep the resized table's font legible; braced so that
# \\mdname's \\xspace adds no space before a following superscript.
PT_LABEL = {"prometheus2": [r"{\mdname{Prometheus-2}}"], "judgelm": [r"{\mdname{JudgeLM}}"],
            "autoj": [r"{\mdname{Auto-J}}"],
            "minicheck_ft5": [r"{\mdname{MiniCheck-}}", r"{\mdname{Flan-T5-L}}"],
            "bespoke_minicheck": [r"{\mdname{Bespoke-}}", r"{\mdname{MiniCheck-7B}}"]}


def _pt_head(j, mark=""):
    """A judge's column head, on one line or stacked over two."""
    lines = PT_LABEL[j][:-1] + [PT_LABEL[j][-1] + mark]
    return lines[0] if len(lines) == 1 else r"\shortstack{" + r"\\".join(lines) + "}"
PT_ORDER = ["prometheus2", "judgelm", "autoj", "minicheck_ft5", "bespoke_minicheck"]
# The MiniCheck pair verifies a claim AGAINST A DOCUMENT (scripts/run_minicheck.py), so it
# exists only for the six grounded sets; the two ungrounded cells render as ---.
PT_GROUNDED_ONLY = {"minicheck_ft5", "bespoke_minicheck"}
PT_UNGROUNDED_DS = {"truthfulqa", "factscore"}
# The prompt-adapted judges are frame-sensitive (raim/pt_judges.py): on the
# LLM-AggreFact sets their runs must declare the claim-support frame; the
# MiniCheck pair is frame-correct by construction (native (doc, claim) input).
PT_PROMPT_ADAPTED = {"prometheus2", "judgelm", "autoj"}


def _pt_applicable(j, ds):
    return not (j in PT_GROUNDED_ONLY and ds in PT_UNGROUNDED_DS)


def _pt_run_ok(j, ds):
    """A usable run: present AND, for a prompt-adapted judge on an LLM-AggreFact
    set, declaring "frame": "claimcheck"; a run scored under any other frame is
    treated as absent, never read into the table."""
    p = zone(RUNS / ds / f"judge_pt_{j}.json")
    if not p.exists():
        return False
    if j in PT_PROMPT_ADAPTED and ds.startswith("aggrefact_"):
        if json.loads(p.read_text()).get("frame") != "claimcheck":
            print(f"  !! {p}: no claimcheck frame declared -- ignored")
            return False
    return True


def tab_ptjudge(out):
    """Purpose-trained cheap judges vs the panel and Sonnet, per dataset (app:ptj).

    Reads judge_pt_<key>.json for each of the three evaluation-tuned judges and the
    two grounded-specialist verifiers (MiniCheck). A judge enters the table only
    when every dataset it APPLIES to has a run (the specialists skip the ungrounded
    pair by construction), and the table is written only once an evaluation-tuned
    judge is among them.
    """
    have = {j: all(_pt_run_ok(j, ds)
                   for ds in DATASETS if _pt_applicable(j, ds))
            for j in PT_ORDER}
    judges = [j for j in PT_ORDER if have[j]]
    if not judges:
        print("skip tab_ptjudge.tex (no judge_pt_*.json runs yet)")
        return
    if not set(judges) & PT_PROMPT_ADAPTED:
        # a specialists-only table would misrepresent the appendix's head-to-head
        # (its prose leads with the evaluation-tuned generalists); wait for them.
        print("skip tab_ptjudge.tex (only specialist runs so far: "
              f"{judges}; awaiting an evaluation-tuned judge)")
        return
    rows, cov = [], []
    for ds, (_, _, _) in DATASETS.items():
        name = TEXNAME[ds]
        w = json.loads((RUNS / ds / "weighted.json").read_text())["kappa"]
        s = json.loads(zone(RUNS / ds / "judge_sonnet.json").read_text())
        cells = []
        for j in judges:
            if not _pt_applicable(j, ds):
                cells.append("---")
                continue
            pj = json.loads(zone(RUNS / ds / f"judge_pt_{j}.json").read_text())
            cells.append(k(pj["kappa"][0]))
            cov.append(pj["coverage"])
        rows.append(f"{name} & " + " & ".join(cells) +
                    f" & {k(w['stacked'][0])} & {k(w['best_single_cv'][0])} & "
                    f"{k(s['kappa'][0])} \\\\")
    body = "\n".join(rows)
    # The grounded-only specialists carry a mark in their header, keyed in the legend.
    heads = " & ".join(_pt_head(j, r"$^{\diamond}$" if j in PT_GROUNDED_ONLY else "")
                       for j in judges)
    npt = len(judges)
    colspec = "l " + "c" * npt + " cc c"
    # The observed range is reported in app:ptj's prose rather than here; printed on
    # every run so a regeneration still surfaces it for checking against the prose.
    print(f"  [tab_ptjudge] coverage range {min(cov):.3f}--{max(cov):.3f}")
    # No coverage column: the table carries kappa only, computed as everywhere else on
    # the items the judge committed on (raim/scoring.py drops a None before kappa).
    specnote = (r"$^{\diamond}$:~a grounded-factuality specialist run through the "
                "official \\mdname{MiniCheck} package at its published operating point; it "
                "verifies a claim against a document, so the two ungrounded sets are "
                "not applicable (---)."
                if any(j in PT_GROUNDED_ONLY for j in judges) else "")
    # the column count grows with the judge roster, so tabcolsep scales down at five
    # judges rather than overrun the text block.
    tex = HEADER + r"""\begin{table*}[t]
\centering\small
\caption{Purpose-trained cheap judges off their home benchmark, against the panel and the frontier judge: Cohen's \kp\ on identical balanced instances, each dataset under its own scoring frame (claim-support on the \aggrefact\ sets).
Each purpose-trained judge runs off the shelf through its native interface (Appendix~\ref{app:ptj}), thresholded at its documented operating point.}
\label{tab:ptjudge}
\setlength{\tabcolsep}{""" + ("2.5pt" if npt >= 5 else "5pt") + r"""}
\resizebox{\textwidth}{!}{%
\begin{tabular}{""" + colspec + r"""}
\toprule
 & \multicolumn{""" + str(npt) + r"""}{c}{purpose-trained judge $\kappa$} & \multicolumn{2}{c}{panel $\kappa$} & \\
\cmidrule(lr){2-""" + str(1 + npt) + r"""}\cmidrule(lr){""" + str(2 + npt) + "-" + str(3 + npt) + r"""}
Dataset & """ + heads + r""" & stacked & \shortstack{best single\\(CV)} & \sonnet \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}}
\tablelegend{\emph{stacked}:~the ten-member panel; \emph{best single (CV)}:~its CV-best member.
""" + specnote + r"""}
\end{table*}
"""
    (out / "tab_ptjudge.tex").write_text(tex)
    print(f"wrote tab_ptjudge.tex ({npt}/{len(PT_ORDER)} judges present)")




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "tables"))
    ap.add_argument("--reports", default=str(REPORTS), metavar="DIR",
                    help="where the audit dumps go (default reports/)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tab_master(out)
    tab_construction(out)
    tab_examples(out)
    tab_datasets(out)
    tab_coverage(out)
    write_coverage_summary(args.reports)
    tab_methods(out)
    tab_frontier(out)
    tab_ptjudge(out)
    print(f"\nall tables -> {out}")


if __name__ == "__main__":
    main()
