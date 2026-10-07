#!/usr/bin/env python
r"""Hallucinated-class F1 on the QA-form sets, for comparison with published tables.

WHY THIS EXISTS. Outside LLM-AggreFact the paper reports balanced accuracy, whereas
every published anchor on those benchmarks reports hallucinated-class F1. That is a
metric mismatch rather than an impossibility: the rows differ in slice, but F1 is
computable on ours -- and on MedHallu the published detector table includes THREE of
our own panel members, so the two tables can be lined up model by model.

WHAT IS COMPUTED. Precision, recall and F1 on the hallucinated class (gold = 1) for
the stacked panel, each of the ten members, and the frontier judge, on the two sets
carrying a published F1 anchor.

ABSTENTIONS ARE CHARGED AS MISSED DETECTIONS. A member that returns no parseable
verdict is scored as predicting "supported", so an abstention on a hallucinated item
is a false negative and never a false positive. The published detectors have full
coverage, so this is the reading that does NOT flatter us; coverage is reported
beside each row so the effect is visible rather than buried.

THE SLICE CAVEAT IS REAL AND IS NOT REMOVED BY THIS. MedHallu's published table
pools the full 10,000-pair superset whilst we score the 1,000-row pqa_labeled slice
(2,000 balanced items); RAGTruth's published numbers are on its own QA test split at
its own prevalence, ours on a balanced paired construction. Both sets are class-
balanced on our side, which is stated with the numbers.

PREVALENCE IS FLAGGED, NOT CORRECTED FOR. F1 on the positive class rises with that
class's share -- recall is invariant to it but precision is not -- so a number
measured on our 50/50 slice cannot be read beside one measured at a different
hallucinated rate. MedHallu needs no correction: its published superset is itself
paired, hence balanced. RAGTruth's anchors are scored on that corpus's QA *test*
split, which runs at 0.178 (160 of 900 responses under the is_hallucinated rule
raim/tasks.py uses), against our 0.500, so the paper reads the RAGTruth blocks side
by side without ranking them. The re-priced value (holding the measured TPR and FPR
fixed and re-deriving precision at the anchor's rate) is kept in the JSON only, as
`f1_at_anchor`, for the record.

    external_f1.py                 # -> derived/_summary/external_f1.json
    external_f1.py --tex           # + tables/tab_extf1.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from raim_lib.verdicts import load as load_verdicts
from raim_lib.verdicts import resolve as zone
from raim_lib.verdicts import DERIVED, VERDICTS
from raim_lib.weighting import crossfit

from paper_tables import pick_condition, TEXNAME, TEXTAG

# Only the sets on which a published hallucinated-class F1 anchor exists. The
# LLM-AggreFact four are covered by
# tab_extbase on the leaderboard's own metric and are not repeated here.
SETS = ["medhallu", "ragtruth"]

# The published anchors, transcribed from the cited papers; not from our artefacts.
#   dataset -> [(system, F1, note, citation)]
PUBLISHED_F1 = {
    "medhallu": [
        (r"\mdname{GPT-4o}",          0.877, "frontier, prompted",        "Pandit_Xu_Hong_etal_25"),
        (r"\mdname{Qwen2.5-14B}",     0.852, "single open judge",         "Pandit_Xu_Hong_etal_25"),
        (r"\mdname{GPT-4o-mini}",     0.841, "budget \\textsc{api}",      "Pandit_Xu_Hong_etal_25"),
        (r"\mdname{Qwen2.5-7B}",      0.839, "\\emph{our panel member}",  "Pandit_Xu_Hong_etal_25"),
        (r"\mdname{Gemma-2-9B}",      0.838, "\\emph{our panel member}",  "Pandit_Xu_Hong_etal_25"),
        (r"\mdname{Llama-3.1-8B}",    0.797, "\\emph{our panel member}",  "Pandit_Xu_Hong_etal_25"),
    ],
    # The QA-subtask columns of RAGTruth Table 5, LettuceDetect Table 2 and RAG-HAT
    # Table 2 run Precision, Recall, F1; the values below are the F1 column. The
    # SelfCheckGPT baseline of RAGTruth Table 5 runs on gpt-3.5-turbo. The RAGTruth
    # paper's own fine-tuned Llama-2-13B is the trained anchor closest to our panel.
    "ragtruth": [
        (r"\mdname{RAG-HAT} (\mdname{Llama-3-8B})", 0.748, "trained on the \\ragtruth\\ train split", "Song_Wang_Zhu_etal_24"),
        (r"\mdname{LettuceDetect-large}",      0.702, "trained on the \\ragtruth\\ train split", "Kovacs_Recski_25"),
        (r"fine-tuned \mdname{Llama-2-13B}",   0.682, "trained on the \\ragtruth\\ train split", "Niu_Wu_Zhu_etal_24"),
        (r"\mdname{GPT-4-turbo}, prompted",    0.456, "prompt-only frontier",                    "Niu_Wu_Zhu_etal_24"),
        (r"SelfCheckGPT (\mdname{GPT-3.5})",   0.437, "prompt-only, resampling",                 "Niu_Wu_Zhu_etal_24"),
    ],
}

# Hallucinated-class prevalence of the slice the PUBLISHED anchors are measured on.
# MedHallu's published table pools the 10,000-PAIR superset, so it is balanced
# exactly as ours is. RAGTruth's anchors are scored on that corpus's QA TEST split:
# 160 of the 900 QA responses in the wandb/RAGTruth-processed mirror's `test` split
# carry at least one annotated span, under the same is_hallucinated rule
# raim/tasks.py uses. (Our items come from the mirror's `train` split, whose QA pool
# runs at 1,564/5,034 = 0.311, which is not the anchors' base rate.) Used for the
# JSON's f1_at_anchor only.
ANCHOR_PREVALENCE = {"medhallu": 0.500, "ragtruth": 0.178}

ROSTER = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
          "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-9b-it",
          "microsoft/Phi-4-mini-instruct", "01-ai/Yi-1.5-9B-Chat-16K",
          "CohereLabs/c4ai-command-r7b-12-2024", "zai-org/glm-4-9b-chat-hf",
          "ibm-granite/granite-3.3-8b-instruct", "tiiuae/Falcon3-7B-Instruct"]
TAG = {m: t for m, t in zip(ROSTER, ["Llama", "Qwen", "Mistral", "Gemma", "Phi",
                                     "Yi", "Command-R", "GLM", "Granite", "Falcon"])}
# The three the MedHallu paper reports individually, so a reader can line them up.
SHARED = {"Qwen/Qwen2.5-7B-Instruct", "google/gemma-2-9b-it",
          "meta-llama/Llama-3.1-8B-Instruct"}


def prf(y, p):
    """Precision, recall, F1 on the positive (hallucinated) class."""
    y, p = np.asarray(y), np.asarray(p)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    tn = int(((p == 0) & (y == 0)).sum())
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn, tn=tn)


def f1_at(d, pi):
    """The positive-class F1 this row would attain at prevalence `pi`.

    Recall is prevalence-invariant, so we hold the measured TPR and FPR fixed and
    re-derive precision at the target base rate; where the measured prevalence and
    `pi` coincide this returns the measured F1 exactly.
    """
    tpr = d["tp"] / (d["tp"] + d["fn"]) if d["tp"] + d["fn"] else 0.0
    fpr = d["fp"] / (d["fp"] + d["tn"]) if d["fp"] + d["tn"] else 0.0
    den = tpr * pi + fpr * (1 - pi)
    if den <= 0:
        return 0.0
    prec = tpr * pi / den
    return 2 * prec * tpr / (prec + tpr) if prec + tpr else 0.0


def run(ds, folds, seed):
    rows = load_verdicts(VERDICTS / ds / "experiment.jsonl")
    cond = pick_condition({r["condition"] for r in rows})
    votes, gold, rowid = {}, {}, {}
    for r in rows:
        if r.get("condition") != cond:
            continue
        votes.setdefault(r["model"], {})[r["uid"]] = r["pred"]
        gold[r["uid"]] = r["gold"]
        rowid[r["uid"]] = r["row_id"]
    uids = sorted(gold)
    y = [gold[u] for u in uids]

    preds, _ = crossfit(votes, gold, uids, rowid, folds=folds, seed=seed)
    out = {"dataset": ds, "condition": cond, "n": len(uids),
           "prevalence": float(np.mean(y)), "members": {}}

    # abstention -> "supported": a missed detection, never a false alarm
    def committed(vm):
        return [1 if vm.get(u) == 1 else 0 for u in uids], \
               float(np.mean([vm.get(u) is not None for u in uids]))

    out["stacked"] = prf(y, [1 if preds["stacked"].get(u) == 1 else 0 for u in uids])
    out["stacked"]["coverage"] = 1.0
    out["unweighted"] = prf(y, [1 if preds["unweighted"].get(u) == 1 else 0 for u in uids])
    out["unweighted"]["coverage"] = 1.0

    for m in ROSTER:
        if m not in votes:
            continue
        p, cov = committed(votes[m])
        r = prf(y, p)
        r["coverage"] = cov
        r["shared_with_published"] = m in SHARED
        out["members"][TAG[m]] = r

    jp = zone(VERDICTS / ds / "judge_sonnet.jsonl")
    if jp.exists():
        sv = {r["uid"]: r["pred"] for r in load_verdicts(jp)
              if r.get("condition") == cond}
        p, cov = committed(sv)
        out["sonnet"] = prf(y, p)
        out["sonnet"]["coverage"] = cov

    best = max((v["f1"], k) for k, v in out["members"].items())
    out["best_member"] = {"tag": best[1], "f1": best[0]}

    # Re-price our own rows at the prevalence the published anchors sit on, so
    # that the table's two blocks are read on one base rate (see module docstring).
    pi = ANCHOR_PREVALENCE[ds]
    out["anchor_prevalence"] = pi
    for key in ("stacked", "unweighted", "sonnet"):
        if key in out:
            out[key]["f1_at_anchor"] = f1_at(out[key], pi)
    for tag, d in out["members"].items():
        d["f1_at_anchor"] = f1_at(d, pi)
    out["best_member"]["f1_at_anchor"] = f1_at(out["members"][best[1]], pi)
    return out


def write_tex(res, texdir):
    texdir.mkdir(parents=True, exist_ok=True)
    L = [f"% AUTO-GENERATED by {Path(__file__).name} in scripts/ "
         f"-- do not edit by hand; regenerate with `make tables`.",
         r"\begin{table}[t]", r"\centering\footnotesize",
         r"\caption{Hallucinated-class F1 on \medhallu\ and \ragtruth, published and measured here.",
         r"Published rows are transcribed from the cited papers (\ragtruth: response-level F1 on its QA subtask, which we adopt); ours are computed on our balanced construction (Appendix~\ref{app:construction}), an abstention counting as a missed detection.",
         r"Slices differ: \medhallu's published table pools the $10{,}000$-pair superset against our $1{,}000$-row \mdname{pqa\_labeled} slice, and \ragtruth's sits on its QA test split, at a hallucinated rate of $" + f"{ANCHOR_PREVALENCE['ragtruth']:.3f}" + r"$ against our $0.5$, so its two blocks are not comparable in level, F1 moving with prevalence.",
         r"Three published \medhallu\ detectors are members of our panel (\S\ref{sec:frontier}).}",
         r"\label{tab:extf1}",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{l l c}", r"\toprule",
         r"System & & F1 \\", r"\midrule"]   # the title names the class
    for i, ds in enumerate(SETS):
        r = res[ds]
        if i:
            L.append(r"\midrule")
        L.append(r"\multicolumn{3}{l}{" + TEXNAME[ds]
                 + r" \emph{--- published}} \\")
        for name, f1, note, key in PUBLISHED_F1[ds]:
            L.append(f"\\quad {name}~{{\\scriptsize\\citep{{{key}}}}} & "
                     f"{{\\scriptsize {note}}} & ${100*f1:.1f}$ \\\\")
        L.append(r"\multicolumn{3}{l}{\emph{measured here, no member trained on the corpus}} \\")
        # No re-priced value is printed (module docstring): the prevalence
        # difference is stated in the caption instead.
        def cell(d, bold=False):
            body = f"{100*d['f1']:.1f}"
            main = f"\\mathbf{{{body}}}" if bold else body
            return f"${main}$"

        bm = r["best_member"]
        L.append(r"\quad best member by F1 (" + TEXTAG.get(bm["tag"], bm["tag"])
                 + r") & {\scriptsize ours} & "
                 + cell(r["members"][bm["tag"]]) + r" \\")
        L.append(r"\quad \textbf{stacked cheap panel} & {\scriptsize ours} & "
                 + cell(r["stacked"], bold=True) + r" \\")
        if "sonnet" in r:
            L.append(r"\quad \sonnet & {\scriptsize ours} & "
                     + cell(r["sonnet"]) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          # no legend: the table carries no symbol or abbreviation to key
          r"\end{table}"]
    (texdir / "tab_extf1.tex").write_text("\n".join(L) + "\n")
    print(f"wrote {texdir / 'tab_extf1.tex'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--tex", action="store_true")
    ap.add_argument("--texdir", default=str(Path(__file__).resolve().parent / "tables"))
    args = ap.parse_args()

    res = {ds: run(ds, args.folds, args.seed) for ds in SETS}
    path = Path(args.out) if args.out else DERIVED / "_summary" / "external_f1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {path}\n")

    for ds in SETS:
        r = res[ds]
        print(f"=== {ds} (n={r['n']}, prevalence={r['prevalence']:.3f}) ===")
        print(f"  {'stacked panel':28} F1={r['stacked']['f1']:.3f}  "
              f"P={r['stacked']['precision']:.3f} R={r['stacked']['recall']:.3f}")
        if "sonnet" in r:
            print(f"  {'Sonnet':28} F1={r['sonnet']['f1']:.3f}  "
                  f"P={r['sonnet']['precision']:.3f} R={r['sonnet']['recall']:.3f}")
        for tag, m in sorted(r["members"].items(), key=lambda kv: -kv[1]["f1"]):
            mark = "  <- also published" if m.get("shared_with_published") else ""
            print(f"  {'member ' + tag:28} F1={m['f1']:.3f}  "
                  f"cov={m['coverage']:.3f}{mark}")
        print("  published anchors:")
        for name, f1, note, _ in PUBLISHED_F1[ds]:
            print(f"    {name:28} F1={f1:.3f}   ({note})")
        print()
    if args.tex:
        write_tex(res, Path(args.texdir))


if __name__ == "__main__":
    main()
