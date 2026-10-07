#!/usr/bin/env python
r"""Appendix analyses: the cross-dataset competence/correlation predictor, the
Dawid--Skene unsupervised-aggregator baseline (\\S\\ref{ssec:otheragg}), the MedHallu
common-denominator coverage check (\\S\\ref{ssec:confound}), and the Holm-adjusted
multiplicity check across the four core (stacked - best-single) contrasts
(\\S\\ref{ssec:metrics}). All additive: reuses raim_lib.weighting.crossfit and the
cluster-bootstrap protocol, no change to the frozen artefacts.

The Dawid--Skene fit is the zone-3 half and is done on the derive side: with
--derive this refits it and writes derived/_extras/baseline_stats.json (every
number the baseline console report prints, incl. the Holm p-values quoted in
\\S\\ref{ssec:metrics}); without --derive, the report and the tables are rendered
from that JSON, so the paper build neither refits nor writes into zone 3.

From scripts/, in the README's environment:
    ../.venv/bin/python3 analyze_extras.py --derive   # the fit -> baseline_stats.json
    ../.venv/bin/python3 analyze_extras.py           # console + tab_predictor.tex
    ../.venv/bin/python3 analyze_extras.py --tex      # also tab_aggregators.tex
"""
from __future__ import annotations
import argparse
import collections
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

# raim_lib sits one level up; put the repository root on the path so this
# script runs the same whether driven by make or invoked directly.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from raim_lib.verdicts import load as load_verdicts
from raim_lib.weighting import crossfit
from paper_tables import DATASETS, CORE_ORDER, TEXNAME, load, HEADER, RUNS
from paper_figures import mean_corr, spread

B, SEED = 2000, 0



# Subsets the cross-dataset predictor is read on. CNN and ExpertQA are the two sets
# held back from the primary tally in \S\ref{ssec:data}; dropping them is the
# sensitivity a reader will run, so we report it rather than leave it to be found.
PRED_SUBSETS = [("all eight", ()),
                ("without CNN", ("aggrefact_cnn",)),
                ("without CNN and ExpertQA", ("aggrefact_cnn", "aggrefact_expertqa"))]


def predictor(out=None):
    """The exploratory cross-dataset predictor, each coordinate on its own and the
    composite, over the three subsets. Reported so that the correlation coordinate's
    weakness at the dataset level is on the page rather than left to be discovered:
    it is the item-level test of analyze_mechanism.py that carries that axis, not
    this one. Writes tab_predictor.tex so no number here is hand-transcribed."""
    from scipy.stats import spearmanr

    res = {}
    for label, drop in PRED_SUBSETS:
        ds_list = [d for d in DATASETS if d not in drop]
        g = np.array([load(d)["weighted"]["diffs"]["stacked-best_single_cv"][0]
                      for d in ds_list])
        phi = np.array([mean_corr(d) for d in ds_list])
        spr = np.array([float(spread(d)) for d in ds_list])
        comp = (1 - phi) * spr
        row = {"n": len(ds_list), "datasets": ds_list}
        for key, x in (("phi", phi), ("spread", spr), ("composite", comp)):
            r = spearmanr(x, g)
            row[key] = {"rho": float(r.statistic), "p": float(r.pvalue)}
        res[label] = row
        print(f"predictor [{label:26s} n={row['n']}]  "
              + "  ".join(f"rho({k})={row[k]['rho']:+.2f} (p={row[k]['p']:.2f})"
                          for k in ("phi", "spread", "composite")))

    # the same two coordinates against the panel's absolute kappa, which is the
    # comparison that looks strong and is confounded by task difficulty
    ds_list = list(DATASETS)
    kap = np.array([load(d)["weighted"]["kappa"]["stacked"][0] for d in ds_list])
    phi = np.array([mean_corr(d) for d in ds_list])
    spr = np.array([float(spread(d)) for d in ds_list])
    res["absolute"] = {k: {"rho": float(spearmanr(x, kap).statistic),
                           "p": float(spearmanr(x, kap).pvalue)}
                       for k, x in (("phi", phi), ("spread", spr))}
    print(f"predictor [absolute stacked kappa, n={len(ds_list)}]  "
          f"rho(phi)={res['absolute']['phi']['rho']:+.2f}  "
          f"rho(spread)={res['absolute']['spread']['rho']:+.2f}")

    if out is not None:
        _predictor_tex(Path(out), res)
    return res


def _predictor_tex(out, res):
    def cell(d):
        return f"${d['rho']:+.2f}$ ({d['p']:.2f})"
    L = [HEADER, r"\begin{table}[t]", r"\centering\small",
         r"\caption{Cross-dataset predictor of the panel's gain over its best single judge. "
         r"Each coordinate is taken separately and as the composite; rows drop, in turn, "
         r"the two sets held back in~\S\ref{ssec:data}.}",
         r"\label{tab:predictor}",
         r"\setlength{\tabcolsep}{6pt}",
         r"\begin{tabular}{l c c c c}", r"\toprule",
         r" & & \multicolumn{3}{c}{Spearman $\rho$ (two-sided $p$) against the "
         r"\mbox{(stacked $-$ best-single)} \kp\ gap} \\",
         r"\cmidrule(lr){3-5}",
         r"subset & $n$ & $\bar\phi$ & members $\kp\geq0.30$ "
         r"& $(1-\bar\phi)\times$ members \\",
         r"\midrule"]
    for label, _ in PRED_SUBSETS:
        r = res[label]
        L.append(f"{label} & {r['n']} & {cell(r['phi'])} & {cell(r['spread'])} "
                 f"& {cell(r['composite'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\tablelegend{$n$: number of datasets left in the subset.}",
          r"\end{table}", ""]
    (out / "tab_predictor.tex").write_text("\n".join(L) + "\n")
    print("wrote tab_predictor.tex")


# ---- Dawid--Skene EM (binary, missing-vote tolerant) -------------------------
def dawid_skene(votes, uids, models, n_iter=100, tol=1e-6):
    """Unsupervised EM over the K-annotator vote matrix; returns MAP label per uid.
    votes[m][u] in {0,1,None}; no gold used."""
    M = models
    # init posterior T from majority vote
    T = np.zeros((len(uids), 2))
    for i, u in enumerate(uids):
        vs = [votes[m].get(u) for m in M]
        vs = [v for v in vs if v is not None]
        p1 = np.mean(vs) if vs else 0.5
        T[i] = [1 - p1, p1]
    prev = None
    for _ in range(n_iter):
        # M-step: class prior + per-annotator confusion pi[m][true][obs]
        prior = T.mean(axis=0) + 1e-9
        prior /= prior.sum()
        pi = {}
        for m in M:
            c = np.full((2, 2), 1e-9)
            for i, u in enumerate(uids):
                v = votes[m].get(u)
                if v is None:
                    continue
                c[:, v] += T[i]
            pi[m] = c / c.sum(axis=1, keepdims=True)
        # E-step
        newT = np.zeros_like(T)
        for i, u in enumerate(uids):
            logp = np.log(prior).copy()
            for m in M:
                v = votes[m].get(u)
                if v is None:
                    continue
                logp += np.log(pi[m][:, v])
            logp -= logp.max()
            p = np.exp(logp)
            newT[i] = p / p.sum()
        delta = np.abs(newT - T).max()
        T = newT
        if prev is not None and delta < tol:
            break
        prev = delta
    return {u: int(T[i, 1] >= 0.5) for i, u in enumerate(uids)}


def load_ds(ds):
    raw = RUNS / ds / "experiment.jsonl"
    rows = load_verdicts(raw)
    # each dataset under its own frame (def_on for QA-form, claimcheck for the
    # LLM-AggreFact sets); mirrors paper_tables.pick_condition.
    conds = {r["condition"] for r in rows}
    cond = "def_on" if "def_on" in conds else sorted(conds)[0]
    rows = [r for r in rows if r["condition"] == cond]
    models = sorted({r["model"] for r in rows})
    votes = {m: {} for m in models}
    gold, rowid_of = {}, {}
    for r in rows:
        votes[r["model"]][r["uid"]] = r["pred"]
        gold[r["uid"]] = r["gold"]
        rowid_of[r["uid"]] = r["row_id"]
    uids = sorted(gold)
    return votes, gold, rowid_of, uids, models


def kappa_on(pred, gold, sel):
    yt, yp = [], []
    for u in sel:
        p = pred.get(u)
        if p is not None:
            yt.append(gold[u]); yp.append(p)
    if len(yt) < 2 or len(set(yt)) < 2:
        return float("nan")
    return cohen_kappa_score(yt, yp)


def analyse_baselines(ds):
    """Per-dataset Dawid--Skene fit + the common-denominator and multiplicity
    bootstrap quantities, all against the same cross-fitted stacker predictions."""
    votes, gold, rowid_of, uids, models = load_ds(ds)
    preds, _ = crossfit(votes, gold, uids, rowid_of, folds=5, seed=SEED)
    preds["dawid_skene"] = dawid_skene(votes, uids, models)

    row_to_uids = defaultdict(list)
    for u in uids:
        row_to_uids[rowid_of[u]].append(u)
    row_ids = sorted(row_to_uids)
    R = len(row_ids)

    # common denominator for the best_single_cv contrast = where that baseline voted
    common = [u for u in uids if preds["best_single_cv"].get(u) is not None]

    rng = np.random.default_rng(SEED)
    methods = ["stacked", "best_single_cv", "unweighted", "dawid_skene"]
    acc = defaultdict(list)
    for _ in range(B):
        pick = rng.integers(0, R, size=R)
        sel = [u for p in pick for u in row_to_uids[row_ids[p]]]
        k = {m: kappa_on(preds[m], gold, sel) for m in methods}
        for m in methods:
            acc[("k", m)].append(k[m])
        acc[("d", "stacked", "dawid_skene")].append(k["stacked"] - k["dawid_skene"])
        acc[("d", "dawid_skene", "best_single_cv")].append(
            k["dawid_skene"] - k["best_single_cv"])
        acc[("d", "stacked", "best_single_cv")].append(
            k["stacked"] - k["best_single_cv"])
        # common-denominator stacked - best_single_cv
        selc = [u for u in sel if u in set(common)]
        ks = kappa_on(preds["stacked"], gold, selc)
        kb = kappa_on(preds["best_single_cv"], gold, selc)
        acc[("dc", "stacked", "best_single_cv")].append(ks - kb)

    def ci(key):
        a = np.array(acc[key]); a = a[~np.isnan(a)]
        return [float(np.mean(a)), float(np.percentile(a, 2.5)),
                float(np.percentile(a, 97.5))]

    def pval_one_sided(key):
        a = np.array(acc[key]); a = a[~np.isnan(a)]
        # H0: difference <= 0; p = fraction of bootstrap draws at or below 0
        return float(np.mean(a <= 0.0))

    committed_best = sum(preds["best_single_cv"].get(u) is not None for u in uids)
    return dict(
        ds=ds, n=len(uids), R=R, committed_best=committed_best, n_common=len(common),
        kappa={m: ci(("k", m)) for m in methods},
        d_stacked_ds=ci(("d", "stacked", "dawid_skene")),
        d_ds_best=ci(("d", "dawid_skene", "best_single_cv")),
        d_stacked_best_full=ci(("d", "stacked", "best_single_cv")),
        d_stacked_best_common=ci(("dc", "stacked", "best_single_cv")),
        p_stacked_best=pval_one_sided(("d", "stacked", "best_single_cv")),
    )


def holm(pvals):
    """Holm step-down adjusted p-values, preserving input order."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    run = 0.0
    for r, i in enumerate(idx):
        v = (m - r) * pvals[i]
        run = max(run, v)
        adj[i] = min(run, 1.0)
    return adj


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def baselines(out, write_latex, derive=False):
    """The Dawid--Skene report and its tables, from a fit that happens on the
    derive side.

    The fit is 4.5 minutes of Dawid--Skene expectation-maximisation and B = 2000
    cluster bootstrap over eight datasets. None of it is rendering: it produces the
    numbers that `write_baseline_json` dumps losslessly into zone 3, that
    `paper_tables.tab_master` READS for its Dawid--Skene column, and that the tables
    below select fields from. So it belongs where the rest of zone 3 is computed:
    `--derive` fits and writes the JSON, the default path renders from it in under a
    second, and baseline_stats.json is checked by `make derivecheck` like every
    other derived artefact.
    """
    order = list(CORE_ORDER) + ["factscore", "truthfulqa",
                                "aggrefact_cnn", "aggrefact_expertqa"]
    core = list(CORE_ORDER)
    if derive:
        res = {ds: analyse_baselines(ds) for ds in order}
        praw = [res[ds]["p_stacked_best"] for ds in core]
        padj = holm(praw)
        write_baseline_json(res, order, core, praw, padj)
    else:
        res, praw, padj = read_baseline_json(order, core)

    print("\n=== Dawid--Skene (unsupervised aggregator) vs stacker / single ===")
    print(f"{'dataset':10} {'DS kappa':>20} {'stacked-DS':>22} {'DS-bestCV':>22}")
    for ds in order:
        r = res[ds]
        print(f"{DATASETS[ds][0]:10} {fmt(r['kappa']['dawid_skene']):>20}  "
              f"{fmt(r['d_stacked_ds']):>20}  {fmt(r['d_ds_best']):>20}")

    print("\n=== MedHallu (and all) common-denominator coverage check ===")
    print(f"{'dataset':10} {'n':>5} {'n_best':>7} {'full stacked-bestCV':>26} "
          f"{'common-denom stacked-bestCV':>30}")
    for ds in order:
        r = res[ds]
        print(f"{DATASETS[ds][0]:10} {r['n']:>5} {r['committed_best']:>7} "
              f"{fmt(r['d_stacked_best_full']):>26} "
              f"{fmt(r['d_stacked_best_common']):>30}")

    print("\n=== Multiplicity: stacked - best_single_cv on the four core ===")
    print(f"{'dataset':10} {'delta':>22} {'p_raw':>8} {'p_holm':>8} {'sig@.05':>8}")
    for ds, pr, pa in zip(core, praw, padj):
        r = res[ds]
        print(f"{DATASETS[ds][0]:10} {fmt(r['d_stacked_best_full']):>22} "
              f"{pr:>8.3f} {pa:>8.3f} {'yes' if pa < 0.05 else 'no':>8}")

    if write_latex:
        write_aggregator_tex(res, order, out)


def read_baseline_json(order, core):
    """The same numbers as `analyse_baselines` produced, read back rather than refitted.

    Loud rather than lenient about a JSON that does not cover what is being asked
    of it: a table rendered from a partial artefact is the failure mode this whole
    arrangement exists to avoid, and it would otherwise present as a KeyError deep
    inside a formatter.
    """
    p = RUNS / "_extras" / "baseline_stats.json"
    if not p.exists():
        raise SystemExit(f"missing {p}\n"
                         f"  it is derived: run `analyze_extras.py --derive`, or `make derive`")
    payload = json.loads(p.read_text())
    res = payload["per_dataset"]
    missing = [ds for ds in order if ds not in res]
    if missing:
        raise SystemExit(f"{p} does not cover {missing}\n"
                         f"  re-derive it: `analyze_extras.py --derive`")
    mult = payload["multiplicity"]
    return (res,
            [mult["p_raw"][ds] for ds in core],
            [mult["p_holm"][ds] for ds in core])


def write_baseline_json(res, order, core, praw, padj):
    """Audit dump of every console-reported baseline number (Dawid-Skene, the
    common-denominator coverage check, Holm-adjusted multiplicity), so
    \\S\\ref{ssec:confound} and \\S\\ref{ssec:metrics} numbers can be audited or
    re-derived without re-running the console report."""
    dest = RUNS / "_extras"
    dest.mkdir(parents=True, exist_ok=True)
    payload = dict(
        per_dataset={ds: res[ds] for ds in order},
        multiplicity=dict(
            core_order=core,
            p_raw=dict(zip(core, praw)),
            p_holm=dict(zip(core, padj)),
            sig_at_05=dict(zip(core, [pa < 0.05 for pa in padj])),
        ),
    )
    (dest / "baseline_stats.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {dest / 'baseline_stats.json'}")




def write_aggregator_tex(res, order, out):
    """tab_aggregators: the rescue contrast and the Dawid--Skene contrast in one
    table.

    Both halves are lifted from where they already live rather than recomputed:
    the rescue deltas from each dataset's weighted.json and the Dawid--Skene deltas
    from the fit above, so no number in the table is new. Both use the CV-best
    single judge as the single-judge reference, the paper's fair primary baseline,
    so the two halves share it (the CV-best and the oracle coincide on every
    dataset quoted in the prose, differing only on WiCE and XSum, where the CV-best
    is the weaker of the two)."""
    def dcell(t):
        s = f"${t[0]:+.3f}$"
        if t[1] > 0 or t[2] < 0:
            s += r"\,*"
        return s + r"\,{\tiny[" + f"{t[1]:+.2f}" + r",\," + f"{t[2]:+.2f}" + r"]}"

    def neg(t):                       # [point, lo, hi] of -X, from that of X
        return [-t[0], -t[2], -t[1]]

    lines = [
        f"% AUTO-GENERATED by {Path(_sys.argv[0]).name} in scripts/ "
        f"-- do not edit by hand.",
        r"\begin{table*}[t]\centering\scriptsize",
        r"\caption{The stacked panel against the two label-free ways of combining the "
        r"same ten votes (Cohen's \kp, each dataset under its own frame). "
        # ssec:otheragg is a \fakepar label inside ssec:rescue, so both refs printed
        # the same section number; one pointer says the same thing without the echo.
        r"The \emph{rescue} and \emph{unsupervised aggregator} blocks are read "
        r"in~\S\ref{ssec:rescue}.}",
        r"\label{tab:aggregators}",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{l ccc rrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Cohen's \kp} & \multicolumn{2}{c}{\emph{rescue}} "
        r"& \multicolumn{2}{c}{\emph{unsupervised aggregator}} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Dataset & unw. & D--S & stk. & $\Delta_{\text{stk}-\text{unw}}$ "
        r"& $\Delta_{\text{unw}-\text{best}}$ & $\Delta_{\text{stk}-\text{DS}}$ "
        r"& $\Delta_{\text{DS}-\text{best}}$ \\",
        r"\midrule",
    ]
    for ds in order:
        w = load(ds)["weighted"]
        r = res[ds]
        lines.append(
            f"{TEXNAME[ds]} & {w['kappa']['unweighted'][0]:.3f} & "
            f"{r['kappa']['dawid_skene'][0]:.3f} & {w['kappa']['stacked'][0]:.3f} & "
            f"{dcell(w['diffs']['stacked-unweighted'])} & "
            f"{dcell(neg(w['diffs']['best_single_cv-unweighted']))} & "
            f"{dcell(r['d_stacked_ds'])} & {dcell(r['d_ds_best'])} \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\tablelegend{\emph{unw.}: unweighted majority; \emph{D--S}: Dawid--Skene, fitted by "
        r"expectation-maximisation on the same vote matrix without labels; "
        r"\emph{stk.}: stacked panel; \emph{best}: CV-best single judge. "
        r"$\Delta$: paired difference with its 95\% cluster-bootstrap interval, $*$ excluding zero.}",
        r"\end{table*}"]
    (out / "tab_aggregators.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {out/'tab_aggregators.tex'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "tables"))
    ap.add_argument("--tex", action="store_true",
                     help="also (re-)write tables/tab_aggregators.tex")
    ap.add_argument("--predictor", action="store_true",
                     help="only recompute the cross-dataset predictor and rewrite "
                          "tables/tab_predictor.tex (seconds; reads derived/ only)")
    ap.add_argument("--derive", action="store_true",
                     help="refit the baselines and rewrite "
                          "derived/_extras/baseline_stats.json (~4.5 min; this is "
                          "the zone-3 half, and derive.sh is what runs it). Without "
                          "it, the report and tables are rendered from that JSON")
    args = ap.parse_args()
    if args.predictor:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        predictor(out)
        return
    if args.derive:
        # Derivation only: the exploratory predictor writes and prints zone-4
        # material, which a `make derive` has no business touching.
        baselines(Path(args.out), write_latex=False, derive=True)
        return
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    predictor(out)
    baselines(out, args.tex)


if __name__ == "__main__":
    main()
