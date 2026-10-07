#!/usr/bin/env python
"""Cross-dataset transferability of the stacked panel (pooled + LODO).

The per-dataset headline (stacking beats the in-domain best single) presupposes
per-dataset gold labels, because the stacker is fitted with them. This experiment
asks whether a stacker fitted on OTHER datasets still beats the in-domain best
single on a held-out dataset -- if so, the per-dataset-label requirement largely
disappears and the deployment claim firms up.

Experiment A (pooled global stacker) -- fit ONE logistic regression on the pooled
member-vote rows of all supplied datasets and dump the learned coefficients. This
is for inspecting the weights only; the in-sample pooled kappa is NOT honest and
is not reported (the honest per-dataset numbers come from B).

Experiment B (leave-one-dataset-out; the headline) -- for each held-out dataset d,
fit the stacker on the union of the other datasets and predict on ALL of d. Since
d never appears in training, the predictions are leakage-free and no within-d
cross-fit is needed. The transferred stacker is then evaluated on d against the
in-domain baselines, in honesty order:
    best_single_cv      in-domain CV-best single  -- the transfer-WIN test
    oracle_best         in-domain oracle single   -- the CLEAN-transfer test
    indomain_stacked    within-d cross-fit stacker -- the transfer COST
    unweighted          in-domain unweighted panel
    best_on_avg_single  the single model best on mean kappa over the training pool
    sonnet              frontier reference (optional, from a judge_<tag>.jsonl)
with cluster-bootstrap CIs by row_id, reporting kappa (per-method denominator, as
in weighted.py) AND balanced accuracy with non-answer = error (full denominator),
plus committed-n -- mirroring the weighted.json schema.

Decision rule (extends the frozen per-dataset rule):
    transfer win    : (transferred - best_single_cv) lower CI > 0
    clean transfer  : additionally (transferred - oracle_best) lower CI > 0
    transfer cost   : (indomain_stacked - transferred); a NON-significant cost is a
                      positive result ("transfer is free").


Usage:
  python transfer_lodo.py \
      --datasets medhallu:verdicts/medhallu/experiment.jsonl \
                 ragtruth:verdicts/ragtruth/experiment.jsonl \
                 aggrefact_xsum:verdicts/aggrefact_xsum/experiment.jsonl \
                 aggrefact_wice:verdicts/aggrefact_wice/experiment.jsonl \
      --outdir derived/_transfer --B 2000 \
      --summary derived/_transfer/lodo_summary.md \
      [--sonnet-tag judge_sonnet] [--condition def_on] [--folds 5] [--tag _all8]
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score

# raim_lib sits one level up; put the repository root on the path so this
# script runs the same whether driven by make or invoked directly.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from raim_lib.verdicts import exists as verdict_exists
from raim_lib.bootstrap import _ci
from transfer import load, feat
from raim_lib.weighting import crossfit


# ----------------------------------------------------------------------------- metrics
def _kappa(yt, yp):
    if len(yt) < 2 or len(set(yt)) < 2:
        return float("nan")
    return cohen_kappa_score(yt, yp)


def full_kappa(votes_m, gold, uids):
    """Full-data kappa of a single member's votes (None dropped)."""
    yt = [gold[u] for u in uids if votes_m.get(u) is not None]
    yp = [votes_m[u] for u in uids if votes_m.get(u) is not None]
    return _kappa(yt, yp)


def kap_sel(preds_method, gold, sel):
    """kappa on a resample, per-method denominator (drop this method's None)."""
    yt, yp = [], []
    for u in sel:
        p = preds_method.get(u)
        if p is not None:
            yt.append(gold[u]); yp.append(p)
    return _kappa(yt, yp)


def bacc_sel(preds_method, gold, sel):
    """Balanced accuracy with non-answer = error, full per-class denominator."""
    tp = fn_p = tn = fn_n = 0
    for u in sel:
        p = preds_method.get(u)
        if gold[u] == 1:
            if p == 1: tp += 1
            else:      fn_p += 1
        else:
            if p == 0: tn += 1
            else:      fn_n += 1
    recs = []
    if tp + fn_p: recs.append(tp / (tp + fn_p))
    if tn + fn_n: recs.append(tn / (tn + fn_n))
    return sum(recs) / len(recs) if recs else float("nan")


# ----------------------------------------------------------------------------- guards
def assert_pooling_safe(data, names):
    """data[name] = (votes, gold, rowid). Fail loudly on roster / gold mismatch."""
    ref = sorted(data[names[0]][0])
    for n in names[1:]:
        m = sorted(data[n][0])
        if m != ref:
            only_ref = set(ref) - set(m)
            only_n = set(m) - set(ref)
            raise SystemExit(
                f"ROSTER MISMATCH: '{n}' panel differs from '{names[0]}'.\n"
                f"  only in {names[0]}: {sorted(only_ref)}\n"
                f"  only in {n}: {sorted(only_n)}\n"
                "Pooling requires identical, identically-ordered model lists.")
    for n in names:
        gold = data[n][1]
        vals = set(gold.values())
        if not vals <= {0, 1}:
            raise SystemExit(f"GOLD MISMATCH: '{n}' has labels {vals}, expected subset of {{0,1}}.")
        if vals != {0, 1}:
            raise SystemExit(f"GOLD MISMATCH: '{n}' is single-class ({vals}); cannot pool / score.")
    return ref




# ----------------------------------------------------------------------------- pooling
def pooled_rows(data, names, models, softdata=None):
    """Stack member features + gold across the named datasets -- the binary votes,
    or the members' continuous confidences when softdata is supplied."""
    Xs, ys = [], []
    for n in names:
        votes, gold, _ = data[n]
        uids = sorted(gold)
        Xs.append(feat(softdata[n] if softdata else votes, models, uids))
        ys.append(np.array([gold[u] for u in uids]))
    return np.vstack(Xs), np.concatenate(ys)


# ----------------------------------------------------------------------------- LODO core
def evaluate_heldout(d, data, names, models, args, sonnet, softdata=None):
    votes, gold, rowid = data[d]
    uids = sorted(gold)
    train_names = [n for n in names if n != d]
    soft = softdata[d] if softdata else None

    # transferred stacker: fit on the union of the other datasets, predict all of d
    Xtr, ytr = pooled_rows(data, train_names, models, softdata)
    lr = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    Xd = feat(soft if soft is not None else votes, models, uids)
    tpred = lr.predict(Xd)
    transferred = {u: int(tpred[i]) for i, u in enumerate(uids)}   # never None (0.5-imputed feats)

    # in-domain baselines on d (leakage-free cross-fit) + oracle + best-on-average
    xfit_kw = {}
    preds_d, _ = crossfit(votes, gold, uids, rowid, folds=args.folds, seed=args.seed,
                          **xfit_kw)
    oracle = max(models, key=lambda m: (full_kappa(votes[m], gold, uids), m))
    # best-on-average single: fixed model chosen by mean kappa over the TRAINING pool
    mean_k = {m: float(np.nanmean([full_kappa(data[n][0][m], data[n][1], sorted(data[n][1]))
                                   for n in train_names])) for m in models}
    boa = max(models, key=lambda m: (mean_k[m], m))

    preds = {
        "transferred_stacked": transferred,
        "indomain_stacked":    preds_d["stacked"],
        "best_single_cv":      preds_d["best_single_cv"],
        "oracle_best":         {u: votes[oracle].get(u) for u in uids},
        "unweighted":          preds_d["unweighted"],
        "best_on_avg_single":  {u: votes[boa].get(u) for u in uids},
    }
    interface = "binary"
    if sonnet and d in sonnet:
        preds["sonnet"] = sonnet[d]

    methods = list(preds)
    committed = {m: sum(preds[m].get(u) is not None for u in uids) for m in methods}

    pairs = [("transferred_stacked", "best_single_cv"),   # WIN
             ("transferred_stacked", "oracle_best"),       # CLEAN
             ("indomain_stacked", "transferred_stacked"),  # COST
             ("transferred_stacked", "unweighted"),
             ("transferred_stacked", "best_on_avg_single"),
             ("indomain_stacked", "best_single_cv")]       # in-domain win (sanity vs weighted.json)
    if "sonnet" in preds:
        pairs.append(("transferred_stacked", "sonnet"))

    # cluster bootstrap over d's row_ids
    row_to_uids = defaultdict(list)
    for u in uids:
        row_to_uids[rowid[u]].append(u)
    rids = sorted(row_to_uids)
    R = len(rids)
    rng = np.random.default_rng(args.seed)
    acc = defaultdict(list)
    for _ in range(args.B):
        pick = rng.integers(0, R, size=R)
        sel = [u for p in pick for u in row_to_uids[rids[p]]]
        k = {m: kap_sel(preds[m], gold, sel) for m in methods}
        ba = {m: bacc_sel(preds[m], gold, sel) for m in methods}
        for m in methods:
            acc[("k", m)].append(k[m]); acc[("ba", m)].append(ba[m])
        for a, b in pairs:
            acc[("dk", a, b)].append(k[a] - k[b])
            acc[("dba", a, b)].append(ba[a] - ba[b])

    out = dict(
        held_out=d, train_datasets=train_names, models=models, n=len(uids), R=R,
        B=args.B, condition=args.condition,
        interface=interface,
        oracle_model=oracle, best_on_avg_model=boa,
        transferred_coefficients={m: float(c) for m, c in zip(models, lr.coef_[0])},
        transferred_intercept=float(lr.intercept_[0]),
        committed=committed,
        kappa={m: _ci(acc[("k", m)]) for m in methods},
        bacc={m: _ci(acc[("ba", m)]) for m in methods},
        diffs={f"{a}-{b}": _ci(acc[("dk", a, b)]) for a, b in pairs},
        bacc_diffs={f"{a}-{b}": _ci(acc[("dba", a, b)]) for a, b in pairs},
    )

    win_lo = out["diffs"]["transferred_stacked-best_single_cv"][1]
    clean_lo = out["diffs"]["transferred_stacked-oracle_best"][1]
    cost = out["diffs"]["indomain_stacked-transferred_stacked"]
    out["verdict"] = dict(
        transfer_win=bool(win_lo > 0),
        clean_transfer=bool(win_lo > 0 and clean_lo > 0),
        cost_significant=bool(cost[1] > 0 or cost[2] < 0),   # True => transfer is NOT free
    )
    return out


# ----------------------------------------------------------------------------- reporting
def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def verdict_label(v):
    if v["clean_transfer"]:
        return "CLEAN TRANSFER"
    if v["transfer_win"]:
        return "TRANSFER WIN (not clean)"
    return "no transfer win (tie/loss)"


def print_dataset(out):
    print(f"\n=== held-out: {out['held_out']}  (trained on {', '.join(out['train_datasets'])}) "
          f"[{out.get('interface', 'binary')} interface] ===")
    print(f"oracle single = {out['oracle_model'].split('/')[-1]} ; "
          f"best-on-avg single = {out['best_on_avg_model'].split('/')[-1]}")
    print("kappa (per-method denom) / bacc (non-answer=error), 95% CI, committed-n:")
    order = sorted(out["kappa"], key=lambda m: -out["kappa"][m][0])
    for m in order:
        c = out["committed"][m]
        flag = "" if c == out["n"] else f"  (abstained {out['n']-c})"
        print(f"  {m:<22} k={out['kappa'][m][0]:.3f} "
              f"[{out['kappa'][m][1]:.3f},{out['kappa'][m][2]:.3f}]  "
              f"ba={out['bacc'][m][0]:.3f} "
              f"[{out['bacc'][m][1]:.3f},{out['bacc'][m][2]:.3f}]  "
              f"n={c}/{out['n']}{flag}")
    print("key paired diffs (kappa | bacc; CI excluding 0 => *):")
    for a, b in [("transferred_stacked", "best_single_cv"),
                 ("transferred_stacked", "oracle_best"),
                 ("indomain_stacked", "transferred_stacked")]:
        dk = out["diffs"][f"{a}-{b}"]; db = out["bacc_diffs"][f"{a}-{b}"]
        sk = "*" if (dk[1] > 0 or dk[2] < 0) else " "
        sb = "*" if (db[1] > 0 or db[2] < 0) else " "
        tag = {"transferred_stacked-best_single_cv": "WIN ",
               "transferred_stacked-oracle_best": "CLEAN",
               "indomain_stacked-transferred_stacked": "COST"}[f"{a}-{b}"]
        print(f"  [{tag}] {a} - {b:<18} k={fmt(dk)}{sk}  ba={fmt(db)}{sb}")
    print(f"  VERDICT: {verdict_label(out['verdict'])}; "
          f"transfer {'NOT free (sig cost)' if out['verdict']['cost_significant'] else 'free (cost n.s.)'}")


def write_summary(results, path):
    L = []
    L.append("# Cross-dataset transfer (LODO) -- verdicts\n")
    L.append("_Kappa uses the per-method denominator (weighted.py convention); "
             "bacc counts non-answer as error on the full denominator._")
    L.append("_A transfer win = (transferred - in-domain CV-best single) lower CI > 0; "
             "clean = also beats the in-domain oracle single._")
    L.append("_Transfer cost = (in-domain stacker - transferred); a non-significant "
             "cost means transfer is free._\n")
    for out in results:
        v = out["verdict"]
        win = out["diffs"]["transferred_stacked-best_single_cv"]
        clean = out["diffs"]["transferred_stacked-oracle_best"]
        cost = out["diffs"]["indomain_stacked-transferred_stacked"]
        iface = out.get("interface", "binary")
        L.append(f"## {out['held_out']}\n")
        L.append(f"Verdict: **{verdict_label(v)}** ({iface} interface).")
        L.append(f"Transferred stacker kappa {fmt(out['kappa']['transferred_stacked'])}, "
                 f"bacc {fmt(out['bacc']['transferred_stacked'])}.")
        L.append(f"WIN test (transferred - CV-best single): {fmt(win)}.")
        L.append(f"CLEAN test (transferred - oracle single): {fmt(clean)}.")
        L.append(f"COST (in-domain stacker - transferred): {fmt(cost)} "
                 f"-- {'significant (transfer is not free)' if v['cost_significant'] else 'not significant (transfer is free)'}.")
        flagged = [m for m in out['committed'] if out['committed'][m] != out['n']]
        if flagged:
            L.append(f"Coverage note: abstentions present in {', '.join(flagged)} "
                     f"(committed-n below {out['n']}); kappa excuses them, bacc charges them.")
        else:
            L.append(f"Ground note: full coverage (all methods committed n={out['n']}); "
                     "kappa and bacc denominators coincide.")
        L.append("")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(L).rstrip() + "\n")
    print(f"\nwrote summary {path}")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="name:path entries, e.g. medhallu:verdicts/medhallu/experiment.jsonl")
    ap.add_argument("--condition", default=None,
                    help="scoring frame; default None = per-dataset auto-detect "
                         "(def_on for QA-form sets, claimcheck for the "
                         "LLM-AggreFact sets), so a mixed-frame pool is scored "
                         "under each dataset's own frame")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--outdir", default="derived/_transfer")
    ap.add_argument("--summary", default=None, help="markdown summary path")
    ap.add_argument("--sonnet-tag", default=None,
                    help="judge tag to load as frontier reference, e.g. judge_sonnet "
                         "(expects verdicts/<dataset>/<tag>.jsonl alongside each experiment)")
    ap.add_argument("--pooled-out", default=None,
                    help="path for the Experiment-A pooled-weights JSON "
                         "(default: <outdir>/pooled_stacker_weights<tag>.json)")
    ap.add_argument("--tag", default="",
                    help="suffix for every output filename, e.g. '_all8'; use it "
                         "whenever a second pathway is run so its artefacts cannot "
                         "overwrite the frozen generative ones")
    args = ap.parse_args()

    # parse name:path
    data, names, paths = {}, [], {}
    for spec in args.datasets:
        name, _, path = spec.partition(":")
        if not path:
            raise SystemExit(f"bad --datasets entry '{spec}', expected name:path")
        names.append(name); paths[name] = path
        data[name] = load(path, args.condition)

    models = assert_pooling_safe(data, names)
    print(f"pooling-safe: {len(names)} datasets, {len(models)} identical models, "
          f"condition={args.condition or 'per-dataset auto (def_on / claimcheck)'}")

    softdata = None

    # optional Sonnet per-dataset preds
    sonnet = None
    if args.sonnet_tag:
        sonnet = {}
        for n in names:
            jp = Path(paths[n]).with_name(f"{args.sonnet_tag}.jsonl")
            # verdict_exists, not jp.exists(): the judge file may be stored
            # compressed, and a raw-path check does not raise when it is -- it
            # silently drops the frontier reference, and the leave-one-out JSONs
            # come out missing their sonnet comparison with nothing to say so.
            if verdict_exists(jp):
                # None, not args.condition: a judge file carries its dataset's own
                # frame, and forcing one frame across a mixed-frame suite loads
                # nothing for the sets that disagree with it.
                sv, _, _ = load(str(jp), None)
                # one model in a judge file; take its per-uid preds
                if sv:
                    sonnet[n] = next(iter(sv.values()))
        print(f"sonnet reference loaded for: {sorted(sonnet) or 'none'}")

    # Experiment A: pooled global stacker (weights for inspection only)
    Xall, yall = pooled_rows(data, names, models, softdata)
    lrA = LogisticRegression(max_iter=1000).fit(Xall, yall)
    pooled = dict(
        note="In-sample pooled fit -- coefficients for inspection ONLY; "
             "in-sample kappa is not honest (use the LODO per-dataset JSONs).",
        datasets=names, n_pooled=int(len(yall)), models=models,
        interface="soft" if softdata else "binary",
        coefficients={m: float(c) for m, c in zip(models, lrA.coef_[0])},
        intercept=float(lrA.intercept_[0]))
    pooled_out = args.pooled_out or str(
        Path(args.outdir) / f"pooled_stacker_weights{args.tag}.json")
    Path(pooled_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(pooled, open(pooled_out, "w"), indent=2)
    print(f"\n[A] pooled global stacker (n={len(yall)}), coefficients (inspection only):")
    for m in sorted(models, key=lambda m: -pooled["coefficients"][m]):
        print(f"    {m.split('/')[-1]:<28} {pooled['coefficients'][m]:+.3f}")
    print(f"    wrote {pooled_out}")

    # Experiment B: LODO per held-out dataset
    results = []
    for d in names:
        out = evaluate_heldout(d, data, names, models, args, sonnet, softdata)
        print_dataset(out)
        op = Path(args.outdir) / f"{d}_transfer_lodo{args.tag}.json"
        op.parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(op, "w"), indent=2)
        print(f"  wrote {op}")
        results.append(out)

    if args.summary:
        write_summary(results, args.summary)


if __name__ == "__main__":
    main()
