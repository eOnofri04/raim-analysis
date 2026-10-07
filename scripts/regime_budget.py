#!/usr/bin/env python
r"""How many gold labels does it take to read the regime?

WHY THIS EXISTS. The competence/correlation account says the panel pays where
member competence is spread and member errors decorrelate, and the paper leans
on that to say *in advance* whether a cheap panel can stand in for a frontier
judge on a given task. Both coordinates it names are computed against gold:

    phi     mean pairwise Pearson on per-item ERROR indicators   (diversity.py)
    spread  number of members reaching kappa >= 0.30 vs gold     (paper_figures)

so the diagnostic cannot be free. The question this answers is therefore not
whether the regime is readable without labels -- it is not, see the identity
below for the one half that is -- but how *few* labels it takes, against the
calibration set the stacker already consumes. If the answer is far below that
set's size, the diagnostic costs nothing a deployer was not already paying.

WHAT IS MEASURED. Per dataset and per budget n, R independent calibration
samples of n items are drawn, and on each the two coordinates and the panel's
apparent leader are re-estimated. We report how far those land from the
full-sample values, and how often the call they induce is the full-sample call.

Sampling is by CLUSTER, not by item, matching the resampling unit of the paper's
bootstrap (a matched pair on the question-answering sets, a shared grounding
document on the claim-verification ones). A deployer buying labels buys records,
and two items of one record are not two independent labels.

THE ONE COORDINATE THAT IS FREE. For binary verdicts,

    v_i = v_j   <=>   e_i = e_j        (since e_k = 1{v_k != y}, y common)

so the mean pairwise AGREEMENT RATE among members is identically the mean
pairwise error-agreement rate: the correlation coordinate has an exact
label-free counterpart, and only its normalisation into a Pearson phi needs the
per-member error rates. `agreement_identity` records both sides of that equality
per dataset, so the claim is checked rather than asserted.

    regime_budget.py --out derived/_summary/regime_budget.json

Deterministic at --seed: every dataset draws from its own seeded generator, so a
single-dataset run reproduces that dataset's block of a whole-suite run.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import zlib
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

# raim_lib sits one level up; put the repository root on the path so this
# script runs the same whether driven by make or invoked directly.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from raim_lib.verdicts import load as load_verdicts
from raim_lib.verdicts import DERIVED, VERDICTS

from paper_tables import DATASETS, pick_condition
from analyze_extras import dawid_skene

# The competence criterion is the paper's own, used unchanged in
# fig_regimeforest and throughout Section 4; it is not tuned here.
KAPPA_COMPETENT = 0.30
# "Competence is spread" = a majority of the panel clears that bar. Declared
# once and applied identically to the sample and the full set, so the budget
# result is about the sampling error in the coordinates and not about a
# threshold fitted to eight datasets. On this suite the competence term is what
# binds; the correlation term is carried for the form of the rule the account
# argues for, both axes being required (truthfulqa is the standing example of a
# decorrelated panel with nothing to aggregate).
SPREAD_FAVOURABLE = 5
PHI_FAVOURABLE = 0.40

# Kish's effective sample size, n_eff = k / (1 + (k-1) * phibar), which turns the
# correlation coordinate into "how many independent judges is this panel worth".
# It is carried because \citet{Kohli_26} summarise a NINE-JUDGE FRONTIER
# panel with the same statistic on the same definition of phi (Pearson on binary
# per-item error vectors over all member pairs), which makes the two studies
# comparable on a derived, panel-internal quantity even though no row is shared.
# Their reported values are recorded here so the comparison is generated rather
# than retyped into the manuscript.
KOHLI = {"ChaosNLI-MNLI": (0.391, 2.18, "+0.2"),
         "ChaosNLI-SNLI": (0.354, 2.35, "-6.5"),
         "ChaosNLI-AlphaNLI": (0.328, 2.48, "-2.5")}
KOHLI_ASYMPTOTE = 2.6   # their 1/phibar bound, stated as a limit on "current models"


def kish_neff(phi, k):
    """Effective number of independent judges in a panel of k at mean pairwise phi."""
    return k / (1.0 + (k - 1) * phi)

# Budgets are in labelled RECORDS (clusters), matching stacker_curve.py, so that
# this experiment and the aggregator's own learning curve are quoted in one unit
# and the claim that the two are "one purchase" needs no conversion.
BUDGETS = (10, 25, 50, 100, 200)
ROSTER = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
          "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-9b-it",
          "microsoft/Phi-4-mini-instruct", "01-ai/Yi-1.5-9B-Chat-16K",
          "CohereLabs/c4ai-command-r7b-12-2024", "zai-org/glm-4-9b-chat-hf",
          "ibm-granite/granite-3.3-8b-instruct", "tiiuae/Falcon3-7B-Instruct"]


def unsupervised_competence(ds, V, models):
    """The label-free competence estimate, and how far it is from the truth.

    Dawid--Skene scores each member against the latent consensus it infers from
    the vote matrix alone. Where the panel shares an error that consensus IS the
    error, so this is expected to fail precisely on the correlated panels -- the
    claim of the appendix is that it does, and this computes the evidence rather
    than asserting it.
    """
    votes = {m: {i: (None if np.isnan(V[k, i]) else int(V[k, i]))
                 for i in range(V.shape[1])} for k, m in enumerate(models)}
    uids = list(range(V.shape[1]))
    lab = dawid_skene(votes, uids, models)
    P = np.array([lab[u] for u in uids], dtype=float)
    kap, _ = member_kappas(V, P)
    return dict(spread=int((kap >= KAPPA_COMPETENT).sum()),
                leader=int(np.argmax(kap)), kappas=kap.tolist())


def load_panel(ds):
    """The ten members' votes, the gold label and the clustering unit, under the
    dataset's own scoring frame (pick_condition, as everywhere else)."""
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
    models = [m for m in ROSTER if m in votes]
    V = np.array([[votes[m].get(u) if votes[m].get(u) is not None else np.nan
                   for u in uids] for m in models], dtype=float)
    y = np.array([gold[u] for u in uids], dtype=float)
    clusters = np.array([rowid[u] for u in uids])
    return V, y, clusters, models, cond


def member_kappas(V, y):
    """Each member's Cohen kappa over the items it committed on.

    A member that returns a single class on the sample, or whose committed set
    carries a single gold class, has an undefined kappa; it is scored 0.0, which
    counts it as not competent. On a 25-item draw that is a real event and it is
    part of what the budget question is asking about -- `degenerate` upstream
    records how often it happens rather than hiding it.
    """
    out, degenerate = np.zeros(V.shape[0]), 0
    for i in range(V.shape[0]):
        msk = ~np.isnan(V[i])
        yt, yp = y[msk], V[i][msk]
        if len(yt) < 2 or len(set(yt)) < 2 or len(set(yp)) < 2:
            out[i], degenerate = 0.0, degenerate + 1
            continue
        out[i] = cohen_kappa_score(yt, yp)
    return out, degenerate


def mean_pair(M, fn):
    """Mean of fn over the 45 member pairs, skipping pairs with too little overlap."""
    vals = [fn(M[i], M[j]) for i, j in itertools.combinations(range(M.shape[0]), 2)]
    vals = [v for v in vals if v == v]
    return float(np.mean(vals)) if vals else float("nan")


def _corr(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 5:
        return float("nan")
    aa, bb = a[m], b[m]
    if aa.std() == 0 or bb.std() == 0:
        return 0.0
    return float(np.corrcoef(aa, bb)[0, 1])


def _agree(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 5:
        return float("nan")
    return float((a[m] == b[m]).mean())


def coordinates(V, y):
    """The two regime coordinates on one item set, plus the leader it names.

    `excess` is the continuous companion to `spread`: the total competence the
    panel carries above the bar, sum_i max(0, kappa_i - 0.30). It is reported
    because the count is a threshold statistic and therefore fragile exactly
    where a panel sits at the bar -- on this suite, factscore (at 5 of 10
    members, the boundary itself) and truthfulqa (whose members cluster near
    kappa = 0.30). Normalised by its own between-dataset spread the continuous
    form resolves about twice as finely at a 100-label budget, which is what
    licenses saying the fragility of the call there is the threshold's and not
    the sample's. The paper's regime plane keeps the count; nothing here
    redefines it.
    """
    E = np.where(np.isnan(V), np.nan, (V != y).astype(float))
    phi = mean_pair(E, _corr)
    kap, degenerate = member_kappas(V, y)
    spread = int((kap >= KAPPA_COMPETENT).sum())
    excess = float(np.maximum(0.0, kap - KAPPA_COMPETENT).sum())
    return dict(phi=phi, spread=spread, excess=excess, leader=int(np.argmax(kap)),
                kappas=kap.tolist(), degenerate=degenerate)


def favourable(phi, spread):
    """The regime call: competence spread across a majority of the panel, and
    errors not strongly shared. Both axes, as the account requires."""
    return bool(spread >= SPREAD_FAVOURABLE and phi <= PHI_FAVOURABLE)


def sample_clusters(clusters, n_records, rng):
    """Draw exactly `n_records` whole clusters without replacement.

    The unit is the record a deployer actually labels -- a matched pair on the
    question-answering sets, a grounding document on the claim-verification ones
    -- so a budget means the same thing here as in stacker_curve.py and the two
    experiments can be compared without a conversion.
    """
    uniq = np.unique(clusters)
    pick = rng.choice(uniq, size=min(n_records, len(uniq)), replace=False)
    return np.flatnonzero(np.isin(clusters, pick))


def run_dataset(ds, budgets, R, seed):
    V, y, clusters, models, cond = load_panel(ds)
    n = len(y)
    full = coordinates(V, y)
    full_fav = favourable(full["phi"], full["spread"])

    # The label-free half, checked rather than asserted: mean pairwise vote
    # agreement against mean pairwise error agreement, which are equal termwise.
    E = np.where(np.isnan(V), np.nan, (V != y).astype(float))
    agree_votes = mean_pair(V, _agree)
    agree_errors = mean_pair(E, _agree)

    out_budgets = []
    for b in budgets:
        if b > len(np.unique(clusters)):
            continue
        # Seeded per (dataset, budget) off a STABLE hash. Python's built-in
        # hash() is salted per process unless PYTHONHASHSEED is pinned, so using
        # it here would make the artefact irreproducible between runs on one
        # machine -- the failure mode `make derivecheck` is meant to catch.
        rng = np.random.default_rng(
            [seed, zlib.crc32(ds.encode()), b])
        phis, spreads, excesses, leaders, calls, sizes, degens = (
            [], [], [], [], [], [], [])
        for _ in range(R):
            idx = sample_clusters(clusters, b, rng)
            c = coordinates(V[:, idx], y[idx])
            phis.append(c["phi"])
            spreads.append(c["spread"])
            excesses.append(c["excess"])
            leaders.append(c["leader"])
            calls.append(favourable(c["phi"], c["spread"]))
            sizes.append(len(idx))
            degens.append(c["degenerate"])
        phis = np.array(phis, dtype=float)
        spreads = np.array(spreads, dtype=float)
        excesses = np.array(excesses, dtype=float)
        out_budgets.append(dict(
            budget=b, realised_n_median=float(np.median(sizes)),
            phi_median=float(np.nanmedian(phis)),
            phi_iqr=[float(np.nanpercentile(phis, 25)), float(np.nanpercentile(phis, 75))],
            phi_abs_err_median=float(np.nanmedian(np.abs(phis - full["phi"]))),
            spread_median=float(np.median(spreads)),
            spread_abs_err_median=float(np.median(np.abs(spreads - full["spread"]))),
            p_spread_within1=float(np.mean(np.abs(spreads - full["spread"]) <= 1)),
            excess_median=float(np.median(excesses)),
            excess_abs_err_median=float(np.median(np.abs(excesses - full["excess"]))),
            p_leader=float(np.mean([l == full["leader"] for l in leaders])),
            p_call_correct=float(np.mean([c == full_fav for c in calls])),
            degenerate_mean=float(np.mean(degens)),
        ))
    unsup = unsupervised_competence(ds, V, models)
    return dict(
        dataset=ds, condition=cond, n=n, n_clusters=int(len(np.unique(clusters))),
        unsupervised=dict(spread=unsup["spread"], leader=models[unsup["leader"]],
                          leader_correct=unsup["leader"] == full["leader"]),
        members=models, k=len(models), n_eff=kish_neff(full["phi"], len(models)),
        full=dict(phi=full["phi"], spread=full["spread"],
                                  excess=full["excess"],
                                  leader=models[full["leader"]],
                                  favourable=full_fav),
        agreement_identity=dict(mean_pairwise_vote_agreement=agree_votes,
                                mean_pairwise_error_agreement=agree_errors,
                                abs_difference=abs(agree_votes - agree_errors)),
        budgets=out_budgets,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS))
    ap.add_argument("--budgets", nargs="*", type=int, default=list(BUDGETS))
    ap.add_argument("--R", type=int, default=200,
                    help="calibration samples drawn per (dataset, budget)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="default: <derived>/_summary/regime_budget.json")
    ap.add_argument("--tex", action="store_true",
                    help="also write tab_regimebudget.tex into --texdir")
    ap.add_argument("--texdir", default=str(Path(__file__).resolve().parent / "tables"))
    args = ap.parse_args()

    per = [run_dataset(ds, args.budgets, args.R, args.seed) for ds in args.datasets]

    # The suite-level reading: at each budget, on how many datasets does the
    # median draw name the full-sample regime, and how often per draw.
    by_budget = {}
    for b in args.budgets:
        rows = [r for d in per for r in d["budgets"] if r["budget"] == b]
        if not rows:
            continue
        by_budget[str(b)] = dict(
            n_datasets=len(rows),
            mean_p_call_correct=float(np.mean([r["p_call_correct"] for r in rows])),
            min_p_call_correct=float(np.min([r["p_call_correct"] for r in rows])),
            mean_p_leader=float(np.mean([r["p_leader"] for r in rows])),
            median_phi_abs_err=float(np.median([r["phi_abs_err_median"] for r in rows])),
            median_spread_abs_err=float(np.median([r["spread_abs_err_median"] for r in rows])),
            median_excess_abs_err=float(np.median([r["excess_abs_err_median"] for r in rows])),
        )

    # ---- count versus continuous competence, on one scale ---------------------
    # app:budget says the continuous `excess` resolves the regime about twice as
    # finely as the count `spread` at a small budget. The two are on different
    # scales, so each median absolute sampling error is divided by that statistic's
    # own range across the eight full-sample datasets, and stored so that the
    # appendix quotes a generated number.
    rng_spread = float(np.ptp([d["full"]["spread"] for d in per]))
    rng_excess = float(np.ptp([d["full"]["excess"] for d in per]))
    for b, v in by_budget.items():
        v["spread_err_over_range"] = v["median_spread_abs_err"] / rng_spread
        v["excess_err_over_range"] = v["median_excess_abs_err"] / rng_excess

    # ---- how far the thresholds can move before a call changes ----------------
    # The two cut-offs were fixed with the eight results in view, so app:budget reports
    # the region over which every full-sample call is unchanged: a stability
    # statement read off these eight points, not a validation of the thresholds.
    def calls_at(s_thr, phi_thr):
        return tuple(d["full"]["spread"] >= s_thr and d["full"]["phi"] <= phi_thr
                     for d in per)
    ref = calls_at(SPREAD_FAVOURABLE, PHI_FAVOURABLE)
    n_members = max(d["full"]["spread"] for d in per)
    spread_ok = [s for s in range(0, n_members + 2) if calls_at(s, PHI_FAVOURABLE) == ref]
    phis = sorted(d["full"]["phi"] for d in per)
    phi_lo = max(p for p in phis if p <= PHI_FAVOURABLE)
    phi_hi = min(p for p in phis if p > PHI_FAVOURABLE)
    joint = all(calls_at(s, t) == ref for s in spread_ok
                for t in (phi_lo, (phi_lo + phi_hi) / 2, np.nextafter(phi_hi, 0)))
    # The EXACT set of (count, phi bar) pairs keeping every call, not just the
    # bracket above: per integer count k, the phi bar must reach every admitted
    # dataset's phi and stay below the phi of every rejected dataset that has at
    # least k competent members. Where no rejected dataset has k, the bar is
    # unbounded above (phi_hi None). In the regime plane, with the count read as
    # a continuous corner y in (k - 1, k], this is the region the figure hatches.
    adm = [d for d, c in zip(per, ref) if c]
    rej = [d for d, c in zip(per, ref) if not c]
    call_region = []
    for k in range(0, n_members + 2):
        if any(d["full"]["spread"] < k for d in adm):
            continue
        lo = max(d["full"]["phi"] for d in adm)
        blockers = [d["full"]["phi"] for d in rej if d["full"]["spread"] >= k]
        hi = min(blockers) if blockers else None
        if hi is None or lo < hi:
            call_region.append(dict(count=k, phi_lo=lo, phi_hi=hi))
    threshold_stability = dict(
        call_region=call_region,
        spread_thresholds_same_calls=spread_ok,
        phi_threshold_same_calls=[phi_lo, phi_hi],   # half-open [lo, hi): the bracket, sufficient only
        joint_region_holds=bool(joint),
        admitted=[d["dataset"] for d, c in zip(per, ref) if c],
    )

    # ---- the two suite-level readings the appendix quotes --------------------
    # Both are rank correlations over the eight datasets, so both are weak by
    # construction (at n = 8 the two-sided 5% critical value is |rho| ~ 0.71).
    # The first clears it comfortably and is the claim; the second is reported
    # because it is a NULL -- an unsupervised competence estimate carries no
    # information about the labelled one -- and a null needs no power to be
    # damning when the alternative it replaces is a claim of equivalence.
    from scipy.stats import spearmanr

    agree = [d["agreement_identity"]["mean_pairwise_vote_agreement"] for d in per]
    phi_err = [d["full"]["phi"] for d in per]
    spread_gold = [d["full"]["spread"] for d in per]
    spread_unsup = [d["unsupervised"]["spread"] for d in per]
    r_agree = spearmanr(agree, phi_err)
    r_unsup = spearmanr(spread_unsup, spread_gold)

    out = dict(
        seed=args.seed, R=args.R, budgets=args.budgets,
        kappa_competent=KAPPA_COMPETENT,
        spread_favourable=SPREAD_FAVOURABLE, phi_favourable=PHI_FAVOURABLE,
        per_dataset=per, by_budget=by_budget,
        threshold_stability=threshold_stability,
        agreement_identity_max_abs_difference=max(
            d["agreement_identity"]["abs_difference"] for d in per),
        kohli_comparison=dict(
            reference=KOHLI, asymptote=KOHLI_ASYMPTOTE,
            ours={d["dataset"]: d["n_eff"] for d in per},
            n_above_their_best=sum(d["n_eff"] > max(v[1] for v in KOHLI.values())
                                   for d in per),
            n_above_asymptote=sum(d["n_eff"] > KOHLI_ASYMPTOTE for d in per),
            max_ours=max(d["n_eff"] for d in per),
        ),
        labelfree=dict(
            rho_agreement_vs_phi=float(r_agree.statistic),
            p_agreement_vs_phi=float(r_agree.pvalue),
            rho_unsup_spread_vs_gold=float(r_unsup.statistic),
            p_unsup_spread_vs_gold=float(r_unsup.pvalue),
            n_leader_recovered_unsupervised=sum(
                d["unsupervised"]["leader_correct"] for d in per),
            n_datasets=len(per),
        ),
    )

    path = Path(args.out) if args.out else DERIVED / "_summary" / "regime_budget.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")

    lf = out["labelfree"]
    print(f"\nagreement identity: max |vote-agreement - error-agreement| = "
          f"{out['agreement_identity_max_abs_difference']:.2e}  (exact = 0)")
    print(f"label-free correlation coordinate: rho(agreement, phi_err) = "
          f"{lf['rho_agreement_vs_phi']:+.3f}  (p = {lf['p_agreement_vs_phi']:.3f})")
    print(f"label-free competence coordinate:  rho(DS spread, gold spread) = "
          f"{lf['rho_unsup_spread_vs_gold']:+.3f}  (p = "
          f"{lf['p_unsup_spread_vs_gold']:.3f}); DS names the true leader on "
          f"{lf['n_leader_recovered_unsupervised']}/{lf['n_datasets']} datasets")
    print(f"\n{'dataset':18} {'n':>5} {'phi':>6} {'spr':>4} {'regime':>12}   "
          + "  ".join(f"p(call)@{b}" for b in args.budgets))
    for d in per:
        cells = {r["budget"]: r["p_call_correct"] for r in d["budgets"]}
        line = "  ".join(f"{cells[b]:10.2f}" if b in cells else f"{'--':>10}"
                         for b in args.budgets)
        print(f"{d['dataset']:18} {d['n']:5d} {d['full']['phi']:6.3f} "
              f"{d['full']['spread']:4d} "
              f"{('favourable' if d['full']['favourable'] else 'unfavourable'):>12}   {line}")
    kc = out["kohli_comparison"]
    print(f"\neffective independent judges (Kish n_eff, k={per[0]['k']}):")
    for d in sorted(per, key=lambda x: -x["n_eff"]):
        print(f"  {d['dataset']:20} phibar={d['full']['phi']:.3f}  n_eff={d['n_eff']:.2f}")
    print(f"  -- frontier panel of nine (Kohli_26): n_eff "
          f"{min(v[1] for v in KOHLI.values()):.2f}--{max(v[1] for v in KOHLI.values()):.2f}, "
          f"asymptote ~{KOHLI_ASYMPTOTE}")
    print(f"  -- ours exceed their best on {kc['n_above_their_best']}/{len(per)} datasets, "
          f"their asymptote on {kc['n_above_asymptote']}; max {kc['max_ours']:.2f}")

    print(f"\n{'budget':>7} {'datasets':>9} {'mean p(call)':>13} {'min p(call)':>12} "
          f"{'p(leader)':>10} {'|d phi|':>8} {'|d spread|':>11}")
    for b, r in by_budget.items():
        print(f"{b:>7} {r['n_datasets']:9d} {r['mean_p_call_correct']:13.3f} "
              f"{r['min_p_call_correct']:12.3f} {r['mean_p_leader']:10.3f} "
              f"{r['median_phi_abs_err']:8.3f} {r['median_spread_abs_err']:11.1f}")

    if args.tex:
        write_tex(out, Path(args.texdir))


def write_tex(out, texdir):
    """tab_regimebudget: the per-dataset call recovery, and the suite reading."""
    from paper_tables import DATASETS as DS_ORDER, TEXNAME

    budgets = out["budgets"]
    texdir.mkdir(parents=True, exist_ok=True)
    L = [f"% AUTO-GENERATED by {Path(__file__).name} in scripts/ "
         f"-- do not edit by hand; regenerate with `make tables`.",
         r"\begin{table}[tp]", r"\centering\footnotesize",
         r"\caption{Recovery of the regime call from a limited labelling budget. "
         r"Per dataset and budget, $R=" + str(out["R"]) + r"$ calibration "
         r"samples of that many labelled records are drawn as whole clusters (\S\ref{ssec:metrics}), "
         r"both coordinates of~\S\ref{ssec:mechanism} being re-estimated on each.}",
         r"\label{tab:regimebudget}",
         r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{l r rr c " + "r" * len(budgets) + "}",
         r"\toprule",
         r" & & \multicolumn{3}{c}{\emph{full sample}} & "
         r"\multicolumn{" + str(len(budgets)) + r"}{c}{$\Pr$(call matches) "
         r"\emph{at a budget of}} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-" + str(5 + len(budgets)) + "}",
         r"Dataset & $n$ & $\bar\phi$ & spread & regime & "
         + " & ".join(str(b) for b in budgets) + r" \\",
         r"\midrule"]
    per = {d["dataset"]: d for d in out["per_dataset"]}
    for ds in DS_ORDER:
        d = per[ds]
        cells = {r["budget"]: r["p_call_correct"] for r in d["budgets"]}
        row = " & ".join(f"${cells[b]:.2f}$" if b in cells else "---" for b in budgets)
        L.append(f"{TEXNAME.get(ds, ds)} & {d['n']} & ${d['full']['phi']:.3f}$ & "
                 f"${d['full']['spread']}$ & "
                 + ("admissible" if d["full"]["favourable"] else "not admissible")
                 + f" & {row} " + r"\\")
    L.append(r"\midrule")
    bb = out["by_budget"]
    L.append(r"\emph{suite}: mean $\Pr$(call) & & & & & "
             + " & ".join(f"$\\mathbf{{{bb[str(b)]['mean_p_call_correct']:.2f}}}$"
                          if str(b) in bb else "---" for b in budgets) + r" \\")
    L.append(r"\emph{suite}: $\Pr$(same leader) & & & & & "
             + " & ".join(f"${bb[str(b)]['mean_p_leader']:.2f}$"
                          if str(b) in bb else "---" for b in budgets) + r" \\")
    L.append(r"\emph{suite}: median $|\Delta\bar\phi|$ & & & & & "
             + " & ".join(f"${bb[str(b)]['median_phi_abs_err']:.3f}$"
                          if str(b) in bb else "---" for b in budgets) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\tablelegend{\emph{regime}: the full-sample call, a task being admissible at a "
          r"competence spread of at least "
          + {5: "five"}.get(out["spread_favourable"], str(out["spread_favourable"]))
          + r" of the ten members at $\kp\geq"
          + f"{out['kappa_competent']:.2f}" + r"$ and $\bar\phi\leq"
          + f"{out['phi_favourable']:.2f}" + r"$. "
          r"Budget columns: fraction of samples reproducing that call; ---: budget exceeding the dataset.}",
          r"\end{table}"]
    (texdir / "tab_regimebudget.tex").write_text("\n".join(L) + "\n")
    print(f"wrote {texdir / 'tab_regimebudget.tex'}")


if __name__ == "__main__":
    main()
