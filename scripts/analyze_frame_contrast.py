#!/usr/bin/env python
r"""The frame contrast on the four LLM-AggreFact sets: def_on against claim-support.

What this substantiates. The paper reports two things about the frame, and both are
claims about a measurement rather than about a design choice
(Appendix~\ref{app:extbase}):

  * the question-answer rubric presented each claim twice, and under it "member
    competence collapsed to the floor on all four sets";
  * scored that way, the unweighted panel "sat some twenty balanced-accuracy points
    below the best published result", and the claim-support frame lifts it by
    10.7 of those points on average.

The def_on arm sits beside the canonical one as `experiment_defon.jsonl`, which
keeps it out of the scoring path by name whilst leaving it inside the tree the
lockfile pins and the mirror carries, so this is an ordinary zone-2 read.

Method, matched to the frozen methodology so the two arms are comparable. Balanced
accuracy with a non-answer counted as an error, and Cohen's kappa, per member and
for the unweighted panel; paired cluster bootstrap over row_id at B resamples, with
the SAME resample applied to both arms so the difference is paired rather than two
independent intervals differenced. The instance sets are identical by construction
(the same builder at seed 0), and that is asserted rather than assumed.

Reads   <ds>/experiment_defon.jsonl  (the probe arm)
        <ds>/experiment.jsonl        (the claim-support arm)
Writes  <derived>/_summary/frame_contrast.json  under --derive, tab_frame.tex under --tex

The fit and the table are separate invocations, on the house rule that --derive
computes zone 3 and everything else renders from it:

Usage:
  python analyze_frame_contrast.py --derive        # the contrast -> frame_contrast.json
  python analyze_frame_contrast.py                 # the console report, from that JSON
  python analyze_frame_contrast.py --tex --out tables/
"""
from __future__ import annotations

import argparse
import json
import os as _os
import sys as _sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).resolve().parent
_sys.path.insert(0, str(HERE.parent))
from raim_lib.verdicts import load as load_verdicts, exists as verdict_exists  # noqa: E402

RUNS = Path(_os.environ.get("RAIM_DERIVED", HERE.parent / "derived"))
VERDICTS = Path(_os.environ.get("RAIM_VERDICTS", HERE.parent / "verdicts"))
DATASETS = ["aggrefact_xsum", "aggrefact_wice", "aggrefact_cnn", "aggrefact_expertqa"]
DSNAME = {"aggrefact_xsum": r"\xsum", "aggrefact_wice": r"\wice",
          "aggrefact_cnn": r"\cnn", "aggrefact_expertqa": r"\expertqa"}


def arm(path, condition):
    """votes[model][uid] -> 0/1/None, plus gold and row_id, for one condition."""
    votes, gold, rowid = defaultdict(dict), {}, {}
    for r in load_verdicts(path):
        if r["condition"] != condition:
            continue
        votes[r["model"]][r["uid"]] = r["pred"]
        gold[r["uid"]] = r["gold"]
        rowid[r["uid"]] = r["row_id"]
    return dict(votes), gold, rowid


def _bacc(y, p):
    """Balanced accuracy with a non-answer counted as an error, not dropped.

    Dropping abstentions would flatter whichever arm abstains more, which is
    precisely the axis under test: the misframed arm does not merely score worse,
    it fails to parse more often.
    """
    y = np.asarray(y); p = np.asarray(p)
    out = []
    for cls in (0, 1):
        m = y == cls
        if not m.any():
            return float("nan")
        out.append(float((p[m] == cls).mean()))
    return sum(out) / 2


def _score(vote_map, gold, uids):
    """Balanced accuracy, kappa, coverage and TPR on the hallucinated class.

    TPR is carried because the frame's effect is an operating-point shift, not a
    competence one: the question-answer frame is lenient, and the appendix reports
    the frontier judge catching 39% of hallucinated XSum items under it against 95%
    under claim-support. That is the number the argument turns on, and it is
    invisible in balanced accuracy alone.
    """
    y = [gold[u] for u in uids]
    p = [vote_map.get(u) for u in uids]
    p = [(-1 if v is None else v) for v in p]        # non-answer -> never correct
    ba = _bacc(y, p)
    mask = [v >= 0 for v in p]
    yk = [a for a, m in zip(y, mask) if m]
    pk = [b for b, m in zip(p, mask) if m]
    kp = cohen_kappa_score(yk, pk) if len(set(yk)) > 1 else float("nan")
    cov = float(np.mean(mask))
    npos = sum(1 for a in y if a == 1)
    tpr = (sum(1 for a, b in zip(y, p) if a == 1 and b == 1) / npos
           if npos else float("nan"))
    return ba, kp, cov, tpr


def _panel(votes, uids):
    """Unweighted majority over the members present, None when nobody answered."""
    out = {}
    for u in uids:
        vs = [votes[m].get(u) for m in votes]
        vs = [v for v in vs if v is not None]
        out[u] = None if not vs else int(np.mean(vs) >= 0.5)
    return out


def contrast(ds, defon_path, cc_path, B, seed):
    vd, gd, rd = arm(defon_path, "def_on")
    vc, gc, rc = arm(cc_path, "claimcheck")

    uids = sorted(set(gd) & set(gc))
    if not uids:
        raise SystemExit(f"{ds}: the two arms share no instances")
    if set(gd) != set(gc):
        raise SystemExit(
            f"{ds}: the arms disagree on the instance set "
            f"(+{len(set(gd) - set(gc))}/-{len(set(gc) - set(gd))}); they must be "
            f"the same build for the contrast to mean anything")
    if any(gd[u] != gc[u] for u in uids):
        raise SystemExit(f"{ds}: the arms disagree on gold labels")
    models = sorted(set(vd) & set(vc))

    pd_, pc_ = _panel(vd, uids), _panel(vc, uids)

    row_to_uids = defaultdict(list)
    for u in uids:
        row_to_uids[rd[u]].append(u)
    rows = sorted(row_to_uids)
    arrays = [row_to_uids[r] for r in rows]

    rng = np.random.default_rng(seed)
    acc = defaultdict(list)
    for _ in range(B):
        pick = rng.integers(0, len(rows), size=len(rows))
        sel = [u for i in pick for u in arrays[i]]
        for m in models:                       # same resample for both arms
            bd, kd, _, _ = _score(vd[m], gd, sel)
            bc, kc, _, _ = _score(vc[m], gc, sel)
            acc[("bacc", m)].append(bc - bd)
            acc[("kappa", m)].append(kc - kd)
        bd, kd, _, _ = _score(pd_, gd, sel)
        bc, kc, _, _ = _score(pc_, gc, sel)
        acc[("bacc", "PANEL")].append(bc - bd)
        acc[("kappa", "PANEL")].append(kc - kd)

    def ci(v):
        a = np.asarray([x for x in v if x == x])
        if a.size == 0:
            return [float("nan")] * 3
        return [float(a.mean()), float(np.percentile(a, 2.5)),
                float(np.percentile(a, 97.5))]

    members = {}
    for m in models + ["PANEL"]:
        vmd = pd_ if m == "PANEL" else vd[m]
        vmc = pc_ if m == "PANEL" else vc[m]
        bd, kd, cd, td = _score(vmd, gd, uids)
        bc, kc, cc, tc = _score(vmc, gc, uids)
        members[m] = dict(
            defon=dict(bacc=bd, kappa=kd, coverage=cd, tpr=td),
            claimcheck=dict(bacc=bc, kappa=kc, coverage=cc, tpr=tc),
            delta_bacc=ci(acc[("bacc", m)]),
            delta_kappa=ci(acc[("kappa", m)]),
        )
    return dict(dataset=ds, n=len(uids), clusters=len(rows), B=B, seed=seed,
                models=models, members=members)


def judge_contrast(ds, tag, defon_path, cc_path):
    """The same contrast for a single judge, which is what the appendix quotes.

    Appendix~\\ref{app:extbase} reports the frontier judge on XSum moving from
    kappa 0.269 to 0.418 and catching 39% of hallucinated items against 95%. Those
    are the sentence's actual numbers, so they are generated here rather than
    recomputed by hand whenever someone asks where they came from. No interval:
    the appendix quotes point estimates and the panel contrast above carries the
    inferential weight.
    """
    def one(path, condition):
        gold, pred = {}, {}
        for r in load_verdicts(path):
            if condition and r.get("condition") not in (None, condition):
                continue
            gold[r["uid"]] = r["gold"]
            pred[r["uid"]] = r["pred"]
        return gold, pred

    gd, pdn = one(defon_path, "def_on")
    gc, pcn = one(cc_path, "claimcheck")
    uids = sorted(set(gd) & set(gc))
    if not uids:
        return None
    bd, kd, cd, td = _score(pdn, gd, uids)
    bc, kc, cc, tc = _score(pcn, gc, uids)

    # kappa and bacc follow the frozen [point, lo, hi] convention, whose point is
    # the cluster-bootstrap RESAMPLE MEAN (raim-verdicts raim/ci.py), so they are
    # read from the judge's own zone-2 record -- the figure tab_frontier, tab_frame
    # and the prose print. The full-sample rescoring above differs from it in the
    # third decimal (XSum def_on 0.2711 against 0.2695) and is kept under an explicit
    # name. Coverage and TPR have no bootstrap counterpart and stay full-sample.
    def record(path):
        rec = Path(str(path).removesuffix(".gz").removesuffix(".jsonl") + ".json")
        rec = VERDICTS / ds / rec.name
        return json.loads(rec.read_text()) if rec.exists() else None

    def arm_out(rec, b, k, c, t):
        out = dict(bacc_full_sample=b, kappa_full_sample=k, coverage=c, tpr=t)
        if rec:
            out["kappa"] = rec["kappa"][0]
            out["bacc"] = rec["balanced_accuracy"][0]
        else:                                   # no record: say so, do not guess
            out["kappa"] = out["bacc"] = None
        return out

    return dict(tag=tag, n=len(uids),
                defon=arm_out(record(defon_path), bd, kd, cd, td),
                claimcheck=arm_out(record(cc_path), bc, kc, cc, tc))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="experiment_defon.jsonl",
                    help="filename of the probe arm beside each canonical one")
    ap.add_argument("--judges", nargs="*", default=["sonnet"],
                    help="judge tags to contrast where judge_<tag>_defon.jsonl "
                         "sits beside judge_<tag>.jsonl")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tex", action="store_true", help="also write tab_frame.tex")
    ap.add_argument("--out", default=str(HERE / "tables"))
    # The house rule, shared with analyze_extras.py: --derive computes and writes the
    # zone-3 JSON, and everything else renders from it, so the paper build neither
    # recomputes the paired bootstrap nor rewrites the committed frame_contrast.json.
    ap.add_argument("--derive", action="store_true",
                    help="recompute the contrast and rewrite "
                         "derived/_summary/frame_contrast.json (~80 s); without it, "
                         "the report and tab_frame are rendered from that JSON")
    args = ap.parse_args()

    if not args.derive:
        results = read_contrast_json()
        report(results)
        if args.tex:
            write_tex(results, Path(args.out))
        return

    results = {}
    for ds in args.datasets:
        dp = RUNS / ds / args.arm          # zone-mapped and de-gzipped by verdicts
        cp = RUNS / ds / "experiment.jsonl"
        if not verdict_exists(dp):
            print(f"  {ds}: no probe arm ({args.arm}); skipped"); continue
        if not verdict_exists(cp):
            print(f"  {ds}: no claim-support arm; skipped"); continue
        results[ds] = contrast(ds, dp, cp, args.B, args.seed)

        # the frontier judge, where both arms of it exist
        judges = {}
        for tag in args.judges:
            jd = RUNS / ds / f"judge_{tag}_defon.jsonl"
            jc = RUNS / ds / f"judge_{tag}.jsonl"
            if verdict_exists(jd) and verdict_exists(jc):
                got = judge_contrast(ds, tag, jd, jc)
                if got:
                    judges[tag] = got
        if judges:
            results[ds]["judges"] = judges
        print_ds(ds, results[ds])          # progress, one line per dataset

    if not results:
        _sys.exit(
            "!! nothing to compare: no probe arm found beside any canonical one.\n"
            "   Produce it in raim-verdicts with `bash scripts/run_contrast.sh`,\n"
            "   then `make lock && make export` so it reaches the mirror.")

    dest = RUNS / "_summary"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "frame_contrast.json").write_text(json.dumps(results, indent=1) + "\n")
    print(f"\nwrote {dest / 'frame_contrast.json'}")
    mean_gap(results)


def print_ds(ds, r):
    m = r["members"]["PANEL"]
    print(f"  {ds:20} panel bacc {m['defon']['bacc']:.3f} -> "
          f"{m['claimcheck']['bacc']:.3f}  "
          f"delta {m['delta_bacc'][0]:+.3f} "
          f"[{m['delta_bacc'][1]:+.3f}, {m['delta_bacc'][2]:+.3f}]")
    for tag, got in (r.get("judges") or {}).items():
        print(f"  {'':20}   judge {tag}: kappa "
              f"{got['defon']['kappa']:.3f} -> {got['claimcheck']['kappa']:.3f}, "
              f"TPR {got['defon']['tpr']:.2f} -> {got['claimcheck']['tpr']:.2f}")


def mean_gap(results):
    """The headline the prose quotes: the mean balanced-accuracy gap across the four
    sets, member-wise. Printed rather than asserted, so the sentence can be checked
    against a number instead of against a memory."""
    gaps = [m["delta_bacc"][0]
            for r in results.values()
            for k, m in r["members"].items() if k != "PANEL"]
    print(f"  mean member bacc gain, claim-support over def_on: "
          f"{np.mean(gaps):+.3f} over {len(gaps)} member-dataset cells")


def report(results):
    for ds, r in results.items():
        print_ds(ds, r)
    mean_gap(results)


def read_contrast_json():
    """The contrast as derive.sh computed it. Loud when it is absent: rendering
    tab_frame from a missing or partial artefact is the failure this arrangement
    exists to prevent, and it would otherwise surface as a KeyError in a formatter."""
    p = RUNS / "_summary" / "frame_contrast.json"
    if not p.exists():
        raise SystemExit(
            f"missing {p}\n"
            f"  it is derived: run `analyze_frame_contrast.py --derive`, or `make derive`")
    results = json.loads(p.read_text())
    if not results:
        raise SystemExit(f"{p} is empty -- re-derive it with --derive")
    return results


def frontier_kappa(ds, tag):
    """The frontier judge's point kappa under one frame, from its judge record.

    Read from the zone-2 judge_<tag>.json rather than from the contrast above, so
    that the claim-support column is the very figure tab_frontier and tab_master
    print; the contrast rescores the raw verdicts and lands a few thousandths away.
    The column answers the prompt-parity question for the frontier: whether the
    frame each dataset is scored under is the better of the two for the judge the
    panel is meant to replace.
    """
    return json.loads((VERDICTS / ds / f"judge_{tag}.json").read_text())["kappa"][0]


def write_tex(results, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    lines = [f"% AUTO-GENERATED by {Path(_sys.argv[0]).name} in scripts/ "
             f"-- do not edit by hand.",
             r"\begin{table}[t]\centering\footnotesize",
             r"\caption{Frame contrast on the four \aggrefact\ sets: the "
             r"question--answer rubric (\defon) against the claim-support frame. "
             r"\sonnet's \kp\ is read from the judge records the main results use.}",
             r"\label{tab:frame}",
             r"\setlength{\tabcolsep}{4pt}",
             r"\begin{tabular}{l rr l cc cc}", r"\toprule",
             r" & \multicolumn{3}{c}{unweighted panel \bacc} "
             r"& \multicolumn{2}{c}{members at $\kp \ge 0.30$} "
             r"& \multicolumn{2}{c}{\sonnet\ \kp} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
             r"Dataset & \defon & claim & $\Delta$ "
             r"& \defon & claim & \defon & claim \\",
             r"\midrule"]
    for ds, r in results.items():
        p = r["members"]["PANEL"]
        d = p["delta_bacc"]
        nd = sum(1 for m, v in r["members"].items()
                 if m != "PANEL" and v["defon"]["kappa"] >= 0.30)
        nc = sum(1 for m, v in r["members"].items()
                 if m != "PANEL" and v["claimcheck"]["kappa"] >= 0.30)
        k = len(r["models"])
        lines.append(f"{DSNAME.get(ds, ds)} & ${100*p['defon']['bacc']:.1f}$ & "
                     f"${100*p['claimcheck']['bacc']:.1f}$ & "
                     f"${100*d[0]:+.1f}$\\,{{\\scriptsize[{100*d[1]:+.1f},\\,{100*d[2]:+.1f}]}} & "
                     f"${nd}/{k}$ & ${nc}/{k}$ & "
                     f"${frontier_kappa(ds, 'sonnet_defon'):.3f}$ & "
                     f"${frontier_kappa(ds, 'sonnet'):.3f}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\tablelegend{\emph{claim}: claim-support frame. "
              r"$\Delta$: paired difference (claim $-$ \defon) with its 95\% "
              r"cluster-bootstrap interval, the same resample applied to both arms.}",
              r"\end{table}"]
    (out / "tab_frame.tex").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'tab_frame.tex'}")


if __name__ == "__main__":
    main()
