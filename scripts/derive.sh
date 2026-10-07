#!/usr/bin/env bash
# Regenerate zone 3 — every derived JSON — from the committed verdicts in zone 2.
#
# This is the deterministic half of the study. Each step is seeded at 0, reads
# only verdicts and writes only summaries, and reproduces its committed output
# byte for byte on the machine that exported it -- which is what makes committing
# these files sound rather than sloppy.
#
# ACROSS machines it does not, and the difference is arithmetic rather than
# analysis: on another machine about half the files come back differing in the
# ninth decimal or beyond, lbfgs and the correlation sums summing in a different
# order under a different BLAS. So compare with `make derivecheck`, which is
# tolerance-aware (THRESHOLD, default 1e-8) and still compares integers, strings,
# key sets and list lengths exactly. A failure there means a number moved; a plain
# `diff` after `make derive` means only that you changed machine.
#
# THE FRAME IS NOT OPTIONAL. Only weighted.py auto-detects a dataset's condition;
# leave_one_out.py and diversity.py default to def_on. The four LLM-AggreFact
# sets are scored under claimcheck, so they must be told so explicitly, or they
# are silently rescored under a frame that misreads them (app:extbase). Hence
# CC_DATASETS below.
#
#   bash derive.sh              # everything, the slow step (tens of minutes)
#   bash derive.sh medhallu     # one dataset
#   DERIVE_OUT=/tmp/x bash derive.sh   # write elsewhere, to diff without clobbering
set -euo pipefail
cd "$(dirname "$0")"

# PY, when unset, is the virtual environment the README creates (../.venv), then
# the python3 on the PATH.
if [[ -z "${PY:-}" ]]; then
  if [[ -x ../.venv/bin/python3 ]]; then PY=../.venv/bin/python3; else PY=python3; fi
fi
# Reads zone 2, writes zone 3. They are separate trees, so a mistyped output
# path cannot land on a verdict file that costs GPU days to rebuild.
VERDICTS="${RAIM_VERDICTS:-$(cd .. && pwd)/verdicts}"
RUNS="$VERDICTS"                       # where experiment.jsonl(.gz) is read from
OUT="${DERIVE_OUT:-$(cd .. && pwd)/derived}"
B=2000

# Every generator resolves its derived root from RAIM_DERIVED (raim_lib.verdicts,
# and the RUNS constant each one defines). Exporting it here is what makes
# DERIVE_OUT reach the steps that take no --out, so a redirected run writes
# nothing into the real tree. Zone 2 stays where it is: reads are mapped back onto
# RAIM_VERDICTS by raim_lib.verdicts.resolve.
export RAIM_DERIVED="$OUT"
export RAIM_VERDICTS="$VERDICTS"

QA_DATASETS="medhallu ragtruth truthfulqa factscore"
CC_DATASETS="aggrefact_xsum aggrefact_wice aggrefact_cnn aggrefact_expertqa"
ALL_DATASETS="$QA_DATASETS $CC_DATASETS"
CORE_TRANSFER="medhallu ragtruth aggrefact_xsum aggrefact_wice"


DATASETS="${*:-$ALL_DATASETS}"

frame_of () {  # the explicit --condition a dataset needs, or nothing
  case " $CC_DATASETS " in *" $1 "*) echo "--condition claimcheck";; *) echo "";; esac
}

for ds in $DATASETS; do
  echo "==== derive: $ds ===="
  exp="$RUNS/$ds/experiment.jsonl"
  d="$OUT/$ds"; mkdir -p "$d"
  cond="$(frame_of "$ds")"

  # --- the spine: leave-one-out, the aggregation ladder, the K-curve
  "$PY" leave_one_out.py --experiment "$exp" $cond --B $B --out "$d/loo.json"

  # weighted carries the judge overlay. --best-on-avg gemma is load-bearing: it is
  # what puts best_on_avg_model in the committed artefacts (best_on_average.py
  # shows why gemma).
  judges=""
  for t in sonnet qwen32b_awq qwen72b_awq; do
    [ -f "$RUNS/$ds/judge_${t}.json" ] && judges="$judges $RUNS/$ds/judge_${t}.json"
  done
  "$PY" weighted.py --raw "$exp" --B $B --title "$ds" $cond \
      --best-on-avg gemma ${judges:+--judges$judges} --out "$d/weighted.json"

  # The misframed def_on arm of the four LLM-AggreFact sets, stacked; app:extbase
  # quotes the stacker's edge under that frame (WiCE +0.116). The frame is pinned
  # because the arm carries def_on only.
  defon="$RUNS/$ds/experiment_defon.jsonl"
  if "$PY" -c "
import sys, pathlib; sys.path.insert(0, str(pathlib.Path('..').resolve()))
from raim_lib.verdicts import exists; sys.exit(0 if exists('$defon') else 1)"; then
    "$PY" weighted.py --raw "$defon" --condition def_on --B $B \
        --title "$ds (def_on)" --out "$d/weighted_defon.json"
  fi

  "$PY" diversity.py --experiment "$exp" $cond --order decorr --B $B \
      --title "$ds" --out "$d/diversity.json"

done

# ---- cross-dataset: only meaningful over the whole suite ---------------------
# panel_vs_frontier.py and stacker_weights.py take no --out: each computes its
# own output path from the derived root. That root is RAIM_DERIVED, which this
# script exports above, so they follow DERIVE_OUT like everything else and the
# whole chain can be regenerated into a scratch tree and diffed.
if [ -n "${*:-}" ]; then
  echo "==== cross-dataset steps SKIPPED (single-dataset run) ===="
  echo "    They are only meaningful over the whole suite; re-run with no arguments."
else
  echo "==== derive: panel versus the frontier judge ===="
  "$PY" panel_vs_frontier.py

  echo "==== derive: leave-one-dataset-out transfer ===="
  "$PY" transfer_lodo.py \
      --datasets medhallu:"$RUNS/medhallu/experiment.jsonl" \
                 ragtruth:"$RUNS/ragtruth/experiment.jsonl" \
                 aggrefact_xsum:"$RUNS/aggrefact_xsum/experiment.jsonl" \
                 aggrefact_wice:"$RUNS/aggrefact_wice/experiment.jsonl" \
      --outdir "$OUT/_transfer" --B $B \
      --summary "$OUT/_transfer/lodo_summary.md" --sonnet-tag judge_sonnet



  # ---- the frame contrast -----------------------------------------------------
  # The def_on arm sits beside each canonical one as experiment_defon.jsonl, so this
  # is an ordinary zone-2 read; the script skips per dataset when the arm is absent.
  echo "==== derive: the def_on / claim-support frame contrast ===="
  "$PY" analyze_frame_contrast.py --derive --B $B || \
    echo "    (no def_on arm in the mirror; produce it in raim-verdicts with scripts/run_contrast.sh)"


  # The Dawid--Skene baseline, the common-denominator coverage check and the Holm
  # multiplicity adjustment: one fit over all eight datasets, 4.5 minutes of EM and
  # bootstrap; the tables render from the JSON it writes here, in under a second.
  echo "==== derive: Dawid--Skene, coverage and multiplicity ===="
  "$PY" analyze_extras.py --derive

  echo "==== derive: the stacker's own coefficients ===="
  "$PY" stacker_weights.py

  # How few gold labels read the regime. Cross-dataset because the reading is
  # the suite's -- what a budget buys on eight panels, not on one -- and it
  # takes its own seed rather than $B: this resamples calibration sets, not
  # bootstrap draws, so R is a sample count and not an interval width.
  echo "==== derive: the label budget the regime diagnostic needs ===="
  "$PY" regime_budget.py

  # Hallucinated-class F1 on the two QA-form sets carrying a published
  # anchor in that metric, so the comparison outside LLM-AggreFact is a
  # measurement rather than a caveat.
  echo "==== derive: hallucinated-class F1 against the published anchors ===="
  "$PY" external_f1.py

  # The aggregator's own gold learning curve: how many labels the stacker needs.
  echo "==== derive: gold-label learning curve of the stacker ===="
  "$PY" stacker_curve.py

  # The abstention-encoding ablation behind tab_abstain: the 0.5 imputation
  # against a categorical encoding, same folds and estimator.
  echo "==== derive: the cost of the 0.5 abstention imputation ===="
  "$PY" abstain_encoding.py

  # The admissibility rule, thresholds frozen, scored on sub-panels of the same
  # ten members: out-of-configuration evidence for a rule set on eight panels.
  echo "==== derive: the admissibility call on sub-panels ===="
  "$PY" admissibility_subpanels.py

  # The same two thresholds selected, rather than declared, on those sub-panels
  # (Youden's J against substitution safety, and against the gain over the
  # CV-best), with a leave-one-dataset-out leg. Reads the JSON just written.
  echo "==== derive: the admissibility thresholds selected on sub-panels ===="
  "$PY" admissibility_tuning.py
fi

echo
echo "### derive done -> $OUT"
echo "    Diff it: a change here means a number moved, not that a build ran."
