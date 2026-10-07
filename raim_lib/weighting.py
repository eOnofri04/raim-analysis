"""Competence-weighted aggregation with leakage-free cross-fit calibration.

The panel's votes are combined by several schemes, from the unweighted majority to
a stacked meta-learner, each of which learns whatever it learns on held-out
calibration data, so that no comparison is contaminated by the test labels.

Cross-fit (K-fold over row_ids): for each fold, weights are learned on the OTHER
folds and applied to the held-out fold, so every instance gets an out-of-fold
prediction with no leakage and all data is used for evaluation. Splitting by
row_id (not instance) keeps each row's pos/neg pair together.

Schemes (all operate on the main condition's binary votes):
  unweighted        : plain majority
  acc_weighted      : weight w_i = max(2*balanced_acc_i - 1, 0)  (informedness;
                      zeros out chance/below-chance judges)
  logodds_weighted  : w_i = log(p_i/(1-p_i)), p_i = calibration accuracy
                      (Nitzan-Paroush optimal weight under conditional independence;
                      below-chance judges get negative weight = inverted vote)
  best_single_cv    : pick the highest-calibration-accuracy model per fold, use it
                      alone (the "you chose the best model on a dev set" baseline,
                      fairer than a best single chosen on the test labels)
  stacked           : logistic-regression meta-learner on the K-vote matrix
                      (missing vote -> 0.5). Unlike per-model weighting it learns
                      the CORRELATION structure, so redundant judges share
                      influence.

A member's calibration accuracy is its balanced accuracy on the items it answers;
an abstention is neither right nor wrong for the purpose of weighting it.
"""
from __future__ import annotations
import math
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression

METHODS = ("unweighted", "acc_weighted", "logodds_weighted", "stacked",
           "best_single_cv")


def _bacc(votes_m: Dict[str, Optional[int]], gold: Dict[str, int],
          uids: List[str]) -> float:
    yt, yp = [], []
    for u in uids:
        v = votes_m.get(u)
        if v is not None:
            yt.append(gold[u]); yp.append(v)
    if len(yt) < 2 or len(set(yt)) < 2:
        return 0.5
    return balanced_accuracy_score(yt, yp)


def _weighted_pred(votes_at_uid: Dict[str, Optional[int]],
                   weights: Dict[str, float], logodds: bool) -> Optional[int]:
    if logodds:
        s = 0.0; any_vote = False
        for m, w in weights.items():
            v = votes_at_uid.get(m)
            if v is None:
                continue
            any_vote = True
            s += w * (2 * v - 1)
        if not any_vote:
            return None
        return int(s > 0)
    # accuracy-weighted mean; fall back to unweighted if all weights ~0
    num = den = 0.0
    for m, w in weights.items():
        v = votes_at_uid.get(m)
        if v is None:
            continue
        num += w * v; den += w
    if den <= 1e-9:  # all judges at/below chance -> plain majority
        vs = [v for v in votes_at_uid.values() if v is not None]
        return None if not vs else int(np.mean(vs) >= 0.5)
    return int(num / den >= 0.5)


def crossfit(votes: Dict[str, Dict[str, Optional[int]]],
             gold: Dict[str, int], uids: List[str],
             rowid_of: Dict[str, int],
             folds: int = 5, seed: int = 0, eps: float = 1e-3,
             coef_out: Optional[List[dict]] = None,
             featurise: Optional[Callable] = None):
    """Return (preds, mean_weights):
       preds[method][uid] -> 0/1/None (out-of-fold);
       mean_weights[scheme][model] -> averaged learned weight (for reporting).

    `featurise` optionally replaces the binary stacker's feature map. It is
    called as featurise(votes, models, uids) -> np.ndarray of shape
    (len(uids), d) and defaults to the 0.5-imputed K-vote matrix below. It
    serves the abstention-encoding ablation (scripts/abstain_encoding.py),
    which asks whether encoding an abstention as a CATEGORY rather than as the
    scalar 0.5 moves the panel: 0.5 asserts that an abstention lies exactly
    midway in log-odds between the two verdicts, whereas abstentions are
    categorical failures. Answering that requires the two arms to differ in the
    feature map and in NOTHING else -- same folds, same seed, same estimator,
    same 0.5 decision threshold, same degenerate-fold fallback -- which a second
    copy of this function could only promise, whereas a hook enforces it. Left
    None, the default feature map is used; the ablation checks that its
    0.5-imputed arm reproduces weighted.json's stacked kappa on every dataset.


    `coef_out` is an optional caller-supplied list, filled with one
    ``{"fold": f, "models": [...], "coef": [...], "intercept": float}`` record
    per fold on which a stacker could be fitted. Opt-in and side-effect only.
    The run JSONs record the stacker's PREDICTIONS but not its weights, so this
    is how stacker_weights.py reads how much the aggregator leans on each judge:
    from the coefficients of the fit that actually happened, rather than from a
    refit that would restate this function's fold construction, imputation and
    estimator settings in a second place."""
    models = sorted(votes)
    row_ids = sorted({rowid_of[u] for u in uids})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(row_ids)
    fold_of = {r: i % folds for i, r in enumerate(perm)}
    by_uid_fold = {u: fold_of[rowid_of[u]] for u in uids}

    preds = {m: {} for m in METHODS}
    wsum = {"acc_weighted": defaultdict(float),
            "logodds_weighted": defaultdict(float)}

    for f in range(folds):
        cal = [u for u in uids if by_uid_fold[u] != f]
        test = [u for u in uids if by_uid_fold[u] == f]
        acc = {m: _bacc(votes[m], gold, cal) for m in models}
        w_acc = {m: max(2 * acc[m] - 1, 0.0) for m in models}
        w_lo = {m: math.log(min(max(acc[m], eps), 1 - eps) /
                            (1 - min(max(acc[m], eps), 1 - eps))) for m in models}
        best = max(models, key=lambda m: acc[m])
        # stacking meta-learner: logistic regression on the K-vote matrix
        def _feat(uu):
            if featurise is not None:
                return featurise(votes, models, uu)
            return np.array([[0.5 if votes[m].get(u) is None else votes[m][u]
                              for m in models] for u in uu], float)
        stacker = None
        ycal = np.array([gold[u] for u in cal])
        fit, yfit = cal, ycal
        if len(set(yfit.tolist())) == 2:
            stacker = LogisticRegression(max_iter=1000).fit(_feat(fit), yfit)
        for m in models:
            wsum["acc_weighted"][m] += w_acc[m] / folds
            wsum["logodds_weighted"][m] += w_lo[m] / folds
        Xtest = _feat(test) if test else None
        stack_pred = (stacker.predict(Xtest) if (stacker is not None and test)
                      else None)
        if coef_out is not None and stacker is not None:
            coef_out.append(dict(fold=f, models=list(models),
                                 coef=[float(c) for c in stacker.coef_[0]],
                                 intercept=float(stacker.intercept_[0])))
        for j, u in enumerate(test):
            at = {m: votes[m].get(u) for m in models}
            vs = [v for v in at.values() if v is not None]
            preds["unweighted"][u] = None if not vs else int(np.mean(vs) >= 0.5)
            preds["acc_weighted"][u] = _weighted_pred(at, w_acc, logodds=False)
            preds["logodds_weighted"][u] = _weighted_pred(at, w_lo, logodds=True)
            preds["best_single_cv"][u] = at[best]
            preds["stacked"][u] = (int(stack_pred[j]) if stack_pred is not None
                                   else preds["unweighted"][u])
    return preds, {k: dict(v) for k, v in wsum.items()}