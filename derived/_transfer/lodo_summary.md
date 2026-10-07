# Cross-dataset transfer (LODO) -- verdicts

_Kappa uses the per-method denominator (weighted.py convention); bacc counts non-answer as error on the full denominator._
_A transfer win = (transferred - in-domain CV-best single) lower CI > 0; clean = also beats the in-domain oracle single._
_Transfer cost = (in-domain stacker - transferred); a non-significant cost means transfer is free._

## medhallu

Verdict: **no transfer win (tie/loss)** (binary interface).
Transferred stacker kappa +0.614 [+0.582, +0.645], bacc +0.807 [+0.791, +0.823].
WIN test (transferred - CV-best single): -0.079 [-0.112, -0.045].
CLEAN test (transferred - oracle single): -0.079 [-0.112, -0.045].
COST (in-domain stacker - transferred): +0.111 [+0.082, +0.139] -- significant (transfer is not free).
Coverage note: abstentions present in best_single_cv, oracle_best (committed-n below 2000); kappa excuses them, bacc charges them.

## ragtruth

Verdict: **no transfer win (tie/loss)** (binary interface).
Transferred stacker kappa +0.233 [+0.197, +0.270], bacc +0.617 [+0.598, +0.635].
WIN test (transferred - CV-best single): -0.165 [-0.200, -0.131].
CLEAN test (transferred - oracle single): -0.165 [-0.200, -0.131].
COST (in-domain stacker - transferred): +0.175 [+0.134, +0.216] -- significant (transfer is not free).
Ground note: full coverage (all methods committed n=1362); kappa and bacc denominators coincide.

## aggrefact_xsum

Verdict: **no transfer win (tie/loss)** (binary interface).
Transferred stacker kappa +0.420 [+0.345, +0.495], bacc +0.710 [+0.674, +0.748].
WIN test (transferred - CV-best single): +0.020 [-0.041, +0.085].
CLEAN test (transferred - oracle single): -0.016 [-0.085, +0.051].
COST (in-domain stacker - transferred): +0.017 [-0.038, +0.068] -- not significant (transfer is free).
Coverage note: abstentions present in best_single_cv, oracle_best (committed-n below 546); kappa excuses them, bacc charges them.

## aggrefact_wice

Verdict: **no transfer win (tie/loss)** (binary interface).
Transferred stacker kappa +0.549 [+0.435, +0.657], bacc +0.775 [+0.719, +0.829].
WIN test (transferred - CV-best single): +0.065 [-0.025, +0.155].
CLEAN test (transferred - oracle single): +0.001 [-0.063, +0.069].
COST (in-domain stacker - transferred): -0.009 [-0.096, +0.073] -- not significant (transfer is free).
Coverage note: abstentions present in best_single_cv, oracle_best, best_on_avg_single (committed-n below 222); kappa excuses them, bacc charges them.
