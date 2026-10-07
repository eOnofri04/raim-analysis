# RAIM-analysis — everything that regenerates from the judgements

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2609.39229-b31b1b)](https://arxiv.org/abs/2609.39229)
[![Companion](https://img.shields.io/badge/companion-raim--verdicts-238636)](https://github.com/eOnofri04/raim-verdicts)
[![Reproducible](https://img.shields.io/badge/reproducible-23_tables,_9_figures-blue)](#reproducing-the-paper)

This is the reproducible half of the companion code for the paper:

> **RAIM: Robust Aggregation of Inexpensive Models for Hallucination Detection**  
> Elia Onofri and Roberto Di Pietro  
> _Computer, Electrical and Mathematical Sciences and Engineering (CEMSE) Division, King Abdullah University of Science and Technology (KAUST), Thuwal, Saudi Arabia_  
> [arXiv:2609.39229](https://arxiv.org/abs/2609.39229)

Large language models are increasingly used as automatic judges of whether a generated response is faithful to its source, yet the strongest judges are proprietary and costly to run at scale.
RAIM asks whether a panel of ten cheap, open-weight small judges can be aggregated into a viable alternative to a single strong judge, and, more usefully, *when* it can.
Its companion, [`raim-verdicts`](https://github.com/eOnofri04/raim-verdicts), ran those judges over eight faithfulness benchmarks and recorded what each said about each item.

This repository turns those recordings into every number, table and figure in the paper.
**Everything here is reproducible**: no GPU, no network, no API keys.
The judges' judgements are committed alongside the code, and the whole chain from them to the paper is deterministic at `seed=0` and runs on a laptop.

The cut between the two is **irreproducible-for-a-reader versus reproducible-from-data**, so this repository makes neither GPU-gated calls nor API calls.
Nothing here reaches into the measurement repository, by import or by invocation: the judgements cross over as the committed mirror under `verdicts/`.
The paper's _Code and data availability_ section gives a general overview of both repositories.

## Table of contents

1. [Layout](#layout)
2. [Installation](#installation)
3. [Reproducing the paper](#reproducing-the-paper)
4. [Numbers in the text](#numbers-in-the-text)
5. [Other commands](#other-commands)
6. [The verdict mirror](#the-verdict-mirror)
7. [Artefact zones](#artefact-zones)
8. [Citation](#citation)
9. [Licence](#licence)
10. [Contact](#contact)

## Layout

```
raim_lib/               # the method, shared by every generator
  weighting.py          # the cross-fitted stacker and the aggregation ladder
  bootstrap.py          # the [point, lo, hi] interval convention
  verdicts.py           # reading verdict files, plain or gzipped; the zone boundary
  mirror.py             # verifying the mirror against its manifest

scripts/                # the generators, and the tables and figures they write
  paper_tables.py       # the dataset registry, the loaders, and the main tables
  paper_figures.py      # every figure
  weighted.py           # the aggregation ladder over one panel, with paired intervals
  leave_one_out.py, diversity.py, stacker_weights.py, stacker_curve.py
                        # member ablation, panel size, stacker weights, calibration budget
  panel_vs_frontier.py  # the paired panel-minus-frontier differences
  transfer.py, transfer_lodo.py          # leave-one-dataset-out transfer
  analyze_*.py          # the recovery test, the predictor and aggregator baselines,
                        # the frame contrast, the base-versus-instruct probe
  regime_budget.py, admissibility_*.py   # the admissibility test and its label budget
  external_*.py, cost_table.py           # the published comparisons and the cost model
  abstain_encoding.py, best_on_average.py
  headline_numbers.py   # the headline numbers of the text, each with its source (make numbers)
  derive.sh             # zone 3 from zone 2
  derivecheck.py        # diffing two trees of artefacts: float drift tolerated, nothing else
  figcheck.py           # which figure PDFs actually changed, ignoring font-subset churn
  tables/  figures/     # generated, and committed (ZONE 4)
  reports/              # the audit dumps beside two of them (ZONE 4)
  provenance/           # the sources of the hand-transcribed constants
  Makefile              # verify | derive | tables | figures | reports | check | figcheck | derivecheck | derivediff

verdicts/               # ZONE 2: what raim-verdicts measured, mirrored read-only
derived/                # ZONE 3: what `make derive` computes from it
requirements.txt        # the pinned analysis environment
```

The dataset registry, the display names and the table layouts live in `scripts/`, beside the tables they produce; `raim_lib/` holds only the method.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

The versions in `requirements.txt` are the ones every artefact here was built under (Python 3.13); they also install, and every check passes, under Python 3.12 and 3.14.
Timings below span the three machines tested, from a recent laptop to a low-capability Linux server: `make check` 3 to 10 minutes, `make derive` and `make derivecheck` twenty minutes to about an hour.
The figures are set through LaTeX, so `make figures` and `make figcheck` also need `pdflatex`, and `make figcheck` needs `pdftoppm` (poppler) as well.
Besides `times`, `amsmath` and `xcolor`, matplotlib's pgf backend loads `pgf`, `geometry`, `hyperref`, `underscore` and KOMA-Script's `scrextend`, its usetex mode needs `type1cm` and `type1ec`, and Times needs its Type 1 fonts.
A full TeX Live has all of them, whereas a distribution's minimal TeX Live splits them into separate packages: on Debian and Ubuntu, `texlive-latex-extra` and `texlive-fonts-recommended` supply them; on Fedora, `texlive-collection-latexextra` and `texlive-collection-fontsrecommended`.
A missing font shows as `cannot open encoding file 8r.enc`, a missing package as `File ... .sty not found`.
On a machine without them, `make preview` (under "Other commands") still draws the figures, though not as published.
No GPU, no network, no API keys: the judgements are already here.

> **Note on platforms.**
> The procedures here were tested on macOS (Python 3.13) and on two Linux machines (Python 3.12 and 3.14).
> The Makefiles and shell scripts assume a POSIX shell, so on Windows run them under WSL; native Windows is untested.

## Reproducing the paper

```bash
cd scripts
make verify   PY=../.venv/bin/python3   # seconds: is the data intact, and which release is it?
make check    PY=../.venv/bin/python3   # 3 to 10 minutes: regenerate into a scratch tree and diff; changes nothing
make figcheck PY=../.venv/bin/python3   # under a minute: the figures, likewise, by pixel; changes nothing
make tables   PY=../.venv/bin/python3   # 3 to 10 minutes: rewrites tables/ in place
make figures  PY=../.venv/bin/python3   # under a minute: rewrites figures/ in place
```

`tables/` and `figures/` are shipped as the paper used them, so that the checks have something to compare against.
`make tables` and `make figures` overwrite them in place, which is why they come last: a check run afterwards would only compare the regeneration with itself.
All four read the committed derived JSONs under `derived/`, which is the point of committing them.

> **They also read the judgements directly, which matters if you swap the mirror.**
> Most generators open verdict files as well, for the item-level work no summary carries — member κ, error correlations, the recovery regression.
> So replacing `verdicts/` and re-running `make tables` moves exactly those numbers and leaves the summary-derived ones untouched.
> If you analyse a different tree, replace the mirror **and** run `make derive`.

To rebuild the derived JSONs from the judgements themselves:

```bash
make derive      PY=../.venv/bin/python3                  # rewrites derived/
make derivecheck PY=../.venv/bin/python3                  # the same into a scratch tree, diffed; changes nothing
make derivediff  PY=../.venv/bin/python3 TREE=/path/to/it # just the diff, against a tree you already have
```

> **Note on the tolerance.**
> The chain is seeded, but the last bits of the fits depend on the BLAS and hence on the machine: on the machines tested, about a third of the derived files come back differing, at the fourteenth decimal or beyond.
> `make derivecheck` therefore compares to a tolerance — `THRESHOLD`, default `1e-8`, absolute below 1 and relative above it — and counts byte-identical, within-tolerance and differing files separately.
> The tolerance covers floats and nothing else: integers, strings, booleans, key sets and list lengths are compared exactly, so an `n` that moved or a scoring frame that flipped is a real change at any threshold.

`make check` answers the question worth asking after any change: did any number move?
It regenerates every committed table *and* the committed reports into a temporary directory and diffs against the committed set, so it is safe on a dirty tree, and it exits non-zero when anything differs or when a committed artefact exists that no generator rebuilds.
The figures are the other half, and a separate target because they need LaTeX:

```bash
make figcheck PY=../.venv/bin/python3  # the figures by pixel, and the dump beside them
```

> **Why by pixel.**
> matplotlib stamps every PDF with a random font-subset tag, so a re-run changes every file byte for byte and none of them by pixel; `figcheck` regenerates them into a scratch tree and rasterises both sides.
> When a figure differs, `make figcheck` leaves `scripts/figcheck-diffs/<figure>_diff.png`: the shipped figure faded, with every differing pixel in red and a red box around them.
> On a TeX installation other than a full TeX Live, the only differences may be a few pixels of glyph anti-aliasing in the small-caps labels, which these images make plain.
> A figure whose differing pixels are fewer than `FIGTOL` of the whole (0.1% by default) is therefore reported as MINIMAL, with its image, and does not fail the run; `make figcheck FIGTOL=0` fails on any differing pixel.
> The data behind the figures is what `make check` and `make derivecheck` compare exactly, so a change in it fails there whatever `FIGTOL` allows.

> **Note on the provenance banner.**
> Every generated table opens with a comment naming the script that wrote it, so `make check` compares from the second line.
> That is also the answer to "where does this number come from?": start at the table, read line 1, and open the generator it names.
> The published reference values set beside ours are hand-transcribed constants; `scripts/provenance/PUBLISHED_REFERENCE_VALUES.md` gives, for each, the table, page, row and column of its source.

> **The frame is not optional.**
> The four LLM-AggreFact sets are scored under a claim-support frame, the other four under the question–answer frame.
> Only `weighted.py` detects this by itself; `derive.sh` passes the frame to everything else, and anything new must too.

## Numbers in the text

Every number in the paper's text is in one of three places, and most are in the first.

1. **The table or figure it comments on**, which the text cites beside it.
2. **The derived JSONs of that section's generator**, under `derived/`: the per-dataset files (`weighted.json`, `panel_vs_sonnet.json`, `loo.json`, `diversity.json`) and the cross-dataset ones under `_summary/`, `_transfer/` and `_extras/`, each named after the analysis that writes it.
3. **An aggregate over datasets that no single file holds**, such as a median over the eight or a count of wins, which the headline numbers are.

The headline numbers of the abstract, the contributions and the conclusion are printed by one command, each beside the file and key it is read from:

```bash
make numbers PY=../.venv/bin/python3   # read-only; seconds
```

It covers the frontier comparison (the median share of the frontier's kappa, the balanced accuracy given up, and the eight paired differences in their four-way form), the wins over the best member and their multiplicity adjustment, transfer, the calibration budget, and the cost model's per-item prices and break-even volumes.
The comparisons with published systems are read from the mean column of `tab_extbase` and from `tab_profile` directly.
The values quoted from cited papers are recorded, with their source, in `scripts/provenance/PUBLISHED_REFERENCE_VALUES.md`.

## Other commands

```bash
make help    PY=../.venv/bin/python3   # every target, one line each
make reports PY=../.venv/bin/python3   # the cost model's rate table and per-item costs, as the paper prices them
```

One dataset can be re-derived on its own, in one or two minutes rather than twenty to sixty (its `weighted`, `loo` and `diversity` JSONs; the cross-dataset steps need all eight), and compared with the shipped files:

```bash
PY=../.venv/bin/python3 DERIVE_OUT=/tmp/derived bash derive.sh medhallu
../.venv/bin/python3 derivecheck.py --new /tmp/derived --ref ../derived   # the files of the other datasets show as "only in the reference"
```

Pass `DERIVE_OUT`, or `derive.sh` rewrites the shipped `derived/` in place.

Several generators also print, as they run, numbers the text quotes and no table carries: `external_baselines_table.py` prints the per-1,000-item prices and the break-even volumes, and `best_on_average.py` the ranking behind the best-on-average member.
Run them with `--out` pointing elsewhere, since by default the first rewrites its shipped tables in place:

```bash
../.venv/bin/python3 external_baselines_table.py --out /tmp/tabs
../.venv/bin/python3 best_on_average.py --out /tmp/tabs
```

Without LaTeX, the figures can still be drawn for inspection, with matplotlib's own text rendering, into `preview/` rather than over the shipped figures:

```bash
make preview PY=../.venv/bin/python3   # all nine, as PDF and PNG, in seconds
```

A preview shows the data and the layout but not the published typesetting, so it differs from the shipped figures in every pixel and `make figcheck` does not apply to it.

## The verdict mirror

`verdicts/` holds the judgements as `.jsonl.gz`, about 5 MB.
They are a **projection** of the measurement, not the measurement itself: the judge's generated text is dropped, because nothing in this repository reads it; [`raim-verdicts`](https://github.com/eOnofri04/raim-verdicts) carries the same records with the text, as `runs.tar.xz`.

### Which data release produced this table?

That question has a string answer, carried in `verdicts/MANIFEST.json` and printed by `make verify`:

```
release       77778b28868fdbbae971247c7d218e67e61dd03fba885c505861a8710fa00aeb
join index    49705322fd0ca907bfd3a1210bcb95ca3becc00f8e101b912c88b780c452b070
```

The first names the exact set of verdict files released with the paper (it matches `verdicts.lock.json` in `raim-verdicts`); the second names the join index those judgements are keyed against.
`make verify` additionally re-hashes every mirrored file, so a truncated copy announces itself rather than producing quietly incomplete numbers.

> **Note.**
> Read verdicts through `raim_lib.verdicts`, and check for them with `raim_lib.verdicts.exists()` rather than `Path(...).exists()`: a file may be plain or gzipped, and a raw existence check on the wrong suffix silently returns false.

## Artefact zones

| zone | contents | reproducible? | policy |
|---|---|---|---|
| 1. `raim-verdicts` output | raw judgements, the join index | no: GPU hours and API spend | [`raim-verdicts`](https://github.com/eOnofri04/raim-verdicts): `runs.tar.xz`, `dataset_index/` |
| 2. this repository's input | `verdicts/` — the mirror, plus the judge summaries | — | read-only; nothing here writes it |
| 3. derived | `derived/` — what `make derive` computes | yes: seeded, reproducible to float drift | shipped and diffable |
| 4. output | `scripts/tables/`, `figures/`, `reports/` | yes | shipped, as the paper consumes them |

Zone 2 is never written by anything in this repository, and that is structural rather than behavioural: the two zones are separate trees, so a mistyped `--out` cannot land on a verdict file.
Membership is decided by **authorship, not extension**: a judge's summary `judge_<tag>.json` sits in zone 2 beside its verdicts, and `raim_lib.verdicts.is_measurement` is the predicate.

## Citation

If you use this code or the released verdicts, please cite the paper:

```bibtex
@misc{Onofri_DiPietro_26,
  title         = {{RAIM}: Robust Aggregation of Inexpensive Models for Hallucination Detection},
  author        = {Onofri, Elia and Di Pietro, Roberto},
  year          = {2026},
  eprint        = {2609.39229},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.39229}
}
```

GitHub's "Cite this repository" button reads the same entry from `CITATION.cff`.

## Licence

**CC BY-NC-SA 4.0** (`LICENSE`), matching `raim-verdicts`.

![CC BY-NC-SA 4.0](https://upload.wikimedia.org/wikipedia/commons/1/12/Cc-by-nc-sa_icon.svg?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=original)

You are free to share and adapt the material for non-commercial purposes, provided you give appropriate credit, link to the licence, and distribute any derivative under the same terms.

The source benchmarks carry their own terms, recorded in `NOTICE` and, in full, in `../raim-verdicts/NOTICE`.
None of their text is redistributed here: the judgements carry identifiers, labels and our own models' outputs, and no question, document or candidate text.
The datasets themselves are not needed to rebuild the paper.

## Contact

Elia Onofri (corresponding author), `elia[dot]onofri[at]kaust[dot]edu[dot]sa`.
Questions and bug reports are also welcome as GitHub issues.

_CRI-Lab, King Abdullah University of Science and Technology (KAUST)_
