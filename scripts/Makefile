# The paper build — regenerate every table and figure from the committed verdicts.
#
# Everything here is deterministic at seed=0 and runs on a laptop in minutes. The
# expensive, irreproducible half — the judge verdicts themselves — lives in
# raim-verdicts and is consumed here as a committed, compressed mirror under
# ../verdicts. Nothing in this directory writes there.
#
#   make            # tables + figures
#   make check      # regenerate into a scratch tree and diff; changes nothing
#
# PY defaults to the virtual environment the README's install block creates,
# ../.venv, when it exists, and to the python3 on your PATH otherwise (with
# requirements.txt installed); `make ... PY=/path/to/python3` picks another.

PY   ?= $(if $(wildcard ../.venv/bin/python3),../.venv/bin/python3,python3)
# A PY given as a path is made absolute and free of `..`: newer Pythons warn when
# a venv's interpreter is reached through `..` (as in PY=../.venv/bin/python3).
ifneq ($(findstring /,$(PY)),)
override PY := $(abspath $(PY))
endif
HERE := $(CURDIR)

# The tolerance `derivecheck` compares zone 3 at, largest tolerated deviation:
# absolute below 1, relative above it (see derivecheck.py). Byte-identity is not
# reachable across machines -- re-deriving on a second machine, from the same
# verdicts under the same seeds, left about a third of the files differing at the
# fourteenth decimal or beyond on the machines tested, lbfgs and the correlation
# sums summing in a different order under a different BLAS. 1e-8 passes what the
# hardware perturbs and fails everything else; the smallest quantity the paper
# reports is a third-decimal kappa, five orders coarser. Tighten it to inspect
# the drift, don't loosen it.
THRESHOLD ?= 1e-8

# The share of a figure's pixels `figcheck` lets differ before it fails, 0.1%: a TeX
# installation other than a full TeX Live shifts a few pixels of glyph anti-aliasing
# (at most 0.05% on the machines tested), and such a figure is reported MINIMAL,
# with its diff image, rather than passed silently. A change in the data behind a
# figure also fails `check` or `derivecheck`, which compare that data exactly.
# FIGTOL=0 makes any differing pixel fail.
FIGTOL ?= 0.001

# `check` compares with process substitution, which /bin/sh does not provide.
SHELL := /bin/bash

.PHONY: all verify derive tables figures preview reports numbers check figcheck derivecheck derivediff clean help

help:
	@echo "make verify      — check the verdict mirror against its manifest; print the release"
	@echo "make derive      — regenerate every derived JSON from the verdicts (20-60 min)"
	@echo "make tables      — tables/*.tex from the verdicts"
	@echo "make figures     — figures/*.pdf"
	@echo "make reports     — the cost model's rate table, as the paper prices it"
	@echo "make numbers     — the headline numbers of the text, each with the file and key it is read from"
	@echo "make preview     — the figures without LaTeX, into preview/ (not the published typesetting)"
	@echo "make check       — zone 4: regenerate the tables and reports likewise, and diff"
	@echo "make figcheck    — zone 4: the figures, compared by pixel, and the dumps beside them"
	@echo "make derivecheck — zone 3: regenerate the derived JSONs likewise (20-60 min)"
	@echo "make derivediff  — compare an already-regenerated tree: TREE=/path [THRESHOLD=1e-8]"
	@echo "make clean       — remove __pycache__"

# The zone-2 pin. Verifies every mirrored verdict against its recorded checksum
# and prints the release digest the mirror was exported from -- which is what
# makes "which data release produced this table?" answerable by a string rather
# than by a date. Cheap; worth running before trusting a fresh checkout.
verify:
	@cd .. && $(patsubst ../%,%,$(PY)) -m raim_lib.mirror

# Zone 3 from zone 2: the whole deterministic chain. See derive.sh, which says
# why the frame must be passed explicitly to everything except weighted.py.
#
# Not a dependency of `tables` or `figures`: those read the committed derived set,
# which is the point of committing it. Run this when the verdicts change.
derive: verify
	PY=$(PY) bash derive.sh

all: tables figures

# Every generator that writes a table runs here: if it writes a table, it belongs
# in this list and in `check`'s. The paper build reads zone 3 and writes zone 4
# only, so the four generators that also write a derived JSON send it to /dev/null
# here -- derive.sh is what writes zone 3 -- and the fits behind the Dawid--Skene
# and frame-contrast tables are read from their JSONs rather than refitted.
tables:
	$(PY) paper_tables.py
	$(PY) analyze_mechanism.py
	$(PY) analyze_extras.py --tex
	$(PY) external_baselines_table.py
	$(PY) best_on_average.py
	$(PY) analyze_base_probe.py
	$(PY) analyze_frame_contrast.py --tex
	$(PY) regime_budget.py --tex --out /dev/null
	$(PY) external_f1.py --tex --out /dev/null
	$(PY) stacker_curve.py --tex --out /dev/null
	$(PY) abstain_encoding.py --tex --out /dev/null

# `figures` reads zone 3 and writes only zone 4: fig_weights reads the committed
# stacker_weights.json like every other figure reads its artefact, and when the
# verdicts change the thing that regenerates it is `make derive`.
figures:
	$(PY) paper_figures.py

# The figures drawn WITHOUT LaTeX, for a machine that has none: matplotlib's own
# text rendering, into preview/ rather than over the shipped figures. It shows the
# data and the layout, not the published typesetting, so figcheck does not apply.
preview:
	$(PY) paper_figures.py --no-usetex --png --out preview --reports preview

# The headline numbers of the abstract, the contributions and the conclusion, each
# printed beside the file and key it is read from. Read-only.
numbers:
	$(PY) headline_numbers.py

# The cost model: every cost figure in the paper is priced at cost_table.py's rate
# constants, and `$(PY) cost_table.py --rate NAME=VALUE` reprices them. The
# measurements the constants were transcribed from are in provenance/.
reports:
	$(PY) cost_table.py --show-rates

# The frozen-artefact check. Regenerates into a scratch directory and diffs, so it
# never touches the committed artefacts and can be run on a dirty tree, and EXITS
# NON-ZERO when anything differs. Tables are compared from line 2, skipping the
# provenance banner, which names the script that produced them. Every generator in
# `tables` is listed here; tab_roster.tex is exempt, being hand-authored.
#
# ZONE 4 IS NOT ONLY THE TABLES: the committed reports are checked too.
# loo_concentration.json is exempt HERE and checked by `make figcheck`, since it
# comes from paper_figures.py, which needs a working LaTeX. Between the two targets
# every committed zone-4 artefact is compared.
#
# The comparison is derivecheck.py, the same one zone 3 uses, so both zones report
# the same way and stored floats are compared to a tolerance. A rendered table needs no
# indulgence: it rounds to three decimals, so any drift large enough to change one
# is drift the paper would print, and it rightly fails.
check:
	@tmp=$$(mktemp -d); rep=$$tmp/reports; mkdir -p $$rep; \
	  $(PY) paper_tables.py --out $$tmp --reports $$rep >/dev/null; \
	  $(PY) analyze_mechanism.py --out $$tmp >/dev/null; \
	  $(PY) analyze_extras.py --tex --out $$tmp >/dev/null; \
	  $(PY) external_baselines_table.py --out $$tmp >/dev/null; \
	  $(PY) best_on_average.py --out $$tmp >/dev/null; \
	  $(PY) analyze_base_probe.py --out $$tmp >/dev/null; \
	  $(PY) analyze_frame_contrast.py --tex --out $$tmp >/dev/null; \
	  $(PY) regime_budget.py --tex --texdir $$tmp --out $$tmp/regime_budget.json >/dev/null; \
	  $(PY) external_f1.py --tex --texdir $$tmp --out $$tmp/external_f1.json >/dev/null; \
	  $(PY) stacker_curve.py --tex --texdir $$tmp --out $$tmp/stacker_curve.json >/dev/null; \
	  $(PY) abstain_encoding.py --tex --texdir $$tmp --out $$tmp/abstain_encoding.json >/dev/null; \
	  ok=1; \
	  echo "-- zone 4: the tables"; \
	  $(PY) derivecheck.py --new $$tmp --ref tables --patterns '*.tex' \
	      --skip-lines 1 --threshold $(THRESHOLD) --exempt tab_roster.tex || ok=0; \
	  echo "-- zone 4: the reports"; \
	  $(PY) derivecheck.py --new $$rep --ref reports --patterns '*.json' '*.md' \
	      --threshold $(THRESHOLD) --exempt $(FIGURES_WRITE) || ok=0; \
	  rm -rf $$tmp; \
	  test $$ok -eq 1

# The same check one zone down. `check` covers zone 4 -- the tables a reader sees --
# but those read the committed derived JSONs, so a number could move in zone 3 and
# `check` would still pass by regenerating a table from an artefact that had itself
# drifted. This closes that: regenerate zone 3 into a scratch tree and diff.
#
# Slow, because it is the whole deterministic chain; it is the check to run after
# touching raim_lib or any generator, not on every edit. Exits non-zero on any
# difference, and on any committed derived file the run did not produce: derive.sh
# produces every committed derived file and the paper build writes none, so there is
# no exemption list. derivecheck.py compares integers, strings, key sets and list
# lengths exactly, and floats to THRESHOLD.
#
# What `check` exempts and `figcheck` covers, and the reverse: each target declares
# what the other one is responsible for, so neither list can quietly become a gap.
FIGURES_WRITE = loo_concentration.json
TABLES_REPORTS = coverage_summary.json

# The figure half of zone 4. It regenerates every figure and the two dumps beside
# them into a scratch tree and compares: the PDFs BY PIXEL, because every PDF
# differs by `cmp` after any re-run (matplotlib's font-subset tag) and none of them
# by pixel, and the dumps by the same tolerance zone 3 uses. It needs a working
# LaTeX, which is why it is separate from `check`: the figures are set through the
# pgf backend under usetex, and a --no-usetex rebuild would not compare. For "which
# figures did I just change?" call figcheck.py directly with --rev; that needs no
# regeneration at all.
figcheck:
	@tmp=$$(mktemp -d); \
	  $(PY) paper_figures.py --out $$tmp --reports $$tmp >/dev/null; \
	  ok=1; \
	  echo "-- zone 4: the figures, by pixel"; \
	  rm -rf figcheck-diffs; \
	  $(PY) figcheck.py --against $$tmp --label "a fresh regeneration" --tolerance $(FIGTOL) --diffs figcheck-diffs || ok=0; \
	  if [ -d figcheck-diffs ]; then echo "   the differing pixels, in red: scripts/figcheck-diffs/"; fi; \
	  echo "-- zone 4: the dumps beside them"; \
	  $(PY) derivecheck.py --new $$tmp --ref reports --patterns '*.json' \
	      --threshold $(THRESHOLD) --exempt $(TABLES_REPORTS) || ok=0; \
	  rm -rf $$tmp; \
	  test $$ok -eq 1

# The regenerated tree is kept when the comparison fails, and only then: it is the
# evidence -- which threshold would have passed, and on which field -- and
# `make derivediff` re-reads it without spending the hour again.
derivecheck:
	@tmp=$$(mktemp -d); \
	  echo "==> regenerating zone 3 into $$tmp (this is the slow one)"; \
	  DERIVE_OUT=$$tmp PY=$(PY) bash derive.sh >$$tmp/.log 2>&1 || \
	    { echo "  derive.sh FAILED; tail:"; tail -20 $$tmp/.log; rm -rf $$tmp; exit 1; }; \
	  if $(PY) derivecheck.py --new $$tmp --ref ../derived \
	      --threshold $(THRESHOLD); then \
	    rm -rf $$tmp; \
	  else \
	    echo "  the regenerated tree is kept at $$tmp"; \
	    echo "  re-compare it without re-deriving: make derivediff TREE=$$tmp THRESHOLD=..."; \
	    exit 1; \
	  fi

# The comparison alone, against a tree derive.sh has already written. Regenerating
# zone 3 takes twenty minutes to an hour and comparing it a second, so choosing THRESHOLD -- or
# reading a failure -- should not cost the regeneration again.
derivediff:
	@test -n "$(TREE)" || \
	  { echo "usage: make derivediff TREE=/path/to/regenerated [THRESHOLD=1e-8]"; exit 2; }
	$(PY) derivecheck.py --new $(TREE) --ref ../derived \
	    --threshold $(THRESHOLD)

clean:
	@rm -rf __pycache__ ../raim_lib/__pycache__
	@echo cleaned
