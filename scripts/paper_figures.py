#!/usr/bin/env python
r"""RAIM paper figures (matplotlib + seaborn, LaTeX text), in the palette of the TikZ
figures. From scripts/, in the README's environment:

    ../.venv/bin/python3 paper_figures.py            # -> figures/*.pdf
    ../.venv/bin/python3 paper_figures.py --png      # also PNGs for preview
    ../.venv/bin/python3 paper_figures.py --no-usetex

Datasets are ordered consistently by mean single-member kappa (descending) across the
per-dataset figures. Numbers come from the derived JSONs via paper_tables /
analyze_base_probe.

  fig_competence       violin + strip of the ten member kappas, with stacked, oracle
                       and Sonnet.
  fig_regimeforest     (a) the regime plane with the admissibility quadrant;
                       (b) the paired forest against three comparators.
  fig_paircorr         pairwise error-correlation matrices of all eight panels.
  fig_loo              heatmap of leave-one-out Delta-kappa (members x datasets).
  fig_budget           (a) the panel-size sweep; (b) the gold-label learning curve.
  fig_baseprobe        split violins of the 28 pairwise phi, base vs instruct.
  fig_transfer_narrow  in-domain -> transferred (LODO) on the four core sets.
  fig_frontier         the open 32B/72B judges against the stacked panel and Sonnet.
  fig_weights          member kappa and stacker coefficient per dataset.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
import seaborn as sns

from paper_tables import DATASETS, CORE_ORDER, load, member_kappa, RUNS
from analyze_base_probe import load_panel, err_matrix, REV

# raim_lib sits one level up; put the repository root on the path so this
# script runs the same whether driven by make or invoked directly.
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from raim_lib.verdicts import exists as verdict_exists
from raim_lib.verdicts import resolve as zone

C = dict(stacked="#7E2F8E", unweighted="#8C8C8C", best="#A2142F", avg="#0072BD",
         frontier="#D95319", transfer="#B79BC4",
         clean="#5A9E2F", win="#9ACD5A", tie="#8C8C8C", chance="#C9C9C9")

# How a paired difference is marked, in the four-way form tab_master's count line
# uses: direction by triangle, resolution by fill. The markers encode the outcome of
# a paired difference, which is what a forest carrying three comparators needs; the
# decision rule -- a win is a significant gain over the CV-best, clean if it also
# beats the oracle -- is defined against one comparator only, and is stated in the
# text.
#   key: (ahead?, interval excludes zero?) -> (marker, colour, size, filled)
OSTYLE = {
    (True,  True):  ("^", C["clean"], 44, True),
    (True,  False): ("^", C["clean"], 34, False),
    (False, False): ("v", C["best"],  34, False),
    (False, True):  ("v", C["best"],  44, True),
}
OLABEL = {(True, True): "improvement, clear", (True, False): "improvement, unresolved",
          (False, False): "worsening, unresolved", (False, True): "worsening, clear"}


def outcome(triple):
    """(ahead?, resolved?) for a [point, lo, hi] paired difference."""
    p, lo, hi = triple
    return (p > 0, not (lo <= 0 <= hi))


def oscatter(ax, x, y, triple, scale=1.0, zorder=3):
    """Plot one point under the outcome key above."""
    mk, col, sz, filled = OSTYLE[outcome(triple)]
    ax.scatter([x], [y], marker=mk, s=sz * scale,
               facecolor=col if filled else "white", edgecolor=col if filled else col,
               linewidth=0.5 if filled else 1.0, zorder=zorder)


def ohandles():
    return [Line2D([0], [0], marker=OSTYLE[k][0], color="w",
                   markerfacecolor=OSTYLE[k][1] if OSTYLE[k][3] else "white",
                   markeredgecolor=OSTYLE[k][1],
                   markeredgewidth=0.5 if OSTYLE[k][3] else 1.0,
                   markersize=6.5, label=OLABEL[k])
            for k in [(True, True), (True, False), (False, False), (False, True)]]
LEG = dict(frameon=True, framealpha=0.92, edgecolor="0.75", facecolor="white",
           borderpad=0.4, handletextpad=0.3)
ROSTER = ["meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct",
          "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-9b-it",
          "microsoft/Phi-4-mini-instruct", "01-ai/Yi-1.5-9B-Chat-16K",
          "CohereLabs/c4ai-command-r7b-12-2024", "zai-org/glm-4-9b-chat-hf",
          "ibm-granite/granite-3.3-8b-instruct", "tiiuae/Falcon3-7B-Instruct"]
# short family tags (no parameter size) for compact axis labels
TAG = {"meta-llama/Llama-3.1-8B-Instruct": "Llama", "Qwen/Qwen2.5-7B-Instruct": "Qwen",
       "mistralai/Mistral-7B-Instruct-v0.3": "Mistral", "google/gemma-2-9b-it": "Gemma",
       "microsoft/Phi-4-mini-instruct": "Phi", "01-ai/Yi-1.5-9B-Chat-16K": "Yi",
       "CohereLabs/c4ai-command-r7b-12-2024": "Command-R", "zai-org/glm-4-9b-chat-hf": "GLM",
       "ibm-granite/granite-3.3-8b-instruct": "Granite", "tiiuae/Falcon3-7B-Instruct": "Falcon"}


def SHORT(m):
    return (m.split("/")[-1].replace("-Instruct", "").replace("-instruct", "")
            .replace("-Chat-16K", "").replace("-chat-hf", "").replace("-it", ""))


OUT = Path(__file__).resolve().parent / "figures"
# Zone 4, like OUT: the audit dumps of what the figures plot -- loo_concentration.json
# for fig_loo,
# read by nothing and produced by no derive.sh step, so they sit outside the derived
# tree `make derivecheck` grades; loo_concentration.json carries unrounded deltas,
# which move in the last bits from one machine to another.
REPORTS = Path(__file__).resolve().parent / "reports"


# xcolor: the gold kappa_0 in fig_regimeforest. \dsname and \mdname are the
# manuscript's naming macros (datasets and named judges in small caps, model
# identifiers in typewriter), restated without \xspace so figure text is set as the
# prose sets it.
PREAMBLE = (r"\usepackage{times}\usepackage{amsmath}\usepackage{xcolor}"
            r"\newcommand{\dsname}[1]{\textsc{#1}}\newcommand{\mdname}[1]{\texttt{#1}}")


def style(usetex):
    plt.rcParams.update({
        "text.usetex": usetex, "font.family": "serif", "font.serif": ["Times"],
        "text.latex.preamble": PREAMBLE,
        # the PDF is written through the pgf backend (see save()), which sets its
        # text with pdflatex and therefore needs the same preamble of its own
        "pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "pgf.preamble": PREAMBLE,
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.grid": False, "figure.dpi": 150,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def save(fig, name, png, usetex=True):
    """Write the figure, PDF through the pgf backend when usetex is on.

    Why not the default PDF backend: under `text.usetex` it embeds the dvi's
    fonts directly, and the maths minus lives at slot 0 of cmsy10 --- a code
    point it silently drops, so every negative tick label and every `$-$` in an
    axis label would print WITHOUT its minus. The pgf backend sets the text with
    pdflatex instead, which embeds the glyph correctly.

    metadata CreationDate=None removes ONE source of spurious churn: without it
    every regeneration stamps the current time, so every file differs even when the
    plot is identical. It does not make the output byte-reproducible: a re-run on
    the same machine still rewrites every PDF with its font-subset tag changed,
    whilst all of them rasterise identically. `git status` therefore cannot tell a
    regeneration from an edit here, which is why figcheck.py compares pixels rather
    than bytes; `make figcheck` is the question this cannot answer."""
    kw = dict(backend="pgf") if usetex else {}
    fig.savefig(OUT / f"{name}.pdf", metadata={"CreationDate": None}, **kw)
    if png:
        fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"wrote {name}.pdf")


# ---- shared helpers ----------------------------------------------------------
def _tex():
    return plt.rcParams["text.usetex"]


def DSN(name):
    """A dataset or named judge, set as the manuscript sets it (plain without usetex)."""
    return rf"\dsname{{{name}}}" if _tex() else name


def DS(ds):
    """A dataset's display name, via DSN."""
    return DSN(DATASETS[ds][0])


def MD(name):
    """A model identifier, set as the manuscript sets it (plain without usetex)."""
    return rf"\mdname{{{name}}}" if _tex() else name


def members(ds):
    return {m: v for m, v in member_kappa(ds).items() if v == v}


def mean_member_kappa(ds):
    return float(np.mean(list(members(ds).values())))


def ds_order():
    """Single consistent ordering: descending mean single-member kappa, so the
    competence violin is monotonic; reused across all per-dataset figures."""
    return sorted(DATASETS, key=lambda d: -mean_member_kappa(d))


def mean_corr(ds):
    return load(ds)["diversity"]["sweep"][-1]["mean_corr"]


def spread(ds):
    return sum(1 for v in members(ds).values() if v >= 0.30)


def kap(ds, m):
    return load(ds)["weighted"]["kappa"][m]


def sonnet(ds):
    p = zone(RUNS / ds / "judge_sonnet.json")
    return json.loads(p.read_text())["kappa"][0] if p.exists() else np.nan


# ================================================================================
def _draw_forest(ax, verdict_legend=True, compact=False, xmax=None):
    """Forest body, drawn onto a caller-supplied axes.

    verdict_legend=False suppresses the in-axes verdict key so the paired
    single-column variant (fig_regimeforest) can carry one shared key for both
    panels instead of repeating it twice. compact=True replaces the boxed baseline
    key with rotated labels down the empty left strip (no interval reaches below
    about $-0.08$), which costs no vertical space at all."""
    order = ds_order()
    ax.axvline(0, ls="--", lw=0.7, color="black", alpha=0.5)
    # Three sub-rows per dataset, so that every paired difference the paper decides
    # on carries its interval somewhere in the main text, which lets tab_master print
    # no brackets and stay readable.
    # Datasets are set 1.6 apart rather than 1.0 so the three sub-rows of one dataset
    # group visibly tighter than the gap to the next, which is what lets the eye read
    # the panel by dataset instead of as twenty-four unrelated bars.
    step, off = 1.6, 0.36
    ys = [step * (len(order) - i) for i in range(len(order))]
    SERIES = [("stacked-best_single_cv", "o", C["best"], 30, "vs best single (CV)"),
              ("stacked-average_member", "s", C["avg"], 26, "vs average member"),
              ("panel_vs_sonnet", "D", C["frontier"], 24, rf"vs frontier ({DSN('Sonnet')})")]
    for i, ds in enumerate(order):
        y = ys[i]
        w = load(ds)["weighted"]
        if i:   # faint rule between datasets, at the midpoint of the gap
            ax.axhline(y + step / 2, color="0.90", lw=0.5, zorder=0)
        for j, (key, mk, col, sz, _) in enumerate(SERIES):
            if key == "panel_vs_sonnet":
                trip = json.loads((RUNS / ds / "panel_vs_sonnet.json").read_text()
                                  )["diff_panel_minus_frontier"]
            else:
                trip = w["diffs"][key]
            p, lo, hi = trip
            yy = y + off * (1 - j)
            ax.plot([lo, hi], [yy, yy], color=col, alpha=0.55, lw=1.0, zorder=2)
            ax.scatter([p], [yy], marker=mk, s=sz, facecolor=col, edgecolor="black",
                       linewidth=0.4, zorder=3)
    ax.set_yticks(ys)
    ax.set_yticklabels([DS(d) for d in order])
    ax.set_ylim(min(ys) - step / 2, max(ys) + step / 2)
    ax.set_xlim(-0.32, xmax or 0.42)
    if xmax:   # see _draw_regime: the extension seats the key, it is not data range
        ax.set_xticks([-0.2, 0.0, 0.2, 0.4])
    ax.set_xlabel(r"stacked $-$ comparator, Cohen's $\kappa$",
                  **({"fontsize": 8} if compact else {}))
    if verdict_legend:
        ax.legend(handles=[Line2D([0], [0], marker=mk, color="w", markerfacecolor=col,
                                  markeredgecolor="black", markersize=6.5, label=lab)
                           for _, mk, col, _, lab in SERIES],
                  loc="upper center", ncol=3, columnspacing=1.0,
                  bbox_to_anchor=(0.5, -0.16), **LEG)




def _draw_regime(ax, verdict_legend=True, compact=False, xmax=None, admissible=False):
    """Regime-plane body, drawn onto a caller-supplied axes (see _draw_forest).

    compact=True shrinks the point labels and opens the limits, so the eight
    dataset names still clear the axes at half width. xmax extends the axis past
    the data to open a corner for an in-axes key; it drops TruthfulQA's label under
    its marker, because the extension pulls the points together in figure units and
    its centred name then runs into RAGTruth's. Under, not left: TruthfulQA sits at
    x=0.26, so a leftward label runs off the axis instead."""
    jit = {}   # no two datasets coincide in the plane; see the y-limits below
    lab = {"aggrefact_wice": (-8, 0, "right"), "aggrefact_expertqa": (-8, 0, "right"),
           "aggrefact_xsum": (8, 0, "left"), "aggrefact_cnn": (8, 0, "left"),
           "factscore": (8, 0, "left")}   # left-most point: label inwards
    if xmax:
        lab["truthfulqa"] = (0, -9, "center")
    if admissible:
        # The admissibility test at the paper's thresholds is a quadrant of this
        # plane (at least SPREAD members competent, phi at most PHI), and the eight
        # calls are unchanged while its corner stays in the box below, whose edges
        # are the datasets' own coordinates (regime_budget.json, threshold_stability).
        # Edges sit at the thresholds' own values; a point on an edge is admitted,
        # the count and the lower correlation bound being inclusive.
        from regime_budget import SPREAD_FAVOURABLE, PHI_FAVOURABLE
        ts = json.loads((RUNS / "_summary" / "regime_budget.json").read_text())["threshold_stability"]
        y0 = SPREAD_FAVOURABLE
        ax.add_patch(Rectangle((-1, y0), 1 + PHI_FAVOURABLE, 20, facecolor="0.92",
                               edgecolor="none", zorder=0))
        # The two thresholds drawn in, full span and in the palette's gold, so that
        # each axis is tied to its parameter: the share c as a count of the ten
        # (y = c N) and the correlation bar phi_0; the quadrant is their corner.
        GOLD, GOLD_TXT = "#EDB120", "#9A7200"
        ax.axhline(y0, color=GOLD, lw=1.0, zorder=2)
        ax.axvline(PHI_FAVOURABLE, color=GOLD, lw=1.0, zorder=2)
        ax.text((xmax or 0.95) - 0.03, y0 + 0.2, rf"$c={SPREAD_FAVOURABLE / 10:.1f}$",
                fontsize=6, color=GOLD_TXT, ha="right", va="bottom")
        ax.text(PHI_FAVOURABLE + 0.015, 7.6, rf"$\phi_0={PHI_FAVOURABLE:.2f}$", fontsize=6,
                color=GOLD_TXT, ha="left", va="center", rotation=90)
        ax.scatter([PHI_FAVOURABLE], [y0], marker="o", s=16, color=GOLD, edgecolor="black",
                   linewidth=0.4, zorder=6)   # the corner: the operating point itself
        # The region of corner positions keeping every call, exactly: per integer
        # count k (drawn as the band k-1 < y <= k) a phi interval from the largest
        # admitted phi to the smallest phi of a rejected dataset with >= k competent
        # members, unbounded where there is none (drawn to the axis edge, its right
        # end left open). Each edge touches the dataset that sets it.
        reg = sorted(ts["call_region"], key=lambda r: r["count"])
        right = xmax or 0.95   # to the axis edge, as far as the gold c line runs
        his = [r["phi_hi"] if r["phi_hi"] is not None else right for r in reg]
        lo, k0, k1 = reg[0]["phi_lo"], reg[0]["count"] - 1, reg[-1]["count"]
        path = [(lo, k0), (lo, k1)]
        for r, h in zip(reversed(reg), reversed(his)):
            path += [(h, r["count"]), (h, r["count"] - 1)]
        with matplotlib.rc_context({"hatch.linewidth": 0.35, "hatch.color": "0.55"}):
            ax.fill([p[0] for p in path], [p[1] for p in path], facecolor="none",
                    edgecolor="0.6", linewidth=0, hatch="///", zorder=1)
        closed = path + [path[0]]
        for (xa, ya), (xb, yb) in zip(closed[:-1], closed[1:]):
            if not (xa == xb == right):          # the unbounded end stays open
                ax.plot([xa, xb], [ya, yb], color="0.45", lw=0.6, zorder=1)
        lab["ragtruth"] = (8, 0, "left")   # its centred name would sit on the box
        lab["factscore"] = (6, 6, "left")  # off the c line, north-east
        lab["aggrefact_expertqa"] = (0, 9, "center")   # above: clear of the phi_0 line and of CNN
        lab["aggrefact_xsum"] = (6, -6, "left")   # off the region's inner corner, south-east
        ax.text(0.04, 10.35, "admissible", fontsize=5.8, color="0.35", ha="left", va="center")
        # beneath the unbounded strip, mirroring the c label above it
        ax.text((xmax or 0.95) - 0.03, reg[-1]["count"] - 1 - 0.2, "invariance region",
                fontsize=5.8, color="0.35", ha="right", va="top")
    for ds in DATASETS:
        x, s = mean_corr(ds), spread(ds) + jit.get(ds, 0.0)
        oscatter(ax, x, s, load(ds)["weighted"]["diffs"]["stacked-best_single_cv"],
                 scale=1.5 if not compact else 1.2)
        dx, dy, ha = lab.get(ds, (0, 9, "center"))
        ax.annotate(DS(ds), (x, s), textcoords="offset points",
                    xytext=(dx, dy), ha=ha, va="center", fontsize=6.5 if compact else 8)
    # y counts members, so it is bounded by 0 and 10: keep the ticks integral and
    # the padding tick-free, or a point at 0 reads as if it were negative.
    # x must reach CNN's 0.854, the highest error-correlation in the suite, or the
    # plane silently shows seven of the eight datasets.
    if compact:
        ax.set_xlim(0.02, xmax or 0.95); ax.set_ylim(-0.6, 10.9)
    else:
        ax.set_xlim(0.05, 0.93); ax.set_ylim(-0.5, 10.7)
    if xmax:
        # tick the range the data occupy and no further: the extension exists to
        # seat the key, and ticking it would advertise empty plane as measured
        ax.set_xticks([0.25, 0.50, 0.75])
    ax.set_yticks(range(0, 11, 2))
    if compact:   # the full sentences do not fit a half-width panel
        ax.set_xlabel(r"mean error-correlation $\bar\phi$ $\rightarrow$" if admissible
                      else r"mean error-correlation $\rightarrow$", fontsize=8)
        gold_k0 = (r"\textcolor[HTML]{9A7200}{\kappa_0{=}0.30}" if plt.rcParams["text.usetex"]
                   else r"\kappa_0{=}0.30")
        ax.set_ylabel(r"members $\kappa\!\geq\!" + gold_k0 + r"$ $\rightarrow$" if admissible
                      else r"members $\kappa\!\geq\!0.30$ $\rightarrow$", fontsize=8)
    else:
        ax.set_xlabel(r"mean pairwise error-correlation $\rightarrow$ \emph{more redundant}")
        ax.set_ylabel(r"members $\kappa\!\geq\!0.30$ $\rightarrow$ \emph{competence more widely held}")
    if verdict_legend:
        ax.legend(handles=ohandles(), loc="upper center", ncol=2, columnspacing=1.0,
                  bbox_to_anchor=(0.5, -0.20), **LEG)




def _draw_transfer(ax, legend_ncol=2):
    """Transfer-collapse body, drawn onto a caller-supplied axes (see _draw_forest)."""
    for i, ds in enumerate(CORE_ORDER):
        t = json.loads((RUNS / "_transfer" / f"{ds}_transfer_lodo.json").read_text())
        ind, tr = t["kappa"]["indomain_stacked"][0], t["kappa"]["transferred_stacked"][0]
        bs = t["kappa"]["best_single_cv"][0]
        am = mean_member_kappa(ds)
        ax.hlines(bs, i - 0.22, i + 0.22, color=C["best"], lw=1.4, zorder=2)
        ax.hlines(am, i - 0.22, i + 0.22, color=C["avg"], lw=1.2, ls=(0, (4, 2)),
                  zorder=2)
        ax.add_patch(FancyArrowPatch((i, ind), (i, tr), arrowstyle="-|>",
                     mutation_scale=9, color=C["stacked"], lw=1.0, zorder=2))
        ax.scatter([i], [ind], marker="o", s=40, color=C["stacked"],
                   edgecolor="black", linewidth=0.4, zorder=3)
        ax.scatter([i], [tr], marker="o", s=40, color=C["transfer"],
                   edgecolor=C["stacked"], linewidth=0.6, zorder=3)
    ax.set_xticks(range(len(CORE_ORDER)))
    ax.set_xticklabels([DS(d) for d in CORE_ORDER])
    ax.set_ylim(0, 0.8); ax.set_ylabel(r"Cohen's $\kappa$")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C["stacked"],
                      markeredgecolor="black", markersize=7, label="in-domain stacked"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor=C["transfer"],
                      markeredgecolor=C["stacked"], markersize=7, label="transferred (LODO)"),
               Line2D([0], [0], color=C["best"], lw=1.4, label="best single (CV)"),
               Line2D([0], [0], color=C["avg"], lw=1.2, ls=(0, (4, 2)),
                      label="average member")]
    ax.legend(handles=handles, loc="upper right", ncol=legend_ncol,
              columnspacing=1.0, **LEG)




# ---- single-column variants -------------------------------------------------------
# A single-column page has a 5.5in text block, so a figure drawn for two columns
# and set at 0.85\columnwidth there prints ~4.7in wide, upscaled, its axis text
# larger than the body. These variants redraw the same bodies at their true
# printed size.

def fig_regimeforest(png):
    """Regime plane and forest paired side by side, for a single-column page.

    Each panel carries its own key: (a) marks the outcome of one comparison and (b)
    distinguishes three comparisons, so a key covering both would misread one of
    them. Each key sits INSIDE its own panel, upper right, with the x-axis extended
    past the data to open the corner, so that neither key can be read against the
    wrong panel and the plotting area keeps the canvas. The extensions are
    left un-ticked (see both bodies) so that empty plane does not read
    as measured range."""
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(5.4, 2.35))
    _draw_regime(axl, verdict_legend=False, compact=True, xmax=1.30, admissible=True)
    _draw_forest(axr, verdict_legend=False, compact=True, xmax=0.95)
    for ax, tag in ((axl, "(a)"), (axr, "(b)")):
        ax.set_title(tag, loc="left", fontsize=9, pad=2)
        ax.tick_params(labelsize=7)
    fig.subplots_adjust(wspace=0.32, bottom=0.19, left=0.10, right=0.98, top=0.92)
    axl.legend(handles=ohandles() + [Line2D([0], [0], color="#EDB120", lw=1.0, marker="o",
                                            markersize=3.5, markerfacecolor="#EDB120",
                                            markeredgecolor="black", markeredgewidth=0.4,
                                            label="admissibility threshold")],
               loc="upper right", ncol=1, fontsize=5.8,
               borderaxespad=0.3, **LEG)
    axr.legend(handles=[Line2D([0], [0], marker=mk, color="w", markerfacecolor=col,
                               markeredgecolor="black", markersize=6, label=lab)
                        for mk, col, lab in
                        (("o", C["best"], "vs best single (CV)"),
                         ("s", C["avg"], "vs average member"),
                         ("D", C["frontier"], rf"vs frontier ({DSN('Sonnet')})"))],
               loc="upper right", ncol=1, fontsize=5.8, borderaxespad=0.3, **LEG)
    return fig


def fig_transfer_narrow(png):
    """Transfer collapse for a single-column page: the same four core grounded
    sets and the same body, on a 3:1 canvas printing 1:1 at \\textwidth, with the
    caption beneath it. The tex, figcheck and the figure trees key on the name."""
    fig, ax = plt.subplots(figsize=(5.4, 1.8))
    _draw_transfer(ax, legend_ncol=2)
    return fig


def qwenk(ds, size):
    return json.loads(zone(RUNS / ds / f"judge_qwen{size}_awq.json").read_text())["kappa"][0]


def fig_frontier(png):
    """Qwen open-judge scaling (32B->72B) vs the cheap stacked panel and Sonnet."""
    order = ds_order()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for i, ds in enumerate(order):
        q32, q72 = qwenk(ds, "32b"), qwenk(ds, "72b")
        ax.plot([i, i], [q32, q72], color="0.6", lw=0.8, zorder=1)  # scaling segment
        ax.scatter(i, q32, marker="v", s=28, color="#6BAED6", edgecolor="black",
                   linewidth=0.3, zorder=3)
        ax.scatter(i, q72, marker="^", s=28, color="#2171B5", edgecolor="black",
                   linewidth=0.3, zorder=3)
        ax.scatter(i, kap(ds, "stacked")[0], marker="D", s=26, color=C["stacked"],
                   edgecolor="black", linewidth=0.3, zorder=4)
        ax.scatter(i, sonnet(ds), marker="*", s=85, color=C["frontier"],
                   edgecolor="black", linewidth=0.3, zorder=4)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([DS(d) for d in order], rotation=30, ha="right")
    ax.set_ylabel(r"Cohen's $\kappa$ vs gold")
    handles = [Line2D([0], [0], marker="D", color="w", markerfacecolor=C["stacked"],
                      markeredgecolor="black", markersize=7, label="stacked panel (cheap)"),
               Line2D([0], [0], marker="v", color="w", markerfacecolor="#6BAED6",
                      markeredgecolor="black", markersize=7, label=MD("Qwen2.5-32B")),
               Line2D([0], [0], marker="^", color="w", markerfacecolor="#2171B5",
                      markeredgecolor="black", markersize=7, label=MD("Qwen2.5-72B")),
               Line2D([0], [0], marker="*", color="w", markerfacecolor=C["frontier"],
                      markeredgecolor="black", markersize=11, label=DSN("Claude Sonnet") + " (frontier)")]
    ax.legend(handles=handles, loc="upper right", ncol=2, columnspacing=1.0, **LEG)
    return fig


def fig_competence(png):
    # Drawn at the same 3:1 canvas as fig_budget: the information is entirely
    # horizontal -- eight columns of ten points -- and the two figures, which the
    # reader meets a page apart, then sit at the same weight.
    order = ds_order()
    data = [sorted(members(ds).values()) for ds in order]
    pos = np.arange(1, len(order) + 1)
    fig, ax = plt.subplots(figsize=(7.1, 2.35))
    ax.axhline(0, lw=0.5, color="black", alpha=0.3)
    parts = ax.violinplot(data, pos, widths=0.8, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(C["unweighted"]); b.set_alpha(0.22); b.set_edgecolor("none")
    rng = np.random.default_rng(0)
    for j in range(len(order)):
        xs = pos[j] + rng.uniform(-0.08, 0.08, len(data[j]))
        ax.scatter(xs, data[j], s=7, color=C["unweighted"], alpha=0.7, zorder=2,
                   edgecolor="none")
    ax.scatter(pos, [kap(d, "stacked")[0] for d in order], marker="o", s=34,
               color=C["stacked"], edgecolor="black", linewidth=0.4, zorder=4)
    ax.scatter(pos, [kap(d, "oracle_best")[0] for d in order], marker="s", s=26,
               color=C["best"], edgecolor="black", linewidth=0.4, zorder=4)
    ax.scatter(pos, [sonnet(d) for d in order], marker="*", s=80,
               color=C["frontier"], edgecolor="black", linewidth=0.4, zorder=4)
    # Upright tick labels on the wide canvas: rotated ones cost about 0.35in of
    # height, which is most of what the reshaping bought back. In small caps the
    # longest name (TruthfulQA) is wider than in upright text, so the labels are set
    # a point smaller to clear their neighbours unrotated at 0.9in per slot.
    ax.set_xticks(pos); ax.set_xticklabels([DS(d) for d in order], fontsize=7)
    ax.set_ylabel(r"Cohen's $\kappa$ vs gold")
    handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor=C["stacked"],
                      markeredgecolor="black", markersize=7, label="stacked panel"),
               Line2D([0], [0], marker="s", color="w", markerfacecolor=C["best"],
                      markeredgecolor="black", markersize=7, label="best single (oracle)"),
               Line2D([0], [0], marker="*", color="w", markerfacecolor=C["frontier"],
                      markeredgecolor="black", markersize=11, label=DSN("Claude Sonnet") + " (frontier)"),
               Line2D([0], [0], marker="o", color="w", markerfacecolor=C["unweighted"],
                      markeredgecolor="none", markersize=5, label="single members")]
    ax.legend(handles=handles, loc="upper right", ncol=2, columnspacing=1.0, **LEG)
    return fig


LOO_CLIP = 0.07   # colour-scale limit; see fig_loo


def fig_loo(png):
    """Leave-one-out heat map, every cell annotated.

    Two choices. (i) The colour scale is clipped at LOO_CLIP: the largest absolute
    drop in the matrix, gemma on CNN (-0.158), is a single cell three times any
    other, and a scale running to it would flatten the seven remaining columns into
    one pale band; that cell saturates, its value still printed (the caption says
    so). (ii) RdBu puts red at the load-bearing end and blue at the redundant one,
    as the caption describes."""
    order = ds_order()
    rows = ROSTER
    M = np.full((len(rows), len(order)), np.nan)
    for j, ds in enumerate(order):
        loo = {e["dropped"]: e["delta_stacked_vs_full"]
               for e in json.loads((RUNS / ds / "loo.json").read_text())["leave_one_out"]}
        for i, m in enumerate(rows):
            M[i, j] = loo.get(m, np.nan)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    im = ax.imshow(M, cmap="RdBu", vmin=-LOO_CLIP, vmax=LOO_CLIP, aspect="auto")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([DS(d) for d in order],
                                                         rotation=30, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([MD(TAG[m]) for m in rows])
    # every cell carries its value; the column's extremes are bold, so the
    # most load-bearing member and the most rewarding drop still read at a glance
    for j in range(len(order)):
        lo_i, hi_i = int(np.nanargmin(M[:, j])), int(np.nanargmax(M[:, j]))
        for i in range(len(rows)):
            if np.isnan(M[i, j]):
                continue
            extreme = i in (lo_i, hi_i)
            ax.text(j, i, f"{M[i, j]:+.03f}"[:6], ha="center", va="center",
                    fontsize=5.4, fontweight="bold" if extreme else "normal",
                    color="white" if abs(M[i, j]) > 0.85 * LOO_CLIP else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, extend="min")
    cb.set_label(r"$\Delta\kappa$ from dropping member", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return fig


def write_loo_summary():
    """Audit dump of the leave-one-out concentration numbers behind fig_loo (each
    member's cost, sorted, plus the top-vs-runner-up ratio that distinguishes a
    dominant-single from a distributed regime)."""
    summary = {}
    for ds in DATASETS:
        full = json.loads((RUNS / ds / "loo.json").read_text())["full_stacked"][0]
        rows = sorted(
            json.loads((RUNS / ds / "loo.json").read_text())["leave_one_out"],
            key=lambda r: r["delta_stacked_vs_full"])
        costs = [dict(model=r["dropped"], delta=-r["delta_stacked_vs_full"])
                 for r in rows]
        top, runner_up = costs[0]["delta"], costs[1]["delta"]
        summary[ds] = dict(
            full_stacked_kappa=full, costs_sorted_desc=costs,
            top_vs_runner_up_ratio=(top / runner_up if runner_up > 0 else None))
    dest = REPORTS
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "loo_concentration.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {dest.name}/loo_concentration.json")
    return summary


def pairwise_phi(ds, kind):
    v, g, _ = load_panel(RUNS / "lp_probe" / f"{ds}_lp{kind}_{REV}_panel.jsonl")
    uids = sorted(g)
    with np.errstate(invalid="ignore"):
        Cm = np.corrcoef(err_matrix(v, g, uids))
    iu = np.triu_indices(Cm.shape[0], 1)
    return Cm[iu][~np.isnan(Cm[iu])]


def fig_baseprobe(png):
    order = [d for d in ds_order() if verdict_exists(RUNS / "lp_probe" / f"{d}_lpbase_{REV}_panel.jsonl")]
    rows = []
    for ds in order:
        for kind, lab in (("base", "base"), ("inst", "instruct")):
            for val in pairwise_phi(ds, kind):
                rows.append({"ds": DS(ds), "panel": lab, "phi": val})
    df = pd.DataFrame(rows)
    names = [DS(d) for d in order]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    sns.violinplot(data=df, x="ds", y="phi", hue="panel", split=True, cut=0,
                   inner="quartile", density_norm="width", order=names,
                   hue_order=["base", "instruct"],
                   palette={"base": C["avg"], "instruct": C["frontier"]},
                   linewidth=0.6, ax=ax)
    ax.axhline(0, lw=0.5, color="black", alpha=0.3)
    ax.set_xlabel(""); ax.set_ylabel(r"pairwise error-correlation $\phi$")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(title=None, loc="lower left", **LEG)
    return fig


def _roster_idx(fm):
    """Indices into fm (which may hold full ids or short names) in ROSTER order."""
    out = []
    for m in ROSTER:
        s = m.split("/")[-1]
        out.append(fm.index(m) if m in fm else fm.index(s) if s in fm else None)
    keep = [(k, i) for k, i in enumerate(out) if i is not None]
    return [k for k, _ in keep], [i for _, i in keep]


def fig_paircorr(png):
    """All eight panels, in ascending mean error-correlation, so that the reader can
    check the correlation ordering the account of \\S\\ref{ssec:mechanism} rests on."""
    sel = sorted(DATASETS, key=mean_corr)
    cmap = plt.get_cmap("PuOr").copy(); cmap.set_bad("0.7")   # grey diagonal
    ncol = 4
    fig, axes = plt.subplots(2, ncol, figsize=(6.6, 4.0))
    im = None; rkeep = []
    for k, (ax, ds) in enumerate(zip(axes.ravel(), sel)):
        d = load(ds)["diversity"]
        rkeep, idx = _roster_idx(d["final_models"])
        M = np.array(d["corr_matrix"], float)[np.ix_(idx, idx)]
        M = np.ma.masked_array(M, mask=np.eye(M.shape[0], dtype=bool))
        im = ax.imshow(M, cmap=cmap, vmin=-0.8, vmax=0.8, aspect="equal")
        # Two text objects rather than one two-line title: under usetex TeX sets a
        # two-line string as one box, and the bar of the second line's phi then
        # reaches the first line's baseline, showing beside a short dataset name.
        ax.set_title(rf"$\bar\phi={mean_corr(ds):.2f}$", fontsize=8, pad=2)
        ax.annotate(DS(ds), xy=(0.5, 1.0), xycoords="axes fraction", xytext=(0, 12),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if k % ncol == 0:                       # member names on the left column only
            ax.set_yticks(range(len(rkeep)))
            ax.set_yticklabels([MD(TAG[ROSTER[i]]) for i in rkeep], fontsize=5.5)
    fig.subplots_adjust(right=0.9, wspace=0.12, hspace=0.35)
    cax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
    cb = fig.colorbar(im, cax=cax, ticks=[-0.5, 0, 0.5])
    cb.set_label(r"pairwise error-correlation $\phi$", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    return fig


def _draw_ksweep(ax):
    """Panel-size sweep, drawn onto a caller-supplied axes. The end-of-line label
    positions below are hand-placed in data coordinates, so they hold at any
    figure aspect."""
    sel = ["medhallu", "aggrefact_wice", "ragtruth", "aggrefact_xsum", "factscore", "truthfulqa", "aggrefact_expertqa", "aggrefact_cnn"]
    cols = {"medhallu": "#1f77b4", "aggrefact_wice": "#2ca02c", "ragtruth": "#d62728",
            "aggrefact_xsum": "#9467bd", "factscore": "#ff7f0e", "truthfulqa": "#17becf",
            "aggrefact_expertqa": "#aabb44", "aggrefact_cnn": "#bfbfbf"}
    for ds in sel:
        sw = load(ds)["diversity"]["sweep"]
        Ks = [s["K"] for s in sw]; ks = [s["stacked_kappa"][0] for s in sw]
        # the bootstrap interval each K carries, shaded to match panel (b)'s IQR
        # band -- without it the flat curves read as precise rather than as
        # unresolved, which is the opposite of the point they make
        lo = [s["stacked_kappa"][1] for s in sw]
        hi = [s["stacked_kappa"][2] for s in sw]
        ax.fill_between(Ks, lo, hi, color=cols[ds], alpha=0.13, linewidth=0)
        ax.plot(Ks, ks, marker="o", ms=3, lw=1.3, color=cols[ds])
    # Direct labels replace the legend (fewer eye-trips back and forth), each
    # hand-placed where it clears its neighbours (FActScore and TruthfulQA end 0.03
    # apart).
    for ds in sel:
        if ds == "factscore":
            ax.annotate(DS(ds), (9.7, 0.495), ha="right", va="center",
                        fontsize=6.5, color=cols[ds])
            continue
        if ds == 'aggrefact_wice':
            ax.annotate(DS(ds), (9.995, 0.55), ha="right", va="bottom",
                        fontsize=6.5, color=cols[ds])
            continue
        if ds == 'aggrefact_xsum':
            ax.annotate(DS(ds), (9.995,0.445), ha="right", va="bottom",
                        fontsize=6.5, color=cols[ds])
            continue
        if ds == 'medhallu':
            ax.annotate(DS(ds), (1.005, 0.595), ha="left", va="top",
                        fontsize=6.5, color=cols[ds])
            continue
        if ds == 'truthfulqa':
            ax.annotate(DS(ds), (1.005, 0.49), ha="left", va="center",
                        fontsize=6.5, color=cols[ds])
            continue
        if ds == 'aggrefact_expertqa':
            ax.annotate(DS(ds), (9.995, 0.25), ha="right", va="bottom",
                        fontsize=6.5, color=cols[ds])
            continue
        if ds == 'ragtruth':
            ax.annotate(DS(ds), (9.995, 0.385), ha="right", va="center",
                        fontsize=6.5, color=cols[ds])
            continue
        if ds == 'aggrefact_cnn':
            ax.annotate(DS(ds), (9.995, 0.325), ha="right", va="center",
                        fontsize=6.5, color=cols[ds])
            continue
    ax.set_xlabel(r"panel size $K$ (most-decorrelated first)")
    ax.set_ylabel(r"stacked Cohen's $\kappa$")
    ax.set_xticks(range(1, 11))
    ax.set_xlim(0.6, 10.4)




def fig_budget(png):
    """The two deployment budgets on one canvas: how many judges, how many labels.

    (a) is the panel-size sweep, drawn by the same body as fig_ksweep.
    (b) is the aggregator's gold-label learning curve from stacker_curve.py, one
    line per dataset with its interquartile band and the label-free majority vote
    on the same folds dotted beside it; each point's fill says whether it beats that
    vote, its shape whether it beats the best member selected on the same labels.
    """
    import json as _json
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.1, 2.35))
    _draw_ksweep(axa)
    axa.set_title(r"(a) panel size", fontsize=9)

    from raim_lib.verdicts import DERIVED as _D
    sc = _json.loads((_D / "_summary" / "stacker_curve.json").read_text())

    # Same colour scheme and dataset selection as panel (a), so the two panels
    # are read together. One line per dataset in absolute kappa: a suite mean
    # would move with which datasets admit a budget rather than with the budget.
    cols = {"medhallu": "#1f77b4", "aggrefact_wice": "#2ca02c", "ragtruth": "#d62728",
            "aggrefact_xsum": "#9467bd", "factscore": "#ff7f0e", "truthfulqa": "#17becf",
            "aggrefact_expertqa": "#aabb44", "aggrefact_cnn": "#bfbfbf"}
    for d in sc["per_dataset"]:
        ds = d["dataset"]
        xs = [r["budget"] for r in d["budgets"]]
        ys = [r["stacked"]["mean"] for r in d["budgets"]]
        lo = [r["stacked"]["lo"] for r in d["budgets"]]
        hi = [r["stacked"]["hi"] for r in d["budgets"]]
        unw = d["unweighted_reference"]
        axb.plot(xs, ys, "-", color=cols[ds], linewidth=1.0)
        axb.fill_between(xs, lo, hi, color=cols[ds], alpha=0.13, linewidth=0)
        # the label-free majority vote on the same folds: the floor a budget has
        # to clear before fitting an aggregator is worth anything at all.
        axb.plot([xs[0], xs[-1]], [unw] * 2,
                 linestyle=(0, (1, 2)), color=cols[ds], linewidth=0.8, alpha=0.75)
        # Two orthogonal comparisons, encoded on two channels, because they are
        # NOT nested: which is the harder bar varies by dataset and by budget, so an
        # ordinal scale would misdescribe them.
        #   fill  -- does it beat the label-free majority vote?
        #   shape -- does it beat the best member selected ON THE SAME labels,
        #            i.e. the other thing that budget could have bought?
        bsv = [r["best_single"]["mean"] for r in d["budgets"]]
        for x, v, bs in zip(xs, ys, bsv):
            axb.scatter([x], [v], s=15 if v > bs else 13,
                        marker="s" if v > bs else "o",
                        facecolor=cols[ds] if v > unw else "white",
                        edgecolor=cols[ds], linewidth=0.7, zorder=3)
        axb.annotate(DS(ds), (xs[-1], ys[-1]), fontsize=6.5,
                     color=cols[ds], ha="left", va="center",
                     xytext=(4, 0), textcoords="offset points")

    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    # Tick the canonical grid only. Each dataset also contributes a final point at
    # its own training-pool size (87, 176, ... 1088), and ticking those overprints
    # the axis with seven irregular labels that say nothing a reader needs.
    from stacker_curve import BUDGETS as _BUD
    allb = {r["budget"] for d in sc["per_dataset"] for r in d["budgets"]}
    ticks = [b for b in _BUD if b in allb]
    axb.set_xscale("log")
    axb.xaxis.set_major_locator(FixedLocator(ticks))
    axb.xaxis.set_major_formatter(FixedFormatter([str(x) for x in ticks]))
    axb.xaxis.set_minor_locator(NullLocator())
    axb.set_xlim(ticks[0] * 0.85, ticks[-1] * 3.1)
    axb.set_xlabel("labelled records the aggregator is fitted on")
    axb.set_ylabel(r"stacked Cohen's $\kappa$ (held-out)")
    axb.set_title(r"(b) calibration budget", fontsize=9)
    key = [Line2D([0], [0], marker=mk, color="w", markerfacecolor=fc,
                  markeredgecolor="0.35", markeredgewidth=0.7, markersize=5, label=lb)
           for mk, fc, lb in [("o", "white", "beats neither"),
                              ("o", "0.35", "beats majority vote"),
                              ("s", "white", "beats best member"),
                              ("s", "0.35", "beats both")]]
    axb.legend(handles=key, loc="lower right", ncol=2, fontsize=6,
               columnspacing=0.9, handletextpad=0.3, **{k: v for k, v in LEG.items()
                                                        if k != "handletextpad"})
    fig.tight_layout()
    return fig




def fig_weights(png):
    """Who is who, and how much the stacker leans on each of them.

    The left panel gives every member's own kappa per dataset, named, the right its
    mean cross-fit coefficient in the stacker. Weights come from
    derived/_summary/stacker_weights.json (stacker_weights.py)."""
    W = json.loads((RUNS / "_summary" / "stacker_weights.json").read_text())
    order = ds_order()
    rows = ROSTER
    K = np.full((len(rows), len(order)), np.nan)
    B = np.full((len(rows), len(order)), np.nan)
    for j, ds in enumerate(order):
        for i, m in enumerate(rows):
            K[i, j] = W[ds]["kappa"].get(m, np.nan)
            B[i, j] = W[ds]["beta"].get(m, np.nan)

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(6.6, 3.4))
    bv = np.nanmax(np.abs(B))
    for ax, M, cmap, kw, tag, lab in (
            (axl, K, "Blues", dict(vmin=0, vmax=np.nanmax(K)), "(a)",
             r"member Cohen's $\kappa$ vs gold"),
            (axr, B, "RdBu_r", dict(vmin=-bv, vmax=bv), "(b)",
             r"stacker coefficient $\beta$")):
        im = ax.imshow(M, cmap=cmap, aspect="auto", **kw)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([DS(d) for d in order], rotation=35,
                           ha="right", fontsize=6.5)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([MD(TAG[m]) for m in rows] if ax is axl else [], fontsize=7)
        ax.set_title(tag, loc="left", fontsize=9, pad=3)
        # within-column rank (1 = best member on that dataset), so the ordering is
        # readable down a column even though the columns differ wildly in level
        rank = np.full(M.shape, 0, int)
        for j in range(M.shape[1]):
            col = M[:, j]
            good = np.where(~np.isnan(col))[0]
            for r, i in enumerate(good[np.argsort(-col[good], kind="stable")], 1):
                rank[i, j] = r
        for i in range(len(rows)):
            for j in range(len(order)):
                if np.isnan(M[i, j]):
                    continue
                shade = abs(M[i, j] - (kw.get("vmin") or 0)) / (kw["vmax"] - kw["vmin"])
                val = f"{M[i, j]:.2f}" if ax is axl else f"{M[i, j]:+.1f}"
                ax.text(j, i, f"{val}\n({rank[i, j]})", ha="center", va="center",
                        fontsize=4.6, linespacing=0.95,
                        color="white" if shade > 0.72 else "black")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(lab, fontsize=7)
        cb.ax.tick_params(labelsize=6)
    fig.subplots_adjust(wspace=0.34)
    return fig





FIGS = {"fig_competence": fig_competence, "fig_regimeforest": fig_regimeforest,
        "fig_paircorr": fig_paircorr, "fig_loo": fig_loo, "fig_budget": fig_budget,
        "fig_baseprobe": fig_baseprobe, "fig_transfer_narrow": fig_transfer_narrow,
        "fig_frontier": fig_frontier, "fig_weights": fig_weights}


def main():
    # Both destinations are redirectable, which is what lets `make figcheck`
    # regenerate everything this script writes into a scratch tree and compare:
    # the PDFs by pixel, the two dumps by tolerance.
    global OUT, REPORTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-usetex", action="store_true")
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--out", default=str(OUT), metavar="DIR",
                    help="where the figures go (default figures/)")
    ap.add_argument("--reports", default=str(REPORTS), metavar="DIR",
                    help="where the audit dumps go (default reports/)")
    args = ap.parse_args()
    OUT, REPORTS = Path(args.out), Path(args.reports)
    OUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    style(usetex=not args.no_usetex)
    for n in (args.only or list(FIGS)):
        save(FIGS[n](args.png), n, args.png, usetex=not args.no_usetex)
    write_loo_summary()
    print(f"\nfigures -> {OUT}")


if __name__ == "__main__":
    main()
