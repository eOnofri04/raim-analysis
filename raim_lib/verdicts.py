"""Read verdict files, compressed or not — the single loading path for zone 2.

Every generator reads its verdicts through `load()` or `stream()`, so that the
format of the committed mirror is taught in one place rather than in each
caller, and a caller that could not find its input fails loudly instead of
silently reading nothing.

What the mirror is, and why it exists. The paper build reads the raw verdicts,
not merely the small derived JSONs, so the analysis layer cannot be given the
derived tier alone. But nothing here reads the `raw` field -- the judge's
generated text -- and dropping it before gzipping makes the verdict tree small
enough to commit beside the code. The projection is lossless for this
repository: every table and figure rebuilds identically from it.

So a verdict file may be `<name>.jsonl` or `<name>.jsonl.gz`, and callers need
not care. `load()` accepts either, preferring an uncompressed file when both
exist (a local re-run beats the mirror), and reports plainly when neither does
-- a missing verdict tree is the commonest way for a fresh checkout to fail,
and it should not present as an empty result.
"""
from __future__ import annotations

import gzip
import json
import os
from pathlib import Path
from typing import Iterator

_ROOT = Path(__file__).resolve().parent.parent

# The two zones, kept in separate trees so that writing into the wrong one is not
# a matter of care but of path.
#
#   VERDICTS  zone 2 -- what raim-verdicts measured, mirrored here read-only.
#             Nothing in this repository writes it; `make derive` never names it.
#   DERIVED   zone 3 -- what `make derive` computes from zone 2. Seeded and
#             reproducible, committed, and diffable.
#
# Separate trees make the clobber hazard structural rather than behavioural: a
# mistyped --out cannot land on a verdict file that costs GPU days to rebuild.
VERDICTS = Path(os.environ.get("RAIM_VERDICTS", _ROOT / "verdicts"))
DERIVED = Path(os.environ.get("RAIM_DERIVED", _ROOT / "derived"))


def is_measurement(name: str) -> bool:
    """Does `name` belong to zone 2 — that is, did raim-verdicts write it?

    The boundary is authorship, not file extension: a `.json` file may be either,
    since `scripts/run_judge.py`, `scripts/run_ptjudge.py` and
    `scripts/run_minicheck.py` emit a judge summary beside their raw output.

    Zone 2 is therefore:
      *.jsonl / *.jsonl.gz   every raw verdict file
      judge_<tag>.json       the per-judge summary written next to it
    and the one exception is judge_pair_*.json, which judge_pair.py computes HERE
    from two of those summaries, and which is consequently derived.

    The `.json` test on the second branch keeps anything else named `judge_*`
    that is written here (a per-judge plot, say) in the derived zone.
    `verdict_lock.is_measurement` in raim-verdicts applies the same predicate,
    and the two are meant to agree; the `.jsonl.gz` case is the one deliberate
    asymmetry, since this zone is mirrored gzipped and raim-verdicts' own tree
    is not.
    """
    if name.endswith(".jsonl") or name.endswith(".jsonl.gz"):
        return True
    return (name.startswith("judge_") and name.endswith(".json")
            and not name.startswith("judge_pair_"))


def _to_verdict_zone(p: Path) -> Path:
    """Map a path expressed against the derived root onto the verdict root.

    Generators address everything as `<root>/<dataset>/<name>` against the derived
    root. Rather than teach every generator which zone each filename belongs to,
    the mapping happens here, once.
    """
    try:
        rel = p.relative_to(DERIVED)
    except ValueError:
        return p
    return VERDICTS / rel


def resolve(path: Path | str) -> Path:
    """Return the readable form of `path`: right zone, and plain or gzipped.

    An uncompressed file wins over a compressed one: it is either a fresh local
    run or something the operator put there on purpose, and either way it is
    more current than a committed mirror.
    """
    p = Path(path)
    if is_measurement(p.name):
        p = _to_verdict_zone(p)
    if p.suffix == ".gz":
        plain = p.with_suffix("")
        return plain if plain.exists() else p
    if p.exists():
        return p
    gz = p.with_suffix(p.suffix + ".gz")
    return gz if gz.exists() else p


def exists(path: Path | str) -> bool:
    """Is there a readable verdict file at `path`, in either form?"""
    return resolve(path).exists()


def stream(path: Path | str) -> Iterator[dict]:
    """Yield records one at a time, in file order."""
    p = resolve(path)
    if not p.exists():
        raise FileNotFoundError(
            f"no verdict file at {p} (nor a .gz beside it).\n"
            f"  The analysis repository reads a committed mirror under verdicts/. "
            f"If this is a fresh checkout, the mirror should have come with it; if "
            f"you are pointing at a raw tree, export one with raim-verdicts' "
            f"tools/export_votes.py.\n"
            f"  NB the zone mapping is applied relative to {DERIVED}, so a path "
            f"given relative to the current directory is left in the derived tree "
            f"rather than being resolved against the verdict tree.")
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load(path: Path | str) -> list[dict]:
    """All records from a verdict file, in file order."""
    return list(stream(path))
