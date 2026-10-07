#!/usr/bin/env python3
"""Verify the verdict mirror, and name the measurement release it came from.

This is the zone-2 pin. The mirror under verdicts/ is this repository's only input,
and it is committed — so the question "which data release produced this table?"
is answered by a string:

    release_digest  <64 hex digits>

which names the exact set of verdict files raim-verdicts held when the mirror was
exported. Alongside it sits dataset_index_digest, naming the join index those
verdicts are keyed against; a release and an index are only meaningful together.

What this checks, and why each part matters:

  - every file recorded in the manifest is present and hashes as recorded, so a
    truncated checkout or a half-finished clone announces itself rather than
    producing quietly incomplete numbers;
  - no verdict file is present that the manifest does not know about, since an
    untracked file is one the recorded release did not contain;
  - the manifest's own provenance is printed, so a reader can quote the release a
    paper was built from rather than a date;
  - the running environment is compared with the versions requirements.txt pins,
    and any difference is named: a different NumPy, SciPy or scikit-learn moves the
    derived numbers within tolerance, and a different matplotlib or seaborn moves
    the figures by pixel, so a failing check is then explained before it is run.
    A difference warns and never fails, since it need not change anything.

Usage:
    python -m raim_lib.mirror                # verify verdicts/
    python -m raim_lib.mirror --verdicts X   # verify elsewhere
    python -m raim_lib.mirror --digest   # print the release digest and nothing else
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os as _os
import sys
from pathlib import Path

# RAIM_VERDICTS like every other reader, so that pointing the whole analysis at
# another mirror is one environment variable rather than one variable plus a flag
# only this module needs.
DEFAULT_RUNS = Path(_os.environ.get(
    "RAIM_VERDICTS", Path(__file__).resolve().parent.parent / "verdicts"))
REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"


def read_manifest(runs: Path) -> dict:
    p = runs / "MANIFEST.json"
    if not p.exists():
        sys.exit(f"!! no manifest at {p}\n"
                 f"   The verdict mirror should be committed alongside this code. "
                 f"Re-export it from raim-verdicts with tools/export_votes.py.")
    return json.loads(p.read_text())


def verify(runs: Path) -> int:
    manifest = read_manifest(runs)
    recorded = manifest["files"]

    bad, missing = [], []
    for rel, meta in sorted(recorded.items()):
        p = runs / rel
        if not p.exists():
            missing.append(rel)
            continue
        if hashlib.sha256(p.read_bytes()).hexdigest() != meta["sha256"]:
            bad.append(rel)

    # Both halves of zone 2: the projected verdicts and the judge summaries beside
    # them, so that an unrecorded summary is seen as well as an unrecorded verdict.
    from .verdicts import is_measurement
    present = {p.relative_to(runs).as_posix() for p in runs.rglob("*")
               if p.is_file()
               and (p.name.endswith(".jsonl.gz") or is_measurement(p.name))}
    untracked = sorted(present - set(recorded))

    src = manifest.get("source", {})
    print(f"  mirror        {runs}")
    print(f"  exported      {manifest.get('generated')} by {manifest.get('generator')}")
    print(f"  release       {src.get('release_digest', '(unrecorded)')}")
    print(f"  join index    {src.get('dataset_index_digest', '(unrecorded)')}")
    commit = src.get("raim_verdicts_commit")
    print(f"  measured from raim-verdicts {commit[:12] if commit else '(commit not recorded in this release)'}")
    print(f"  files         {len(recorded)}, {manifest['mirror_bytes']/1e6:.1f} MB "
          f"(projected from {manifest['source_bytes']/1e6:.0f} MB)")
    if manifest.get("dropped_fields"):
        print(f"  dropped       {', '.join(manifest['dropped_fields'])} "
              f"(no consumer reads them)")
    print()

    for rel in missing:
        print(f"  MISSING    {rel}")
    for rel in bad:
        print(f"  CORRUPT    {rel}")
    for rel in untracked:
        print(f"  UNTRACKED  {rel}  (not part of the recorded release)")

    if missing or bad:
        print(f"\n!! the mirror does not match its manifest: {len(missing)} missing, "
              f"{len(bad)} corrupt. Numbers derived from it do NOT correspond to "
              f"release {src.get('release_digest', '?')[:12]}.")
        return 1
    if untracked:
        print(f"\nOK, with {len(untracked)} untracked file(s): everything recorded "
              f"verifies, but the tree holds more than the release did.")
        return 0
    print("OK: the mirror matches its manifest exactly.")
    return 0


def environment() -> None:
    """Compare the installed packages with requirements.txt's pins, and say so."""
    from importlib import metadata
    if not REQUIREMENTS.exists():
        return
    pins = [l.split("==") for l in REQUIREMENTS.read_text().splitlines()
            if "==" in l and not l.lstrip().startswith("#")]
    off = []
    for name, want in ((n.strip(), v.strip()) for n, v in pins):
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = None
        if have != want:
            off.append(f"{name} {have or 'not installed'} (pinned {want})")
    py = ".".join(map(str, sys.version_info[:3]))
    if not off:
        print(f"  environment   Python {py}, every package as requirements.txt pins it")
        return
    print(f"\n!! environment (Python {py}, {sys.executable}) differs from requirements.txt:")
    for o in off:
        print(f"     {o}")
    print("   The derived numbers may then differ within tolerance and the figures by pixel;\n"
          "   install requirements.txt into raim-analysis/.venv, which make then uses by default,\n"
          "   or pass another interpreter as PY.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts", "--runs", dest="runs", type=Path,
                    default=DEFAULT_RUNS)
    ap.add_argument("--digest", action="store_true",
                    help="print the release digest alone, for scripting")
    args = ap.parse_args()

    if args.digest:
        src = read_manifest(args.runs).get("source", {})
        print(src.get("release_digest", ""))
        return
    status = verify(args.runs)
    environment()
    sys.exit(status)


if __name__ == "__main__":
    main()
