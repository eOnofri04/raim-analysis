#!/usr/bin/env python3
"""Report which figure PDFs actually CHANGED, ignoring regeneration churn.

matplotlib stamps each PDF with a random font-subset tag, so a re-run rewrites
every file even when the plot is pixel-for-pixel the same and `git status` is
therefore useless for telling a regeneration from a real edit. This compares the
working tree against a git revision by RASTERISING both and diffing the pixels.

The same churn defeats a byte comparison between the figures here and any other
copy of them -- every PDF differs by `cmp` after a re-run, and none of them by
pixel. Hence --against, which asks the same question of another directory rather
than of a revision.

    ../.venv/bin/python3 figcheck.py             # vs HEAD
    ../.venv/bin/python3 figcheck.py --rev HEAD~3
    ../.venv/bin/python3 figcheck.py --against DIR
    ../.venv/bin/python3 figcheck.py --against DIR --diffs figcheck-diffs

Every figure that differs is reported with the share of its pixels that moved.
With --tolerance FRAC, one whose share is below FRAC is reported as MINIMAL and
does not fail the run: a different TeX installation shifts a few pixels of glyph
anti-aliasing, whereas a change in the data behind a figure also fails
`make check` or `make derivecheck`, which compare that data exactly. The default
is 0, so any differing pixel fails.

With --diffs, every figure that differs is also drawn as an image to look at: the
figure in figures/ faded, with every pixel where the other copy differs in red and
a red box around them, so that a difference of a few pixels can still be found.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Figures are generated into this directory's figures/. Pointing at them means
# figcheck compares what was just written against what git holds (or against
# --against), which is the question it exists to answer.
ROOT = Path(__file__).resolve().parent
TREES = ("figures",)
DPI = 80
DIFF_SCALE = 2          # --diffs images are drawn this many times larger


def raster(pdf: Path, out: Path) -> bytes | None:
    r = subprocess.run(["pdftoppm", "-r", str(DPI), "-png", "-singlefile",
                        str(pdf), str(out)], capture_output=True)
    png = out.with_suffix(".png")
    return png.read_bytes() if r.returncode == 0 and png.exists() else None


def pixel_diff(ours: bytes, other: bytes):
    """The differing pixels of two rasters at DPI: (count, total, box, mask, ours
    as an image, a note on the page size)."""
    from io import BytesIO
    from PIL import Image, ImageChops                # a matplotlib dependency
    a = Image.open(BytesIO(ours)).convert("RGB")
    b = Image.open(BytesIO(other)).convert("RGB")
    note = ""
    if a.size != b.size:
        note = f"; page size {b.size} against {a.size}"
        b = b.resize(a.size)
    mask = ImageChops.difference(a, b).convert("L").point(lambda v: 255 if v else 0)
    return mask.histogram()[255], a.size[0] * a.size[1], mask.getbbox(), mask, a, note


def diff_image(a, mask, box, out: Path) -> None:
    """Write `out`: `a` faded, the pixels of `mask` in red with a red box around
    them, enlarged DIFF_SCALE times."""
    from PIL import Image, ImageDraw
    size = (a.size[0] * DIFF_SCALE, a.size[1] * DIFF_SCALE)
    over = Image.blend(a.resize(size, Image.LANCZOS), Image.new("RGB", size, "white"), 0.6)
    over.paste(Image.new("RGB", size, (220, 0, 0)), mask=mask.resize(size, Image.NEAREST))
    if box:
        m = 6 * DIFF_SCALE
        x0, y0, x1, y1 = (v * DIFF_SCALE for v in box)
        ImageDraw.Draw(over).rectangle((x0 - m, y0 - m, x1 + m, y1 + m),
                                       outline=(220, 0, 0), width=DIFF_SCALE)
    out.parent.mkdir(parents=True, exist_ok=True)
    over.save(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="HEAD", help="git revision to compare against")
    ap.add_argument("--against", type=Path, default=None, metavar="DIR",
                    help="compare against the PDFs in DIR instead of against a git "
                         "revision")
    ap.add_argument("--label", default=None, metavar="TEXT",
                    help="how the summary names what was compared against (default: "
                         "the --against directory, or the revision)")
    ap.add_argument("--tolerance", type=float, default=0.0, metavar="FRAC",
                    help="report a figure whose share of differing pixels is below FRAC "
                         "(e.g. 0.001 for 0.1%%) as MINIMAL, not failing the run")
    ap.add_argument("--diffs", type=Path, default=None, metavar="DIR",
                    help="for every figure that differs, write DIR/<figure>_diff.png: "
                         "the figure in figures/ faded, the differing pixels in red")
    args = ap.parse_args()

    changed, minimal, same, missing = [], [], [], []
    notes: dict[str, str] = {}
    with TemporaryDirectory() as td:
        tmp = Path(td)
        for tree in TREES:
            d = ROOT / tree
            if not d.is_dir():
                continue
            for pdf in sorted(d.glob("*.pdf")):
                rel = f"{tree}/{pdf.name}"
                if args.against:
                    ref = args.against / pdf.name
                    if not ref.is_file():
                        missing.append(rel); continue
                else:
                    old = subprocess.run(["git", "-C", str(ROOT), "show", f"{args.rev}:./{rel}"],
                                         capture_output=True)
                    if old.returncode != 0:
                        missing.append(rel); continue
                    ref = tmp / "ref.pdf"
                    ref.write_bytes(old.stdout)
                a, b = raster(ref, tmp / "a"), raster(pdf, tmp / "b")
                if a is None or b is None:
                    missing.append(rel)
                    continue
                if a == b:
                    same.append(rel)
                    continue
                n, total, box, mask, img, note = pixel_diff(b, a)
                if n == 0:
                    same.append(rel)
                    continue
                (minimal if n / total < args.tolerance else changed).append(rel)
                notes[rel] = (f"{n} of {total} pixels differ ({100 * n / total:.3f}%) "
                              f"at {DPI} dpi, within the box {box}{note}")
                if args.diffs:
                    out = args.diffs / f"{pdf.stem}_diff.png"
                    diff_image(img, mask, box, out)
                    notes[rel] += f" -> {out}"

    for label, rels in (("CHANGED   ", changed), ("MINIMAL   ", minimal)):
        for rel in rels:
            print(f"{label} {rel}")
            print(f"           {notes[rel]}")
    for rel in missing:
        print(f"NEW/ERR    {rel}")
    against = args.label or (str(args.against) if args.against else args.rev)
    tol = (f", {len(minimal)} minimal (under {100 * args.tolerance:g}% of pixels)"
           if args.tolerance else "")
    print(f"\n{len(same)} figure(s) in figures/ identical to {against} (regeneration churn only){tol}, "
          f"{len(changed)} genuinely changed, {len(missing)} new or unreadable.")
    # A figure absent from the directory compared against is a failure of it; a figure
    # absent from an older revision is simply new, which is why only --against
    # counts `missing` against the exit code.
    return 1 if changed or (args.against and missing) else 0


if __name__ == "__main__":
    sys.exit(main())
