from __future__ import annotations

import json
import re
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "41_wgs_preshuffle_rare_maf_0.01.json"
DOC = ROOT / "docs" / "REVIEWER_RESPONSES.md"

ROW_PREFIX = "| `--max-maf 0.01`, sparse |"
FOOTNOTE_START = "† The `--max-maf 0.01` figure is the timing recorded by the original"
RERUN_BLOCK = re.compile(
    r"\nTo re-measure it when 1500 GB is free:\n\n```bash\n.*?\n```\n", re.DOTALL)


def main() -> int:
    if not RESULT.exists():
        print("rare_maf_0.01 re-measurement not available; "
              "response keeps the archived 3.65 h and its footnote")
        return 0

    blob = json.loads(RESULT.read_text())
    hours = blob["elapsed_h"]
    out_gib = blob["output_bytes"] / 1024 ** 3
    text = DOC.read_text()

    # Matched on the row prefix and rewritten cell-wise so neighbouring edits cannot defeat it.
    lines = text.splitlines(keepends=True)
    hits = [i for i, ln in enumerate(lines) if ln.startswith(ROW_PREFIX)]
    if len(hits) != 1:
        print(f"ERROR: expected exactly one '{ROW_PREFIX}' row, found {len(hits)}",
              file=sys.stderr)
        return 1
    i = hits[0]
    cells = lines[i].rstrip("\n").split("|")
    # cells = ['', regime, variants, input, output, pre-shuffling, epoch, '']
    if len(cells) != 8:
        print(f"ERROR: row has {len(cells) - 2} columns, expected 6", file=sys.stderr)
        return 1
    cells[4] = f" {out_gib:.0f} GiB Zarr "
    cells[5] = f" **{hours:.2f} h** "
    lines[i] = "|".join(cells) + "\n"
    text = "".join(lines)

    # 2. The footnote describing the OOM, and the instructions for re-running it.
    if FOOTNOTE_START in text:
        start = text.index(FOOTNOTE_START)
        end = text.index("\n\n", start)
        # State what the comparison shows rather than asserting agreement.
        ARCHIVED_H = 3.6492  # 3 h 38 min 57 s, genetic_data_prep_2.sh, 2026-03-11
        delta = abs(hours - ARCHIVED_H) / ARCHIVED_H
        verdict = (f"agrees with this re-measurement to {delta * 100:.0f}%"
                   if delta <= 0.10 else
                   f"differs from this re-measurement by {delta * 100:.0f}%")
        # Reviewer-facing wording: no script names, no scheduling detail, no em-dashes.
        archived = (f"The archived timing from the original data-preparation run "
                    f"(3 h 38 min 57 s) {verdict}. This regime holds 2¹⁷ rows of "
                    f"~224,000 non-zeros in memory at once, about 264 GB per "
                    f"buffer before the concatenation copy, so it needs roughly "
                    f"1.5 TB of memory to run.")
        # Wrapped to match the hand-written prose around it.
        text = text[:start] + textwrap.fill(archived, width=78, break_on_hyphens=False) + text[end:]
        text = RERUN_BLOCK.sub("", text, count=1)
        # The dagger on the row is no longer pointing at anything.
        text = text.replace(f"**{hours:.2f} h**†", f"**{hours:.2f} h**")
    else:
        print("note: footnote already removed; only the row was refreshed")

    DOC.write_text(text)
    print(f"filled: rare_maf_0.01 pre-shuffling re-measured at {hours:.2f} h, "
          f"{out_gib:.0f} GiB output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
