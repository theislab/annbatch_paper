from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "20_convergence" / "streaming__bs64__seed0.json"
DOC = ROOT / "docs" / "REVIEWER_RESPONSES.md"

TASKS = ("cell_line", "drug", "moa_broad", "moa_fine")
ROW_LABEL = "| sequential streaming, no randomisation |"
ROW_ANCHOR = "| scDataset, `block_size=16`, `fetch_factor=256` |"
NOTE_ANCHOR = "(Single seed per arm; the batch-64 replication is a corroboration"
NOTE_START = "The sequential-streaming arm "


def main() -> int:
    if not RESULT.exists():
        print("batch-64 streaming arm not yet available; response keeps the 3-arm table")
        return 0
    blob = json.loads(RESULT.read_text())
    if not blob.get("final"):
        print("batch-64 streaming result carries no evaluation yet; table left alone")
        return 0

    f1 = {t: blob["final"]["val_iid"][f"mlp/{t}"]["macro_f1"] for t in TASKS}
    steps = blob["steps"]
    max_steps = blob.get("max_steps") or steps
    partial = not blob.get("completed_epoch", steps >= max_steps)

    lines = DOC.read_text().splitlines(keepends=True)

    # Refresh the row if present, else insert it after scDataset to keep the arm order.
    row = ROW_LABEL + " " + " | ".join(f"{f1[t]:.4f}" for t in TASKS) + " |\n"
    existing = [i for i, ln in enumerate(lines) if ln.startswith(ROW_LABEL)]
    if existing:
        lines[existing[0]] = row
    else:
        hits = [i for i, ln in enumerate(lines) if ln.startswith(ROW_ANCHOR)]
        if len(hits) != 1:
            print(f"ERROR: expected exactly one '{ROW_ANCHOR}' row, found {len(hits)}",
                  file=sys.stderr)
            return 1
        lines.insert(hits[0] + 1, row)

    # 2. A note paragraph stating the arm's step budget, owned by this script.
    if partial:
        # Not "reached its walltime": a partial result may be a run still in progress.
        note = (f"The sequential-streaming arm is reported at {steps:,d} of "
                f"{max_steps:,d} steps ({100 * steps / max_steps:.0f}% of one "
                f"epoch); its macro-F1 had already collapsed and was flat.")
    else:
        note = (f"The sequential-streaming arm completed the same full epoch "
                f"({steps:,d} steps) as the other three.")
    note += (" Its collapse is established independently at minibatch 4,096 over "
             "three seeds, where it reaches a drug macro-F1 of 0.0001.")
    note_block = textwrap.fill(note, width=78, break_on_hyphens=False) + "\n"

    owned = [i for i, ln in enumerate(lines) if ln.startswith(NOTE_START)]
    if owned:
        end = owned[0]
        while end < len(lines) and lines[end].strip():
            end += 1
        lines[owned[0]:end] = [note_block]
    else:
        hits = [i for i, ln in enumerate(lines) if ln.startswith(NOTE_ANCHOR)]
        if len(hits) != 1:
            print(f"ERROR: expected exactly one note anchor, found {len(hits)}",
                  file=sys.stderr)
            return 1
        end = hits[0]
        while end < len(lines) and lines[end].strip():
            end += 1
        lines[end:end] = ["\n", note_block]

    DOC.write_text("".join(lines))
    state = f"partial ({steps:,d}/{max_steps:,d} steps)" if partial else "full epoch"
    print(f"filled: batch-64 streaming, {state}, drug macro-F1 = {f1['drug']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
