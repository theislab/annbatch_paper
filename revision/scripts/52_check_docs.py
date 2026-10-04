from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from _common import load_config

#: (superseded value, what replaced it, phrases that legitimately mention it)
SUPERSEDED: list[tuple[str, str, tuple[str, ...]]] = [
    ("1.77x", "1.55x (codec control, matched cache regime)",
     ("inflat", "earlier", "superseded", "1.77x, an", "not 1.77", "-> 1.55",
      "from 1.55x to 1.77x", "wrong cache regime", "cacheable")),
    ("0.22 epochs", "0.058 epochs (real MappedCollection at full scale)",
     ("proxy", "earlier draft", "against 0.22")),
    ("33,344", "1,467 +- 43 samples/s (full-budget grid)",
     ("earlier draft", "intermediate note", "100,000-sample budget")),
    ("35,627", "7,679 +- 180 samples/s (full-budget grid)", ("earlier draft",)),
    ("45,639", "12,600 +- 2,040 samples/s (full-budget grid)", ("earlier draft",)),
    # 7.32 h is annbatch's own chunk_size=1 epoch; only its use as a proxy is superseded.
    ("7.32 h", "27.50 h (real MappedCollection epoch at 94.1M cells)",
     ("proxy", "earlier draft", "chunk / block size", "annbatch | one epoch")),
]

DOCS = ("FULL_PACKAGE.md", "REVIEWER_RESPONSES.md", "EXPERIMENTS.md", "README.md")


def _numbers(cell: str, unit: str) -> list[tuple[str, float]]:
    """Every `<number> <unit>` in a table cell, as (as-written, value) pairs."""
    out = []
    for raw in re.findall(rf"([\d,]+\.?\d*)\s*{unit}\b", cell):
        out.append((raw, float(raw.replace(",", ""))))
    return out


def _faithful(written: str, value: float, candidates) -> bool:
    """Does `value` round to `written` for any of the live measurements?"""
    decimals = len(written.split(".")[1]) if "." in written else 0
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(value - c) <= tol for c in candidates)


def _tables(text: str):
    """Yield (header cells, [row cells]) for every pipe table in a document."""
    rows, header = None, None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if header is not None and rows:
                yield header, rows
            rows, header = None, None
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if header is None:
            header, rows = cells, []
        elif set("".join(cells)) <= set("-: "):
            continue          # the alignment row
        else:
            rows.append(cells)
    if header is not None and rows:
        yield header, rows


def stale_wgs_breakeven(root: Path):
    """Yield any WGS break-even row the result files no longer support."""
    sys.path.insert(0, str(root / "notebooks"))
    import figures

    try:
        df = figures.wgs_epoch_table()
    except Exception as exc:                      # noqa: BLE001
        yield ("(all)", "wgs_epoch_table() is unavailable", str(exc),
               "cannot verify the WGS break-even rows")
        return

    live_epochs = list(df.break_even_epochs)
    live_hours = (list(df.epoch_h_annbatch) + list(df.epoch_h_baseline)
                  + list(df.preshuffle_h))

    for name in DOCS:
        path = root / name if name == "README.md" else root / "docs" / name
        if not path.exists():
            continue
        for header, rows in _tables(path.read_text()):
            flat_header = " ".join(header).lower()
            if "break-even" not in flat_header:
                continue
            # Scope to the WGS tables: Tahoe's carry a format column instead.
            body = " ".join(" ".join(r) for r in rows).lower()
            if "regime" not in flat_header and "memmap" not in body \
                    and "max-maf" not in body:
                continue
            for row in rows:
                line = "| " + " | ".join(row) + " |"
                for cell in row:
                    for written, value in _numbers(cell, "epochs"):
                        if not _faithful(written, value, live_epochs):
                            yield (name, f"break-even {written} epochs",
                                   "no live value rounds to it", line)
                    for written, value in _numbers(cell, "h"):
                        if not _faithful(written, value, live_hours):
                            yield (name, f"epoch time {written} h",
                                   "no live value rounds to it", line)


def stale_throughput_rows(root: Path):
    """Yield any chunk-size throughput row in the docs that the results contradict."""
    sys.path.insert(0, str(root / "notebooks"))
    import figures  # imported late: it reads the result files on import

    df = figures.throughput_chunk_table()
    df = df[df.regime == "fixed_buffer"]
    live = {(r.loader, int(r.block_size)): r.samples_per_sec
            for r in df.itertuples()}

    row = re.compile(r"^\s*\|\s*(\d+)\s*\|\s*([\d,]+)\s*\|\s*[\d.]+ h\s*\|"
                     r"\s*([\d,]+|n/a)\s*\|\s*([\d.]+ h|n/a)\s*\|\s*$")
    for name in DOCS:
        path = root / name if name == "README.md" else root / "docs" / name
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            m = row.match(line)
            if not m:
                continue
            chunk = int(m.group(1))
            for loader, group in (("annbatch", 2), ("scDataset", 3)):
                cell = m.group(group)
                if cell == "n/a":
                    continue
                quoted = float(cell.replace(",", ""))
                expected = live.get((loader, chunk))
                if expected is None:
                    continue
                if abs(quoted - expected) > max(1.0, 0.005 * expected):
                    yield (name, f"{loader} at chunk {chunk}: {cell}",
                           f"{expected:,.0f} (from the result files)",
                           line.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    root = Path(load_config()["repo"]) / "revision"
    hits = list(stale_throughput_rows(root))
    hits += list(stale_wgs_breakeven(root))
    for name in DOCS:
        path = root / name if name == "README.md" else root / "docs" / name
        if not path.exists():
            continue
        # Join wrapped lines so a nearby exculpatory phrase is still visible.
        paras = re.split(r"\n\s*\n", path.read_text())
        for para in paras:
            flat = " ".join(para.split())
            for value, replacement, allow in SUPERSEDED:
                if value not in flat:
                    continue
                if any(a.lower() in flat.lower() for a in allow):
                    continue
                hits.append((name, value, replacement, flat[:150]))

    if not hits:
        print("docs consistent: no superseded value used as a current claim, "
              "and the transcribed throughput rows match the result files")
        return 0
    print(f"{len(hits)} stale value(s) presented as current:\n")
    for name, value, replacement, ctx in hits:
        print(f"  {name}: {value!r} -> should be {replacement}")
        print(f"    ...{ctx}...\n")
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
