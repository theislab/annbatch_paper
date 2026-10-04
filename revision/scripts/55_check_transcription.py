from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from _common import load_config

NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
#: Scientific notation proves a value is in memory, not that a reader can see it.
SCIENTIFIC = re.compile(r"\d+\.?\d*[eE][+-]?\d+")
#: Small integers are prose ("two parameters", "section 4"), not claims.
BARE = {str(n) for n in range(0, 17)} | {"50", "64", "95", "100"}


def rendered(nb: dict) -> str:
    """Everything a reader of the executed notebook can actually see in code cells."""
    seen = []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        seen.append("".join(cell["source"]).replace("_", ""))
        for out in cell.get("outputs", []):
            data = out.get("data", {})
            seen.append("".join(data.get("text/plain", "")))
            seen.append("".join(data.get("text/html", "")))
            seen.append("".join(out.get("text", "")))
    text = "\n".join(seen)
    return SCIENTIFIC.sub(" ", text)


def backed(token: str, shown: str, shown_nc: str) -> bool:
    """Is `token` printed verbatim, or as a value that rounds to it as written?"""
    bare = token.replace(",", "")
    if token in shown or bare in shown_nc:
        return True
    try:
        want = float(bare)
    except ValueError:
        return False
    decimals = len(bare.split(".")[1]) if "." in bare else 0
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    for m in NUMBER.finditer(shown_nc):
        try:
            if abs(float(m.group()) - want) <= tol:
                return True
        except ValueError:
            continue
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--notebook", default="revision_reported.ipynb")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    path = Path(load_config()["repo"]) / "revision" / "notebooks" / args.notebook
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    nb = json.loads(path.read_text())
    shown = rendered(nb)
    shown_nc = shown.replace(",", "")

    hits = []
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "markdown":
            continue
        # Quoted reviewer comments carry the reviewer's numbers, not ours.
        body = re.sub(r"^>.*$", "", "".join(cell["source"]), flags=re.M)
        for token in dict.fromkeys(NUMBER.findall(body)):
            if token in BARE or backed(token, shown, shown_nc):
                continue
            line = next((l.strip() for l in body.splitlines() if token in l), "")
            hits.append((i, token, line[:100]))

    if not hits:
        print(f"{args.notebook}: every number in the prose is printed by a cell")
        return 0
    print(f"{len(hits)} number(s) in {args.notebook} that no cell prints:\n")
    for i, token, line in hits:
        print(f"  cell {i}: {token}\n      {line}")
    return 1 if args.strict else 0


if __name__ == "__main__":
    sys.exit(main())
