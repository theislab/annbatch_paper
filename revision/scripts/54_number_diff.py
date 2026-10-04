from __future__ import annotations

import argparse
import collections
import re
import subprocess
import sys
from pathlib import Path

from _common import load_config

#: A number as written; greedy across . and , so 0.2.2 and 94,129,984 stay single tokens.
NUMBER = re.compile(r"\d+(?:[.,]\d+)*")

#: Superscript runs (10⁻⁴, 2²¹) carry values too and contain no ASCII digits.
SUPERSCRIPT = re.compile(r"[⁰¹²³⁴-⁹⁺-ⁿ]+")


def tokens(text: str) -> collections.Counter:
    """Every numeric token in the text, counted."""
    counted: collections.Counter = collections.Counter()
    for line in text.splitlines():
        for pattern in (NUMBER, SUPERSCRIPT):
            for match in pattern.finditer(line):
                counted[match.group()] += 1
    return counted


def contexts(text: str, token: str) -> list[str]:
    """Lines containing `token` as a whole numeric token, trimmed for display."""
    out = []
    for line in text.splitlines():
        for pattern in (NUMBER, SUPERSCRIPT):
            if any(m.group() == token for m in pattern.finditer(line)):
                trimmed = " ".join(line.split())
                out.append(trimmed[:120] + ("..." if len(trimmed) > 120 else ""))
                break
    return out


def baseline_text(rev: str, relpath: str) -> str:
    """The document as of `rev`, read from git rather than from disk."""
    proc = subprocess.run(["git", "show", f"{rev}:{relpath}"],
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"cannot read {relpath} at {rev}: "
                         f"{proc.stderr.strip() or 'unknown git error'}")
    return proc.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", default="REVIEWER_RESPONSES.md",
                    help="document under revision/docs/ (default: the response)")
    ap.add_argument("--baseline", default="HEAD",
                    help="git revision to compare against (default: HEAD)")
    ap.add_argument("--baseline-file",
                    help="compare against a file on disk instead of a git "
                         "revision, for a pass whose input was never committed")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 if any numeric token differs")
    args = ap.parse_args()

    root = Path(load_config()["repo"])
    relpath = f"revision/docs/{args.doc}"
    current_path = root / relpath
    if not current_path.exists():
        raise SystemExit(f"{current_path} does not exist")

    current = current_path.read_text()
    if args.baseline_file:
        source = Path(args.baseline_file)
        if not source.exists():
            raise SystemExit(f"baseline file {source} does not exist")
        before, label = source.read_text(), str(source)
    else:
        before, label = baseline_text(args.baseline, relpath), args.baseline
    was, now = tokens(before), tokens(current)

    if was == now:
        print(f"{args.doc}: {sum(now.values())} numeric tokens "
              f"({len(now)} distinct), identical to {label}")
        return 0

    removed = was - now
    added = now - was
    print(f"{args.doc} differs from {label}: "
          f"{sum(removed.values())} token(s) removed, "
          f"{sum(added.values())} added\n")

    if removed:
        print("no longer present (or quoted fewer times):")
        for token, count in sorted(removed.items()):
            print(f"  {token}  x{count}")
            for line in contexts(before, token)[:2]:
                print(f"      was: {line}")
    if added:
        print("\nnew (or quoted more times):")
        for token, count in sorted(added.items()):
            print(f"  {token}  x{count}")
            for line in contexts(current, token)[:2]:
                print(f"      now: {line}")

    # Same digits reordered means a transposition, not a deliberate cut.
    suspicious = []
    for gone in removed:
        for fresh in added:
            if gone == fresh:
                continue
            if sorted(gone.replace(",", "")) == sorted(fresh.replace(",", "")):
                suspicious.append((gone, fresh))
    if suspicious:
        print("\nsame digits in a different order, i.e. likely a transposition:")
        for gone, fresh in suspicious:
            print(f"  {gone} -> {fresh}")

    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
