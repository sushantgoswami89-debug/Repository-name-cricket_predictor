"""Repo-wide staleness check: does every hardcoded file-path string
literal in every tracked .py file actually resolve to something on disk?

Built 2026-08-23 after a full manual audit, in response to a real
concern: a one-time manual check doesn't protect against a FUTURE
refactor silently breaking a path reference. This makes the check
rerunnable instead of relying on someone remembering to look. Run it any
time -- after a promotion, a refactor, a big rename -- to catch drift
before it becomes a repeat of this session's real bugs (venue never
wired into MatchContext, batting/bowling teams swapped during every
chase): both of those were "reference doesn't reach where it should,"
not something a plain existence-check like this one would have caught
directly, but a codebase where paths silently rot is a codebase where
that kind of bug is more likely to hide. This tool catches the cheaper,
more mechanical half of that risk -- broken/dead literal paths -- so a
human only has to spend attention on the harder, semantic half.

Flags two tiers, since they carry very different risk:
  - HIGH RISK: the unresolved literal is inside a file that's part of
    the live serving path (imported, transitively, by app/ml/prediction_engine.py,
    app/ml/match_winner_engine.py, or app/live/pipeline.py) -- a broken
    reference here could mean live predictions are silently degraded or
    the engine fails to start.
  - LOW RISK: everything else -- almost always a standalone research/
    diagnostic script (train_*.py, diagnose_*.py, evaluate_*.py, ...)
    whose missing dependency is a gitignored, regenerable model artifact
    (models/candidates/**/*.pkl etc. are gitignored by design -- see
    .gitignore). These fail loudly (FileNotFoundError) if actually run
    without first regenerating the upstream artifact -- not a silent
    correctness risk, just an inconvenience.

Not exhaustive by construction (regex-based literal matching, no
f-string interpolation resolution) -- a best-effort net, not a proof.
"""
from __future__ import annotations

import ast
import re
import subprocess
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
ROOT = BACKEND.parent

LIVE_ENTRYPOINTS = {
    "app/ml/prediction_engine.py",
    "app/ml/match_winner_engine.py",
    "app/live/pipeline.py",
}

PATH_LIKE = re.compile(r"^[\w./-]+\.(json|csv|pkl|cbm|txt|md|log|pid|yaml|yml)$")
SKIP_PREFIXES = ("http://", "https://", "bot", "{", "%")


def tracked_files() -> list[str]:
    return subprocess.run(
        ["git", "-C", str(BACKEND), "ls-files", "*.py"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()


def local_module_names(files: list[str]) -> set[str]:
    return {Path(f).stem for f in files if "/" not in f} | {
        f.split("/")[0] for f in files if "/" in f
    }


def imports_of(rel: str) -> set[str]:
    path = BACKEND / rel
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), rel)
    except SyntaxError:
        return set()
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module.split(".")[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
    return found


def reaches_live_entrypoint(rel: str, files: list[str], local_mods: set[str]) -> bool:
    """BFS over the local import graph: does anything reachable from `rel`
    end up inside one of the three live-serving files?"""
    if rel in LIVE_ENTRYPOINTS:
        return True
    by_module: dict[str, str] = {}
    for f in files:
        stem = Path(f).stem
        by_module[stem] = f
        if "/" in f:
            by_module[f.split("/")[0]] = f

    seen = {rel}
    frontier = [rel]
    while frontier:
        current = frontier.pop()
        for mod in imports_of(current):
            if mod not in local_mods:
                continue
            target = by_module.get(mod)
            if not target or target in seen:
                continue
            if target in LIVE_ENTRYPOINTS:
                return True
            seen.add(target)
            frontier.append(target)
    return False


def string_literals(rel: str) -> list[str]:
    path = BACKEND / rel
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), rel)
    except SyntaxError:
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            s = node.value
            if s and not s.startswith(SKIP_PREFIXES) and PATH_LIKE.match(s) and "/" in s:
                out.append(s)
    return out


def resolves(literal: str) -> bool:
    if (BACKEND / literal).exists() or (ROOT / literal).exists():
        return True
    if literal.startswith("../") and (ROOT / literal[3:]).exists():
        return True
    return False


def main() -> int:
    files = tracked_files()
    local_mods = local_module_names(files)

    candidates: dict[str, list[str]] = {}
    for rel in files:
        for literal in string_literals(rel):
            candidates.setdefault(literal, []).append(rel)

    unresolved = {lit: refs for lit, refs in candidates.items() if not resolves(lit)}

    high_risk: list[tuple[str, str]] = []
    low_risk: list[tuple[str, str]] = []
    for literal, refs in sorted(unresolved.items()):
        for rel in refs:
            row = (literal, rel)
            if reaches_live_entrypoint(rel, files, local_mods):
                high_risk.append(row)
            else:
                low_risk.append(row)

    print(f"Scanned {len(candidates)} distinct path-like literals across {len(files)} tracked files.")
    print(f"Unresolved: {len(unresolved)} literal(s), {len(high_risk) + len(low_risk)} reference(s).\n")

    if high_risk:
        print(f"=== HIGH RISK: {len(high_risk)} unresolved reference(s) reachable from live serving ===")
        for literal, rel in high_risk:
            print(f"  '{literal}'  <- {rel}")
        print()
    if low_risk:
        print(f"=== low risk: {len(low_risk)} unresolved reference(s) in standalone/research scripts ===")
        for literal, rel in low_risk:
            print(f"  '{literal}'  <- {rel}")

    if not unresolved:
        print("Nothing unresolved.")

    return 1 if high_risk else 0


if __name__ == "__main__":
    sys.exit(main())
