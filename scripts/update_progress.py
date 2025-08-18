#!/usr/bin/env python
"""Auto-update progress checklist in thesis.md.

Scans thesis.md for the 'Detailed Task List (Live Checklist)' section and updates
checkboxes based on simple heuristics or external signals (placeholder logic now).

Current heuristics (simple pattern examples):
- If semantic manager file exists -> mark 'Add semantic config & algorithm variant'.
- If 'semantic_cfgs' appears in PPOLag.yaml -> consider config added.
- If 'reward shaping' function call present in semantic manager -> mark shaping done.
- If 'risk head' class file exists -> mark risk head done.
(Extend with real metrics/log parsing as needed.)

Usage:
  python scripts/update_progress.py [--dry-run]

Writes updated thesis.md in place (unless dry-run).
"""
from __future__ import annotations
import argparse
import pathlib
import re
from typing import Dict, List

ROOT = pathlib.Path(__file__).resolve().parents[1]
THESIS = ROOT / 'thesis.md'

# Task identifiers mapped to heuristic functions.
# Each function returns True (done), False (leave), or None (no decision).

def exists(path: str) -> bool:
    return (ROOT / path).exists()

def file_contains(path: str, pattern: str) -> bool:
    fp = ROOT / path
    if not fp.exists():
        return False
    try:
        text = fp.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        return False
    return re.search(pattern, text) is not None

Heuristics = {
    'Add semantic config & algorithm variant': lambda: exists('omnisafe/algorithms/on_policy/naive_lagrange/ppo_lag_semantic.py'),
    'Reward shaping with annealed coefficient': lambda: file_contains('omnisafe/common/semantics/semantic_manager.py', r'beta') and file_contains('omnisafe/common/semantics/semantic_manager.py', r'margin'),
    'Risk head auxiliary loss': lambda: exists('omnisafe/common/semantics/risk_head.py'),
}

CHECKBOX_SECTION_HEADER = '### Core Integrity'

CHECKBOX_PATTERN = re.compile(r'^- \[(?P<mark>[ x~])\] (?P<task>.+)$')


def update_checklist(lines: List[str]) -> List[str]:
    """Update checklist marks based on heuristics."""
    in_section = False
    updated: List[str] = []
    for line in lines:
        stripped = line.rstrip('\n')
        if stripped.startswith(CHECKBOX_SECTION_HEADER):
            in_section = True
            updated.append(line)
            continue
        if in_section and stripped.startswith('### ') and not stripped.startswith(CHECKBOX_SECTION_HEADER):
            # End of this subsection
            in_section = False
        if in_section:
            m = CHECKBOX_PATTERN.match(stripped)
            if m:
                task = m.group('task').strip()
                if task in Heuristics:
                    done = Heuristics[task]()
                    if done:
                        # Replace mark with 'x'
                        line = re.sub(r'^- \[[ x~]\]', '- [x]', line)
                updated.append(line)
            else:
                updated.append(line)
        else:
            updated.append(line)
    return updated


def main(dry_run: bool = False) -> None:
    if not THESIS.exists():
        raise SystemExit(f"thesis.md not found at {THESIS}")
    content = THESIS.read_text(encoding='utf-8').splitlines(keepends=True)
    new_content = update_checklist(content)
    if dry_run:
        diff_lines = []
        for old, new in zip(content, new_content):
            if old != new:
                diff_lines.append(f"- {old}+ {new}")
        print('# Dry Run Diff (lines changed)')
        print(''.join(diff_lines))
    else:
        THESIS.write_text(''.join(new_content), encoding='utf-8')
        print('thesis.md checklist updated.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='Show changes without writing file')
    args = parser.parse_args()
    main(dry_run=args.dry_run)
