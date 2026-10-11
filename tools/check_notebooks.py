"""
Static checks for the notebooks in this repository.
===================================================

Run from anywhere::

    python tools/check_notebooks.py

Checks, for every notebooks/**/*.ipynb:

1. Every code cell parses as Python (IPython %/! lines are ignored).
2. Any notebook that imports from code/ (regions / swan_runner) contains
   the canonical bootstrap block from tools/_nb_utils.py verbatim, so path
   handling cannot drift between notebooks.
3. No "except NameError" guards: notebooks must use explicit flags, since a
   NameError guard silently reuses stale variables from an earlier run.

Exit status is 0 when all checks pass, 1 otherwise.
"""

from __future__ import annotations

import ast
import glob
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from _nb_utils import BOOTSTRAP_BLOCK, cell_text  # noqa: E402

REPO = os.path.dirname(_HERE)
_MAGIC = re.compile(r'(?m)^\s*[%!].*$')
_USES_CODE = re.compile(r'^\s*(from\s+(regions|swan_runner)\s+import|import\s+swan_runner)',
                        re.MULTILINE)


def check(path: str) -> list:
    with open(path) as f:
        nb = json.load(f)
    rel = os.path.relpath(path, REPO)
    problems, full = [], []
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code':
            continue
        s = cell_text(c)
        full.append(s)
        try:
            ast.parse(_MAGIC.sub('pass', s))
        except SyntaxError as e:
            problems.append(f'{rel} cell {i}: SyntaxError line {e.lineno}: {e.msg}')
        if 'except NameError' in s:
            problems.append(f'{rel} cell {i}: uses an "except NameError" guard')
    text = '\n'.join(full)
    if _USES_CODE.search(text) and BOOTSTRAP_BLOCK not in text:
        problems.append(f'{rel}: imports from code/ without the canonical bootstrap block')
    return problems


def main() -> int:
    paths = sorted(glob.glob(os.path.join(REPO, 'notebooks', '**', '*.ipynb'),
                             recursive=True))
    problems = [p for path in paths for p in check(path)]
    for p in problems:
        print('FAIL', p)
    print(f'{len(paths)} notebooks checked, {len(problems)} problem(s).')
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
