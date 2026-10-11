"""Utilities for programmatically editing Jupyter notebooks in this repo."""
from __future__ import annotations

import json
from typing import List, Optional


def load_nb(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def save_nb(path: str, nb: dict) -> None:
    with open(path, 'w') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write('\n')


def _as_source(text: str) -> List[str]:
    """Convert a plain string to the notebook ``source`` list-of-lines format."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith('\n'):
        lines[-1] += ''  # keep as-is
    return lines


def code_cell(source: str) -> dict:
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': _as_source(source),
    }


def md_cell(source: str) -> dict:
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': _as_source(source),
    }


def cell_text(cell: dict) -> str:
    s = cell.get('source', '')
    return s if isinstance(s, str) else ''.join(s)


def find_cell(nb: dict, needle: str, cell_type: Optional[str] = None) -> int:
    """Return the index of the first cell whose source contains ``needle``."""
    for i, c in enumerate(nb['cells']):
        if cell_type and c['cell_type'] != cell_type:
            continue
        if needle in cell_text(c):
            return i
    return -1


def replace_cell(nb: dict, index: int, new: dict) -> None:
    nb['cells'][index] = new


def insert_cell(nb: dict, index: int, new: dict) -> None:
    nb['cells'].insert(index, new)


def clear_outputs(nb: dict) -> None:
    for c in nb['cells']:
        if c['cell_type'] == 'code':
            c['outputs'] = []
            c['execution_count'] = None


# ---------------------------------------------------------------------
# The canonical "region selection" cell used across notebooks
# ---------------------------------------------------------------------

REGION_IMPORT_BLOCK = '''\
# =====================================================================
# REGION SELECTION — pick the SWAN model to drive this notebook
# =====================================================================
# Set REGION to one of: {regions}. The whole notebook is parameterised
# through the central config in ``code/regions.py``, so no path or
# time-range strings need to be edited below.
import os, sys

_search = os.path.abspath(os.getcwd())
for _ in range(10):
    if os.path.isdir(os.path.join(_search, 'code')):
        REPO_ROOT = _search
        break
    _search = os.path.dirname(_search)
else:
    raise FileNotFoundError(
        'Could not locate repository root (no code/ directory found). '
        'Run the notebook from inside the Coastal-Hydrodynamics repo.'
    )
_code_dir = os.path.join(REPO_ROOT, 'code')
if _code_dir not in sys.path:
    sys.path.insert(0, _code_dir)

from regions import get_region, list_regions, summarize   # noqa: E402

REGION = os.environ.get('SWAN_REGION', '{default_region}')
cfg = get_region(REGION)
print(summarize(REGION))

missing = cfg.check_inputs({require_nested})
if missing:
    print('WARNING — some required inputs are missing on disk:')
    for _m in missing:
        print('  -', _m)
'''
