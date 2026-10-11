"""
SWAN Model Runner — Live-coupling interface
============================================
Provides functions to:
  1. Modify SWAN input files to add custom output
  2. Invoke the SWAN executable via subprocess (safe, no shell=True)
  3. Parse SWAN TABLE and BLOCK (.MAT) output files

Compatible with Python >= 3.9.
Dependencies: numpy, scipy (for .MAT), subprocess
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ------------------------------------------------------------------
# Input-file utilities
# ------------------------------------------------------------------

def prepare_input(template_path: str, output_path: str,
                  extra_commands: Sequence[str]) -> str:
    """
    Copy a SWAN input template and insert extra commands before COMPUTE.

    Parameters
    ----------
    template_path : str
        Path to the original INPUT file.
    output_path : str
        Path for the modified copy.
    extra_commands : Sequence[str]
        SWAN commands to insert above the COMPUTE block.

    Returns
    -------
    str
        The ``output_path`` that was written.
    """
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"SWAN template not found: {template_path}")

    with open(template_path) as f:
        lines = f.readlines()

    # Find the COMPUTE line (case-insensitive, first match)
    idx = next(
        (i for i, l in enumerate(lines) if l.strip().upper().startswith('COMP')),
        None,
    )
    if idx is None:
        raise ValueError(f"COMPUTE not found in SWAN input file: {template_path}")

    insert = ['!\n', '!--- output added by swan_runner.py ---\n']
    for cmd in extra_commands:
        insert.append(cmd.rstrip('\n') + '\n')
    insert.append('!\n')

    lines = lines[:idx] + insert + lines[idx:]
    with open(output_path, 'w') as f:
        f.writelines(lines)
    return output_path


def write_points_file(filepath: str,
                      locations: Iterable[Tuple[float, float]]) -> str:
    """Write a SWAN-format points file (one ``lon lat`` per line)."""
    locations = list(locations)
    if not locations:
        raise ValueError("At least one (lon, lat) pair is required.")
    with open(filepath, 'w') as f:
        for lon, lat in locations:
            f.write(f'{float(lon):.6f} {float(lat):.6f}\n')
    return filepath


# ------------------------------------------------------------------
# Execution
# ------------------------------------------------------------------

def run_swan(model_dir: str,
             input_name: str = 'INPUT',
             swan_cmd: str = 'swanrun',
             nproc: int = 4,
             timeout: int = 7200,
             extra_args: Optional[Sequence[str]] = None
             ) -> subprocess.CompletedProcess:
    """
    Execute SWAN in ``model_dir`` without invoking a shell.

    Parameters
    ----------
    model_dir : str
        Working directory containing all model files.
    input_name : str, default 'INPUT'
        SWAN input filename (with or without a ``.swn`` extension).
    swan_cmd : str, default 'swanrun'
        Command to invoke SWAN (``swanrun``, ``swan.exe``, a full path, …).
    nproc : int, default 4
        Number of OpenMP threads. Must be a positive integer ≤ 256.
    timeout : int, default 7200
        Max seconds before the process is killed.
    extra_args : Sequence[str], optional
        Additional CLI arguments forwarded to ``swan_cmd`` verbatim.

    Returns
    -------
    subprocess.CompletedProcess

    Raises
    ------
    ValueError
        If ``nproc`` is invalid or ``model_dir`` does not exist.
    FileNotFoundError
        If the SWAN executable cannot be located on PATH.
    RuntimeError
        If SWAN does not write a fresh ``norm_end`` file. Any ``norm_end``
        or ``<input>.erf`` left over from a previous run is deleted first,
        so a stale success marker can never mask a failed run.
    """
    if not os.path.isdir(model_dir):
        raise ValueError(f"model_dir does not exist: {model_dir}")
    if not isinstance(nproc, int) or nproc < 1 or nproc > 256:
        raise ValueError(f"nproc must be an int in [1, 256]; got {nproc!r}")
    if shutil.which(swan_cmd) is None:
        raise FileNotFoundError(
            f"SWAN command {swan_cmd!r} not found on PATH. "
            "Install SWAN (https://swanmodel.sourceforge.io) or set swan_cmd "
            "to a full executable path."
        )

    # swanrun expects <input_name>.swn — create it if only the bare file exists
    base = input_name[:-4] if input_name.lower().endswith('.swn') else input_name
    swn_file = os.path.join(model_dir, f'{base}.swn')
    bare_file = os.path.join(model_dir, base)
    if not os.path.exists(swn_file) and os.path.exists(bare_file):
        shutil.copy2(bare_file, swn_file)
    if not os.path.exists(swn_file):
        raise FileNotFoundError(
            f"SWAN input not found in {model_dir}: "
            f"looked for {base!r} and {base + '.swn'!r}"
        )

    argv: List[str] = [swan_cmd, '-input', base]
    if nproc > 1:
        argv += ['-omp', str(nproc)]
    if extra_args:
        argv += [str(a) for a in extra_args]

    # Remove markers from any previous run so they cannot be mistaken for
    # the result of this one.
    norm_end = os.path.join(model_dir, 'norm_end')
    erf_file = os.path.join(model_dir, f'{base}.erf')
    for stale in (norm_end, erf_file):
        if os.path.exists(stale):
            os.remove(stale)

    print(f"    $ {' '.join(shlex.quote(a) for a in argv)}   (in {model_dir})")
    # shell=False: safe w.r.t. shell metacharacters
    result = subprocess.run(
        argv, cwd=model_dir,
        capture_output=True, text=True, timeout=timeout,
        check=False,
    )

    erf_text = ''
    if os.path.exists(erf_file) and os.path.getsize(erf_file) > 0:
        with open(erf_file, errors='replace') as f:
            erf_text = f.read().strip()

    if not os.path.exists(norm_end):
        tail = ((result.stderr or '') + '\n' + (result.stdout or '')
                ).strip().splitlines()[-30:]
        msg = (f"SWAN did not finish normally (rc={result.returncode}, "
               f"no norm_end written) in {model_dir}.\n" + "\n".join(tail))
        if erf_text:
            msg += f"\n--- {os.path.basename(erf_file)} ---\n{erf_text[:2000]}"
        msg += f"\nInspect {base}.prt for details."
        raise RuntimeError(msg)

    with open(norm_end) as f:
        print(f"    SWAN: {f.read().strip()}")
    if result.returncode != 0:
        print(f"    WARNING: {swan_cmd} exited with rc={result.returncode} "
              "although norm_end was written.")
    if erf_text:
        print(f"    WARNING: SWAN wrote {os.path.basename(erf_file)}:")
        for line in erf_text.splitlines()[:10]:
            print(f"      {line}")

    return result


def check_swan_available(swan_cmd: str = 'swanrun') -> bool:
    """Return True if ``swan_cmd`` resolves to an executable on PATH."""
    return shutil.which(swan_cmd) is not None


# ------------------------------------------------------------------
# Output readers
# ------------------------------------------------------------------

_SKIP_HEADER_PREFIXES = ('Run:', 'Table:', 'SWAN', 'Project', 'Variables')


def read_table(filepath: str) -> dict:
    """
    Parse a SWAN TABLE output file.

    The parser tolerates multiple SWAN versions by:
      * Preferring a header line that starts with ``Variables:`` if present
        (emitted by ``HEAD`` or ``HEADer`` in SWAN output).
      * Otherwise falling back to the last header line that contains no
        units ('[' character) and does not start with a known metadata prefix.

    Parameters
    ----------
    filepath : str
        Path to a SWAN TABLE text file.

    Returns
    -------
    dict
        ``{variable_name: np.ndarray}`` for each column.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"SWAN TABLE not found: {filepath}")

    headers: List[str] = []
    data_lines: List[str] = []
    with open(filepath) as f:
        for line in f:
            s = line.strip()
            if s.startswith('%'):
                headers.append(s)
            elif s:
                data_lines.append(s)

    if not data_lines:
        raise ValueError(f"SWAN TABLE has no data rows: {filepath}")

    # --- Find variable names --------------------------------------------
    var_names: Optional[List[str]] = None

    # Preferred: explicit "% Variables: ..." line
    for h in headers:
        tail = h.lstrip('%').strip()
        if tail.lower().startswith('variables:'):
            var_names = tail.split(':', 1)[1].split()
            break

    # Fallback: the last non-metadata, non-units header line
    if var_names is None:
        for h in reversed(headers):
            tokens = h.lstrip('%').split()
            if not tokens or '[' in h:
                continue
            if any(tok.startswith(_SKIP_HEADER_PREFIXES) for tok in tokens):
                continue
            var_names = tokens
            break
    if var_names is None:
        raise ValueError(
            f"Could not identify variable names in SWAN TABLE header: {filepath}"
        )

    data = np.loadtxt(data_lines)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < len(var_names):
        raise ValueError(
            f"SWAN TABLE column count ({data.shape[1]}) is smaller than "
            f"the number of variable names ({len(var_names)}) in {filepath}"
        )
    # Silently drop any trailing name that has no matching column
    var_names = var_names[: data.shape[1]]
    return {v: data[:, i] for i, v in enumerate(var_names)}


def read_mat_block(filepath: str) -> dict:
    """
    Read a SWAN .MAT BLOCK output file via ``scipy.io.loadmat``.

    Returns
    -------
    dict
        Raw mapping from variable names (e.g. ``'Hsig_20240101_000000'``)
        to ``numpy.ndarray``. Non-data keys (``__header__`` etc.) are kept
        for informational purposes.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file cannot be loaded as a .MAT.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"SWAN BLOCK file not found: {filepath}")
    try:
        import scipy.io
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "scipy is required to read SWAN BLOCK .MAT output. "
            "Install with: pip install scipy"
        ) from exc
    try:
        return scipy.io.loadmat(filepath)
    except Exception as exc:
        raise ValueError(f"Failed to load SWAN BLOCK file {filepath}: {exc}") from exc


def extract_block_variable(mat_data: dict, prefix: str) -> Optional[np.ndarray]:
    """
    Collect all timestep arrays for a variable prefix (e.g. ``'Hsig'``)
    and return a 3-D array of shape ``(time, ny, nx)``.

    All arrays must share the same 2-D shape; if they do not, a
    ``ValueError`` is raised rather than producing a silently wrong stack.
    Returns ``None`` when no matching key is present.
    """
    pattern = re.compile(rf'^{re.escape(prefix)}_\d{{8}}_\d{{6}}$')
    keys = sorted(k for k in mat_data if isinstance(k, str) and pattern.match(k))
    if not keys:
        return None

    first_shape = np.asarray(mat_data[keys[0]]).shape
    frames = []
    for k in keys:
        arr = np.asarray(mat_data[k])
        if arr.shape != first_shape:
            raise ValueError(
                f"SWAN BLOCK frame {k!r} has shape {arr.shape}, "
                f"expected {first_shape} (from {keys[0]!r}). "
                "The SWAN run likely terminated mid-way — inspect the .prt log."
            )
        frames.append(arr)
    return np.stack(frames, axis=0)


def list_block_variables(mat_data: dict) -> List[str]:
    """Return the sorted list of *unique* variable prefixes in a .MAT file."""
    prefixes = set()
    for k in mat_data:
        if not isinstance(k, str):
            continue
        m = re.match(r'^([A-Za-z][A-Za-z0-9]*)_\d{8}_\d{6}$', k)
        if m:
            prefixes.add(m.group(1))
    return sorted(prefixes)
