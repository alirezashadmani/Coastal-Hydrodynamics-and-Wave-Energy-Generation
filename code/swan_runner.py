"""
SWAN Model Runner — Live-coupling interface
============================================
Provides functions to:
  1. Modify SWAN input files to add custom output
  2. Invoke the SWAN executable via subprocess
  3. Parse SWAN TABLE and BLOCK (.MAT) output files

Dependencies: numpy, scipy (for .MAT), subprocess
"""

import os
import re
import shutil
import subprocess
import numpy as np


def prepare_input(template_path, output_path, extra_commands):
    """
    Copy a SWAN input template and insert extra commands before COMPUTE.

    Parameters
    ----------
    template_path : str  — path to the original INPUT file.
    output_path   : str  — path for the modified copy.
    extra_commands: list[str] — SWAN commands to insert.
    """
    with open(template_path) as f:
        lines = f.readlines()

    # Find the COMPUTE line
    idx = next(
        (i for i, l in enumerate(lines) if l.strip().upper().startswith('COMP')),
        None,
    )
    if idx is None:
        raise ValueError("COMPUTE not found in SWAN input file")

    insert = ['!\n', '!--- output added by swan_runner.py ---\n']
    for cmd in extra_commands:
        insert.append(cmd + '\n')
    insert.append('!\n')

    lines = lines[:idx] + insert + lines[idx:]
    with open(output_path, 'w') as f:
        f.writelines(lines)
    return output_path


def write_points_file(filepath, locations):
    """Write a SWAN-format points file (lon lat per line)."""
    with open(filepath, 'w') as f:
        for lon, lat in locations:
            f.write(f'{lon:.6f} {lat:.6f}\n')


def run_swan(model_dir, input_name='INPUT', swan_cmd='swanrun', nproc=4,
             timeout=7200):
    """
    Execute SWAN in *model_dir*.

    Parameters
    ----------
    model_dir  : working directory containing all model files.
    input_name : SWAN input filename (without extension).
    swan_cmd   : command to invoke SWAN ('swanrun', 'swan.exe', …).
    nproc      : number of OpenMP threads.
    timeout    : max seconds before the process is killed.

    Returns
    -------
    subprocess.CompletedProcess
    """
    cmd = f'{swan_cmd} -input {input_name}'
    if nproc > 1:
        cmd += f' -omp {nproc}'

    print(f"    $ {cmd}   (in {model_dir})")
    result = subprocess.run(
        cmd, shell=True, cwd=model_dir,
        capture_output=True, text=True, timeout=timeout,
    )

    norm_end = os.path.join(model_dir, 'norm_end')
    if os.path.exists(norm_end):
        with open(norm_end) as f:
            print(f"    SWAN: {f.read().strip()}")
    elif result.returncode != 0:
        raise RuntimeError(f"SWAN failed (rc={result.returncode}):\n{result.stderr}")

    return result


# ------------------------------------------------------------------
# Output readers
# ------------------------------------------------------------------

def read_table(filepath):
    """
    Parse a SWAN TABLE output file.

    Returns
    -------
    dict — {variable_name: np.array} for each column.
    """
    headers, data_lines = [], []
    with open(filepath) as f:
        for line in f:
            s = line.strip()
            if s.startswith('%'):
                headers.append(s)
            elif s:
                data_lines.append(s)

    # Variable names are in the header line that does NOT contain '[' (units)
    # and is not blank. Search backwards from the end.
    var_names = None
    skip_prefixes = ('Run:', 'Table:', 'SWAN')
    for h in reversed(headers):
        tokens = h.lstrip('%').split()
        if tokens and '[' not in h:
            if not all(t.startswith(skip_prefixes) for t in tokens):
                var_names = tokens
                break
    if var_names is None:
        raise ValueError("Could not find variable names in SWAN TABLE header")
    data = np.loadtxt(data_lines)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    return {v: data[:, i] for i, v in enumerate(var_names) if i < data.shape[1]}


def read_mat_block(filepath):
    """
    Read a SWAN .MAT BLOCK output file.

    Returns
    -------
    dict from scipy.io.loadmat (keys = variable names like 'Hsig_YYYYMMDD_HHMMSS').
    """
    import scipy.io
    return scipy.io.loadmat(filepath)


def extract_block_variable(mat_data, prefix):
    """
    Collect all timestep arrays for a variable prefix (e.g. 'Hsig')
    and return a 3-D array (time, ny, nx).
    """
    pattern = re.compile(rf'^{prefix}_\d{{8}}_\d{{6}}$')
    keys = sorted(k for k in mat_data if pattern.match(k))
    if not keys:
        return None
    return np.stack([mat_data[k] for k in keys], axis=0)


def check_swan_available(swan_cmd='swanrun'):
    """Return True if SWAN executable is found on PATH."""
    return shutil.which(swan_cmd) is not None
