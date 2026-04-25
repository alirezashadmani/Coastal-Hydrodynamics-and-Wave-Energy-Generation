"""
Region configuration for the Wiley textbook companion
=====================================================

Every notebook and script in this repository that touches SWAN model files,
bathymetry, or forcing data reads its paths and parameters from here. This is
what lets the whole pipeline run on either **California Bay** or the
**Sea of Oman** without any other change to the notebooks.

Usage
-----

    from regions import get_region, list_regions, REPO_ROOT

    cfg = get_region('oman')          # or 'california_bay'
    print(cfg.model_dir)
    print(cfg.domain_bounds)

To switch a notebook, set the ``REGION`` variable in its first code cell
(or the ``SWAN_REGION`` environment variable) to one of the names returned
by :func:`list_regions`.

Every attribute is documented in :class:`RegionConfig`. New regions can be
added by appending to the ``_REGISTRY`` dict; the schema is enforced by the
dataclass so missing fields surface immediately.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple


# ------------------------------------------------------------------
# Repository-root discovery
# ------------------------------------------------------------------

def find_repo_root(start: Optional[str] = None, max_up: int = 10) -> str:
    """
    Walk upward from ``start`` (default: this file's directory) looking for
    the repository root. Markers: ``code/`` directory + ``README.md``.
    Falls back to three levels up if no marker is found.
    """
    here = os.path.abspath(start or os.path.dirname(__file__))
    d = here
    for _ in range(max_up):
        if (
            os.path.isdir(os.path.join(d, 'code'))
            and os.path.isfile(os.path.join(d, 'README.md'))
        ):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(here, '..'))


REPO_ROOT: str = find_repo_root()


# ------------------------------------------------------------------
# Config dataclass
# ------------------------------------------------------------------

@dataclass(frozen=True)
class RegionConfig:
    """Per-region parameters for the SWAN workflow."""

    # Human-readable metadata
    key: str
    display_name: str
    description: str

    # Paths relative to REPO_ROOT (absolute paths returned by the resolvers)
    model_subdir: str
    swan_input_name: str          # e.g. 'INPUT' or 'model'
    bathy_file: str               # GEBCO ASCII file name inside model_subdir
    wave_nc: str                  # ERA5 wave NetCDF
    wind_nc: str                  # ERA5 wind NetCDF
    wind_inp: str = 'wind.inp'

    # Optional nested-grid variant
    nested_subdir: Optional[str] = None
    nested_coarse_input: Optional[str] = None
    nested_fine_input: Optional[str] = None

    # Domain (lon_min, lon_max, lat_min, lat_max) — degrees East / North
    domain_bounds: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # Rotation (deg) of the SWAN computational grid, 0 = aligned with E/N
    grid_rotation_deg: float = 0.0
    # Grid origin (lon, lat) in degrees — bottom-left of CGRID
    grid_origin: Tuple[float, float] = (0.0, 0.0)
    # Grid extent in the rotated frame (degrees): xlen, ylen
    grid_extent_deg: Tuple[float, float] = (0.0, 0.0)
    # Number of cells (mxc, myc)
    grid_nc: Tuple[int, int] = (0, 0)

    # Simulation window (UTC)
    t_start: datetime = datetime(2000, 1, 1)
    t_end: datetime = datetime(2000, 1, 2)
    t_step_hours: float = 1.0

    # Candidate-site sampling grid for site-selection / GA workflows
    site_grid_lon: Tuple[float, float, float] = (0.0, 0.0, 0.04)  # min, max, step (deg)
    site_grid_lat: Tuple[float, float, float] = (0.0, 0.0, 0.04)

    # TPAR boundary files that are actively referenced in the INPUT.
    # Empty tuple for a single-boundary model.
    active_tpars: Tuple[str, ...] = ()

    # Observation point names (for validation / postprocessing tables)
    obs_points: Tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------------
    # Resolved absolute paths (properties)
    # ------------------------------------------------------------------

    @property
    def model_dir(self) -> str:
        return os.path.join(REPO_ROOT, self.model_subdir)

    @property
    def swan_input_path(self) -> str:
        return os.path.join(self.model_dir, self.swan_input_name)

    @property
    def bathy_path(self) -> str:
        return os.path.join(self.model_dir, self.bathy_file)

    @property
    def wave_nc_path(self) -> str:
        return os.path.join(self.model_dir, self.wave_nc)

    @property
    def wind_nc_path(self) -> str:
        return os.path.join(self.model_dir, self.wind_nc)

    @property
    def wind_inp_path(self) -> str:
        return os.path.join(self.model_dir, self.wind_inp)

    @property
    def nested_dir(self) -> Optional[str]:
        return (
            os.path.join(REPO_ROOT, self.nested_subdir)
            if self.nested_subdir else None
        )

    # ------------------------------------------------------------------
    # Per-notebook output directories (under <repo>/figs/<notebook>/<region>)
    # ------------------------------------------------------------------

    def figs_dir(self, notebook_subdir: Optional[str] = None) -> str:
        """
        Return (and create) the directory where a notebook should write its
        figures / result JSON. Layout::

            <repo>/figs/<notebook_subdir>/<region_key>/

        If ``notebook_subdir`` is None the per-region root under ``figs/``
        is returned.
        """
        base = os.path.join(REPO_ROOT, 'figs')
        if notebook_subdir:
            base = os.path.join(base, notebook_subdir)
        out = os.path.join(base, self.key)
        os.makedirs(out, exist_ok=True)
        return out

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def check_inputs(self, require_nested: bool = False) -> List[str]:
        """
        Return a list of human-readable problems with the on-disk state of
        this region. Empty list means everything is present.
        """
        missing: List[str] = []
        required = [
            ('SWAN input', self.swan_input_path),
            ('Bathymetry', self.bathy_path),
            ('Wave NetCDF', self.wave_nc_path),
            ('Wind NetCDF', self.wind_nc_path),
        ]
        for label, p in required:
            if not os.path.exists(p):
                missing.append(f"{label} missing: {p}")

        # TPARs referenced in the INPUT
        for name in self.active_tpars:
            tpar = os.path.join(self.model_dir, name)
            if not os.path.exists(tpar):
                missing.append(f"TPAR missing: {tpar}")
            elif os.path.getsize(tpar) < 50:
                missing.append(
                    f"TPAR appears empty (< 50 B): {tpar}. "
                    "Regenerate with tools/build_tpar_files.py or check "
                    "the ERA5 wave NetCDF."
                )

        if require_nested and self.nested_subdir:
            for fname in (self.nested_coarse_input, self.nested_fine_input):
                if not fname:
                    continue
                p = os.path.join(self.nested_dir, fname)
                if not os.path.exists(p):
                    missing.append(f"Nested SWAN input missing: {p}")
        return missing


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

_REGISTRY: Dict[str, RegionConfig] = {
    'california_bay': RegionConfig(
        key='california_bay',
        display_name='California Bay',
        description='San Francisco Bay outer coast — deep-water swell, '
                    'rotated SWAN grid, Jan–Feb 2024 ERA5 forcing.',
        model_subdir='california_bay.F',
        swan_input_name='INPUT',
        bathy_file='GEBCO.asc',
        wave_nc='WAVE.nc',
        wind_nc='WIND.nc',
        nested_subdir='california_bay_nested.F',
        nested_coarse_input='INPUT_coarse',
        nested_fine_input='INPUT_fine',
        domain_bounds=(-122.8018, -121.8026, 37.2947, 38.1933),
        grid_rotation_deg=22.467,
        grid_origin=(-122.3018, 37.2947),
        grid_extent_deg=(0.4992, 0.8986),
        grid_nc=(120, 216),
        t_start=datetime(2024, 1, 1, 0, 0),
        t_end=datetime(2024, 2, 29, 23, 0),
        t_step_hours=1.0,
        site_grid_lon=(-122.55, -121.90, 0.04),
        site_grid_lat=(37.35, 38.25, 0.04),
        active_tpars=('BW2.TPAR',),
        obs_points=(),
    ),
    'oman': RegionConfig(
        key='oman',
        display_name='Sea of Oman',
        description='Gulf of Oman / Arabian Sea — wind-sea dominated, '
                    'multi-sided TPAR boundaries (W/S/E), Apr–May 2021.',
        model_subdir='Oman',
        swan_input_name='model.swn',
        bathy_file='gebco.asc',
        wave_nc='Wave-2021-April-May.nc',
        wind_nc='Wind-2021-April-May.nc',
        domain_bounds=(52.2312, 61.5912, 16.4077, 27.7677),
        grid_rotation_deg=0.0,
        grid_origin=(52.2312, 16.4077),
        grid_extent_deg=(9.36, 11.36),
        grid_nc=(468, 568),
        t_start=datetime(2021, 4, 1, 0, 0),
        t_end=datetime(2021, 5, 31, 23, 0),
        t_step_hours=1.0,
        site_grid_lon=(57.0, 59.5, 0.08),
        site_grid_lat=(22.5, 25.8, 0.08),
        active_tpars=tuple(
            [f'BW{i}.TPAR' for i in range(2, 8)]               # BW2..BW7
            + [f'BS{i}.TPAR' for i in range(3, 20)]            # BS3..BS19
            + [f'BE{i}.TPAR' for i in range(1, 18)]            # BE1..BE17
        ),
        obs_points=('Wudam-N', 'Wudam-S'),
    ),
}


def list_regions() -> List[str]:
    """Return the keys of all registered regions."""
    return list(_REGISTRY)


def get_region(key: Optional[str] = None) -> RegionConfig:
    """
    Return the :class:`RegionConfig` for ``key``. If ``key`` is ``None``,
    falls back to the ``SWAN_REGION`` environment variable and finally to
    ``'california_bay'``.
    """
    if key is None:
        key = os.environ.get('SWAN_REGION', 'california_bay')
    key = key.strip().lower().replace(' ', '_').replace('-', '_')
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown region {key!r}. Available: {list(_REGISTRY)}. "
            "Add a new entry to code/regions.py:_REGISTRY to extend."
        )
    return _REGISTRY[key]


def summarize(region: Optional[str] = None) -> str:
    """Return a short human-readable summary of a region."""
    cfg = get_region(region)
    lb, rb, bb, tb = cfg.domain_bounds
    return (
        f"[{cfg.key}] {cfg.display_name}\n"
        f"  {cfg.description}\n"
        f"  model_dir   : {cfg.model_dir}\n"
        f"  SWAN input  : {cfg.swan_input_name}\n"
        f"  domain      : lon [{lb:.3f}, {rb:.3f}]  lat [{bb:.3f}, {tb:.3f}]\n"
        f"  grid        : {cfg.grid_nc[0]}x{cfg.grid_nc[1]} cells, "
        f"rot={cfg.grid_rotation_deg}\u00b0\n"
        f"  window      : {cfg.t_start:%Y-%m-%d} \u2192 {cfg.t_end:%Y-%m-%d}\n"
        f"  active TPAR : {len(cfg.active_tpars)} file(s)\n"
    )
