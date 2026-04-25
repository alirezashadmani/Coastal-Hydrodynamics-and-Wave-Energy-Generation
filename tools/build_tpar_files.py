"""
Regenerate SWAN TPAR boundary files from an ERA5 wave NetCDF.
=============================================================

SWAN's BOUNDSPEC VARIABLE FILE expects a TPAR file of the form::

    TPAR
    YYYYMMDD.HHMMSS  Hs  Tp  Dir  Spread
    ...

This script samples ERA5 wave variables (``swh``, ``pp1d``, ``mwd``) at
points along the boundary of a SWAN grid and writes one TPAR file per
point. Use it to (re)generate the boundary files for either the
California Bay or Sea of Oman region, or to extend the boundary with
additional sigma positions.

Usage
-----

    # Regenerate the active Oman boundary TPARs
    python tools/build_tpar_files.py --region oman

    # Preview without writing anything
    python tools/build_tpar_files.py --region oman --dry-run

    # Overwrite empty placeholder TPARs without touching populated ones
    python tools/build_tpar_files.py --region oman --only-empty
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import datetime, timedelta
from typing import Dict, Iterable, List, Tuple

import numpy as np

# Repo imports
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), 'code'))
from regions import get_region, list_regions  # noqa: E402


# ------------------------------------------------------------------
# TPAR I/O
# ------------------------------------------------------------------

def write_tpar(path: str,
               times: Iterable[datetime],
               hs: Iterable[float],
               tp: Iterable[float],
               direction: Iterable[float],
               spread: Iterable[float] = None) -> None:
    """Write a SWAN TPAR file. Default spread is 25° (reasonable swell value)."""
    times = list(times)
    hs = np.asarray(list(hs))
    tp = np.asarray(list(tp))
    direction = np.asarray(list(direction))
    if spread is None:
        spread = np.full_like(hs, 25.0, dtype=float)
    else:
        spread = np.asarray(list(spread))

    with open(path, 'w') as f:
        f.write('TPAR\n')
        for t, h, p, d, s in zip(times, hs, tp, direction, spread):
            f.write(
                f'{t:%Y%m%d.%H%M%S} '
                f'{float(h):.2f} {float(p):.2f} {float(d):.2f} {float(s):.2f}\n'
            )


# ------------------------------------------------------------------
# Boundary-point generator
# ------------------------------------------------------------------

_SIDE_OFFSETS: Dict[str, Tuple[float, float]] = {
    # (dx, dy) per unit of "sigma" along the side (deg, relative to grid origin)
    # For an unrotated spherical grid:
    #   W (west edge) goes north   → dx=0, dy=+1
    #   E (east edge) goes north   → dx=0, dy=+1
    #   S (south edge) goes east   → dx=+1, dy=0
    #   N (north edge) goes east   → dx=+1, dy=0
    'W': (0.0, 1.0),
    'E': (0.0, 1.0),
    'S': (1.0, 0.0),
    'N': (1.0, 0.0),
}


def side_starting_corner(side: str, cfg) -> Tuple[float, float]:
    """Return the (lon, lat) corner where 'sigma' is measured from, per side."""
    lon0, lat0 = cfg.grid_origin
    xlen, ylen = cfg.grid_extent_deg
    if side == 'W':
        return (lon0, lat0)
    if side == 'E':
        return (lon0 + xlen, lat0)
    if side == 'S':
        return (lon0, lat0)
    if side == 'N':
        return (lon0, lat0 + ylen)
    raise ValueError(f'Unknown side {side!r}')


def parse_tpar_name(name: str) -> Tuple[str, int]:
    """Parse ``B<Side><Index>.TPAR`` into (side, index). Raise on bad names."""
    m = re.fullmatch(r'B([WESN])(\d+)\.TPAR', name, re.IGNORECASE)
    if not m:
        raise ValueError(f'Cannot parse TPAR file name: {name}')
    return m.group(1).upper(), int(m.group(2))


# The "sigma" (distance in degrees along the side, from the starting corner)
# used inside the INPUT files is encoded positionally, not in the file name.
# We adopt the convention used in the existing Oman/California TPARs:
#   sigma(B<Side>N) = N * 0.5  for N >= 1,  with B<Side>1 at sigma=0.
# This matches Oman's model.swn where BW2→1.0, BW3→1.5, …, BE1→0.0.
def sigma_for_index(idx: int) -> float:
    return 0.0 if idx <= 1 else (idx - 1) * 0.5


def point_for_tpar(name: str, cfg) -> Tuple[float, float]:
    side, idx = parse_tpar_name(name)
    lon0, lat0 = side_starting_corner(side, cfg)
    dx, dy = _SIDE_OFFSETS[side]
    sigma = sigma_for_index(idx)
    return (lon0 + sigma * dx, lat0 + sigma * dy)


# ------------------------------------------------------------------
# ERA5 sampler
# ------------------------------------------------------------------

def sample_era5(nc_path: str,
                lon: float, lat: float) -> Tuple[List[datetime], np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, Hs, Tp, Dir) at the nearest ERA5 grid cell."""
    import netCDF4 as nc

    ds = nc.Dataset(nc_path)
    lon_name = next(k for k in ds.variables if k.lower() in ('longitude', 'lon'))
    lat_name = next(k for k in ds.variables if k.lower() in ('latitude', 'lat'))
    time_name = next(k for k in ds.variables if k.lower() in ('time', 'valid_time'))

    lons = ds.variables[lon_name][:]
    lats = ds.variables[lat_name][:]
    i_lon = int(np.argmin(np.abs(lons - lon)))
    i_lat = int(np.argmin(np.abs(lats - lat)))

    t_var = ds.variables[time_name]
    times = nc.num2date(
        t_var[:], t_var.units,
        only_use_cftime_datetimes=False,
        only_use_python_datetimes=True,
    ).tolist()

    hs = np.asarray(ds.variables['swh'][:, i_lat, i_lon]).astype(float)
    tp = np.asarray(ds.variables['pp1d'][:, i_lat, i_lon]).astype(float)
    dr = np.asarray(ds.variables['mwd'][:, i_lat, i_lon]).astype(float)
    ds.close()

    # Replace masked / NaN with benign numbers (SWAN needs finite entries)
    hs = np.where(np.isfinite(hs), hs, 0.1)
    tp = np.where(np.isfinite(tp) & (tp > 0), tp, 5.0)
    dr = np.where(np.isfinite(dr), dr, 0.0)
    return times, hs, tp, dr


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(argv: List[str] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--region', default='oman', choices=list_regions(),
                   help='Region to regenerate TPARs for.')
    p.add_argument('--only-empty', action='store_true',
                   help='Only regenerate TPAR files that are < 50 bytes.')
    p.add_argument('--dry-run', action='store_true',
                   help='Print actions without writing any file.')
    p.add_argument('--spread', type=float, default=25.0,
                   help='Directional spread in degrees (default 25).')
    args = p.parse_args(argv)

    cfg = get_region(args.region)
    if not cfg.active_tpars:
        print(f'Region {cfg.key!r} has no active TPAR files to regenerate.')
        return 0

    if not os.path.exists(cfg.wave_nc_path):
        print(f'ERROR: ERA5 wave NetCDF not found: {cfg.wave_nc_path}',
              file=sys.stderr)
        return 2

    print(f'Regenerating TPARs for {cfg.display_name}')
    print(f'  source NetCDF : {cfg.wave_nc_path}')
    print(f'  target dir    : {cfg.model_dir}')
    print(f'  spread        : {args.spread}°')
    print(f'  dry_run       : {args.dry_run}')
    print(f'  only_empty    : {args.only_empty}')
    print()

    written, skipped = 0, 0
    for name in cfg.active_tpars:
        out_path = os.path.join(cfg.model_dir, name)
        if args.only_empty and os.path.exists(out_path) \
                and os.path.getsize(out_path) >= 50:
            skipped += 1
            continue

        lon, lat = point_for_tpar(name, cfg)
        times, hs, tp, dr = sample_era5(cfg.wave_nc_path, lon, lat)
        spread = np.full_like(hs, args.spread, dtype=float)

        print(f'  {name:<12s} lon={lon:7.3f} lat={lat:7.3f} '
              f'Hs={hs.mean():.2f}m Tp={tp.mean():.2f}s n_rows={len(times)}')
        if not args.dry_run:
            write_tpar(out_path, times, hs, tp, dr, spread)
            written += 1

    print()
    print(f'Done. Wrote {written}, skipped {skipped}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
