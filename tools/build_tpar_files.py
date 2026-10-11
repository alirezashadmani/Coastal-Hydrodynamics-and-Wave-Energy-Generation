"""
Regenerate SWAN TPAR boundary files from an ERA5 wave NetCDF.
=============================================================

SWAN's BOUNDSPEC VARIABLE FILE expects a TPAR file of the form::

    TPAR
    YYYYMMDD.HHMMSS  Hs  Tp  Dir  Spread
    ...

This script samples ERA5 wave variables (''swh'', ''pp1d'', ''mwd'') at the
boundary points that the region's SWAN input file actually uses, and writes
one TPAR file per point.

The boundary geometry is **read from the SWAN input itself** (the ''CGRID''
line and every ''BOUndspec ... VARiable FILE <len> '<name>''' line), so the
sampled locations always match where SWAN applies each file:

* ''len'' is the distance (degrees, for spherical grids) along the side,
  measured from the side's starting corner.
* The starting corner follows SWAN's convention: with the default ''CCW''
  the grid is traversed counter-clockwise, i.e. S side W->E, E side S->N,
  N side E->W and W side N->S. ''CLOCKWise'' reverses each side.
* Rotated grids (''alpc'' != 0) are handled by rotating about the origin.

Usage
-----

    # Preview: show sample points and the max |dHs| vs. the existing files
    python tools/build_tpar_files.py --region oman --dry-run

    # Regenerate the active Oman boundary TPARs
    python tools/build_tpar_files.py --region oman

    # Only fill empty placeholder TPARs, leave populated ones untouched
    python tools/build_tpar_files.py --region oman --only-empty
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

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
               spread: Optional[Iterable[float]] = None) -> None:
    """Write a SWAN TPAR file. Default spread is 25 deg."""
    times = list(times)
    hs = np.asarray(list(hs), dtype=float)
    tp = np.asarray(list(tp), dtype=float)
    direction = np.asarray(list(direction), dtype=float)
    if spread is None:
        spread = np.full_like(hs, 25.0)
    else:
        spread = np.asarray(list(spread), dtype=float)

    with open(path, 'w') as f:
        f.write('TPAR\n')
        for t, h, p, d, s in zip(times, hs, tp, direction, spread):
            f.write(
                f'{t:%Y%m%d.%H%M%S} '
                f'{float(h):.2f} {float(p):.2f} {float(d):.2f} {float(s):.2f}\n'
            )


def read_tpar_hs(path: str) -> Optional[np.ndarray]:
    """Return the Hs column of an existing TPAR file (None if unreadable)."""
    try:
        with open(path) as f:
            rows = [ln.split() for ln in f.readlines()[1:] if ln.strip()]
        return np.array([float(r[1]) for r in rows])
    except (OSError, ValueError, IndexError):
        return None


# ------------------------------------------------------------------
# SWAN input parsing
# ------------------------------------------------------------------

@dataclass(frozen=True)
class CGrid:
    xpc: float
    ypc: float
    alpc: float
    xlenc: float
    ylenc: float


@dataclass(frozen=True)
class BoundaryFile:
    name: str
    side: str          # 'N' | 'S' | 'E' | 'W'
    clockwise: bool
    length: float      # distance along the side from its starting corner


_NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'


def _strip_comments(text: str) -> List[str]:
    """Drop SWAN comment lines and join '&' continuation lines."""
    lines, buf = [], ''
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s.startswith(('!', '$')):
            continue
        if s.endswith('&'):
            buf += s[:-1] + ' '
            continue
        lines.append(buf + s)
        buf = ''
    if buf:
        lines.append(buf)
    return lines


def parse_cgrid(text: str) -> CGrid:
    for line in _strip_comments(text):
        m = re.match(rf'CGRID\s+(?:REG\w*\s+)?({_NUM})\s+({_NUM})\s+({_NUM})'
                     rf'\s+({_NUM})\s+({_NUM})', line, re.IGNORECASE)
        if m:
            return CGrid(*(float(g) for g in m.groups()))
    raise ValueError('No CGRID REGular line found in SWAN input.')


_SIDE = {'N': 'N', 'NOR': 'N', 'S': 'S', 'SOU': 'S',
         'E': 'E', 'EAS': 'E', 'W': 'W', 'WES': 'W'}


def parse_boundaries(text: str) -> List[BoundaryFile]:
    """Every BOUndspec SIDE ... VARiable FILE <len> '<file>' [seq] pair."""
    out: List[BoundaryFile] = []
    pat = re.compile(
        r"BOU\w*\s+(?:SIDE\s+)?(N|S|E|W|NOR\w*|SOU\w*|EAS\w*|WES\w*)\s+"
        r"(?:(CCW|CLOCKW\w*)\s+)?VAR\w*\s+FILE\s+(.*)$", re.IGNORECASE)
    for line in _strip_comments(text):
        m = pat.match(line)
        if not m:
            continue
        side = _SIDE[m.group(1).upper()[:3] if len(m.group(1)) > 1
                     else m.group(1).upper()]
        clockwise = bool(m.group(2)) and m.group(2).upper().startswith('CLOCKW')
        for length, name in re.findall(rf"({_NUM})\s+'([^']+)'", m.group(3)):
            out.append(BoundaryFile(name, side, clockwise, float(length)))
    return out


def boundary_point(b: BoundaryFile, g: CGrid) -> Tuple[float, float]:
    """(lon, lat) where SWAN applies boundary file ''b''."""
    X, Y = g.xlenc, g.ylenc
    # (start corner, unit direction) in the un-rotated grid frame, CCW order
    ccw = {
        'S': ((0.0, 0.0), (1.0, 0.0)),    # SW -> SE
        'E': ((X, 0.0), (0.0, 1.0)),      # SE -> NE
        'N': ((X, Y), (-1.0, 0.0)),       # NE -> NW
        'W': ((0.0, Y), (0.0, -1.0)),     # NW -> SW
    }
    (cx, cy), (ux, uy) = ccw[b.side]
    if b.clockwise:                       # start from the other end
        side_len = X if b.side in ('S', 'N') else Y
        cx, cy = cx + ux * side_len, cy + uy * side_len
        ux, uy = -ux, -uy
    u, v = cx + ux * b.length, cy + uy * b.length
    a = np.deg2rad(g.alpc)
    lon = g.xpc + u * np.cos(a) - v * np.sin(a)
    lat = g.ypc + u * np.sin(a) + v * np.cos(a)
    return float(lon), float(lat)


# ------------------------------------------------------------------
# ERA5 sampler
# ------------------------------------------------------------------

class Era5Wave:
    """Open an ERA5 wave NetCDF once and sample nearest-cell time series."""

    def __init__(self, nc_path: str):
        import netCDF4 as nc

        ds = nc.Dataset(nc_path)
        try:
            lon_name = next(k for k in ds.variables
                            if k.lower() in ('longitude', 'lon'))
            lat_name = next(k for k in ds.variables
                            if k.lower() in ('latitude', 'lat'))
            time_name = next(k for k in ds.variables
                             if k.lower() in ('time', 'valid_time'))
            self.lons = np.asarray(ds.variables[lon_name][:], dtype=float)
            self.lats = np.asarray(ds.variables[lat_name][:], dtype=float)
            t_var = ds.variables[time_name]
            self.times = nc.num2date(
                t_var[:], t_var.units,
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            ).tolist()
            self.swh = np.ma.filled(ds.variables['swh'][:].astype(float), np.nan)
            self.pp1d = np.ma.filled(ds.variables['pp1d'][:].astype(float), np.nan)
            self.mwd = np.ma.filled(ds.variables['mwd'][:].astype(float), np.nan)
        finally:
            ds.close()

    def sample(self, lon: float, lat: float):
        lons = self.lons
        if lons.max() > 180 and lon < 0:          # 0..360 file, -180..180 query
            lon = lon % 360
        i_lon = int(np.argmin(np.abs(lons - lon)))
        i_lat = int(np.argmin(np.abs(self.lats - lat)))
        hs = self.swh[:, i_lat, i_lon]
        tp = self.pp1d[:, i_lat, i_lon]
        dr = self.mwd[:, i_lat, i_lon]
        n_bad = int((~np.isfinite(hs)).sum())
        # Replace masked / NaN with benign numbers (SWAN needs finite entries)
        hs = np.where(np.isfinite(hs), hs, 0.1)
        tp = np.where(np.isfinite(tp) & (tp > 0), tp, 5.0)
        dr = np.where(np.isfinite(dr), dr, 0.0)
        return (self.times, hs, tp, dr,
                (float(lons[i_lon]), float(self.lats[i_lat])), n_bad)


def sample_era5(nc_path: str, lon: float, lat: float):
    """Backward-compatible helper: (times, Hs, Tp, Dir) at the nearest cell."""
    times, hs, tp, dr, _, _ = Era5Wave(nc_path).sample(lon, lat)
    return times, hs, tp, dr


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description='Regenerate SWAN TPAR boundary files from ERA5.')
    p.add_argument('--region', default='oman', choices=list_regions(),
                   help='Region to regenerate TPARs for.')
    p.add_argument('--only-empty', action='store_true',
                   help='Only regenerate TPAR files that are < 50 bytes.')
    p.add_argument('--dry-run', action='store_true',
                   help='Print sample points (and the change vs. existing '
                        'files) without writing anything.')
    p.add_argument('--spread', type=float, default=25.0,
                   help='Directional spread in degrees (default 25).')
    args = p.parse_args(argv)

    cfg = get_region(args.region)
    if not os.path.exists(cfg.swan_input_path):
        print(f'ERROR: SWAN input not found: {cfg.swan_input_path}',
              file=sys.stderr)
        return 2
    if not os.path.exists(cfg.wave_nc_path):
        print(f'ERROR: ERA5 wave NetCDF not found: {cfg.wave_nc_path}',
              file=sys.stderr)
        return 2

    with open(cfg.swan_input_path) as f:
        text = f.read()
    grid = parse_cgrid(text)
    bounds = parse_boundaries(text)
    if not bounds:
        print(f'No BOUndspec ... VARiable FILE lines in {cfg.swan_input_path}.')
        return 0

    in_input = {b.name for b in bounds}
    for name in cfg.active_tpars:
        if name not in in_input:
            print(f'WARNING: {name} is listed in regions.py but not used by '
                  f'{cfg.swan_input_name}; skipping it.')

    print(f'Regenerating TPARs for {cfg.display_name}')
    print(f'  SWAN input    : {cfg.swan_input_path}')
    print(f'  CGRID         : origin=({grid.xpc}, {grid.ypc}) rot={grid.alpc} '
          f'len=({grid.xlenc}, {grid.ylenc})')
    print(f'  source NetCDF : {cfg.wave_nc_path}')
    print(f'  spread        : {args.spread} deg')
    print(f'  dry_run       : {args.dry_run}')
    print(f'  only_empty    : {args.only_empty}')
    print()

    era5 = Era5Wave(cfg.wave_nc_path)
    written, skipped = 0, 0
    for b in bounds:
        out_path = os.path.join(cfg.model_dir, b.name)
        if args.only_empty and os.path.exists(out_path) \
                and os.path.getsize(out_path) >= 50:
            skipped += 1
            continue

        lon, lat = boundary_point(b, grid)
        times, hs, tp, dr, cell, n_bad = era5.sample(lon, lat)

        diff = ''
        old = read_tpar_hs(out_path) if os.path.exists(out_path) else None
        if old is not None and len(old):
            n = min(len(old), len(hs))
            diff = f' max|dHs| vs existing={np.max(np.abs(old[:n] - hs[:n])):.2f}m'
        warn = f'  [{n_bad} NaN steps filled]' if n_bad else ''
        print(f'  {b.name:<10s} {b.side}{"cw" if b.clockwise else ""} '
              f'len={b.length:<5g} -> ({lon:8.3f},{lat:7.3f}) '
              f'ERA5 cell ({cell[0]:.2f},{cell[1]:.2f}) '
              f'Hs={hs.mean():.2f}m Tp={tp.mean():.2f}s{diff}{warn}')
        if not args.dry_run:
            spread = np.full_like(hs, args.spread, dtype=float)
            write_tpar(out_path, times, hs, tp, dr, spread)
            written += 1

    print()
    print(f'Done. Wrote {written}, skipped {skipped}.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
