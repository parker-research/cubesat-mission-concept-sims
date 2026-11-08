"""Placing coverage circles over the Canadian Arctic.

Two layouts:

* ``grid`` — the original one: every integer lat/lon divisible by ``step``.
* ``hex`` — a hexagonal covering. Circles of radius R laid on a triangular
  lattice cover the plane with no gaps when the centres are ``R * sqrt(3)``
  apart; that spacing is the minimum-overlap covering, so it is the default.
  Tighter spacing buys margin against the gaps that curvature and the coastline
  leave; wider spacing leaves holes.

The lattice is built in an equal-area projection centred on the archipelago, so
"evenly spaced" means evenly spaced on the ground, not on the page. Candidate
centres that contribute nothing are then dropped, greedily, smallest first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from cartopy.geodesic import Geodesic

CENTER_LON = -96.0
CENTER_LAT = 72.0

# Equal-area working plane for the lattice and for the coverage sampling.
WORK_CRS = ccrs.LambertAzimuthalEqualArea(
    central_longitude=CENTER_LON, central_latitude=CENTER_LAT
)
GEODETIC = ccrs.Geodetic()
GEOD = Geodesic()

SQRT3 = math.sqrt(3.0)


@dataclass(frozen=True)
class Coverage:
    """What a set of centres achieves against the target area."""

    covered_fraction: float  # 0..1 of target sample points within some circle
    mean_overlap: float  # mean circles covering a covered sample point
    max_overlap: int
    n_circles: int
    n_samples: int

    def summary(self) -> str:
        return (
            f"{self.n_circles} circles · {self.covered_fraction * 100:.1f}% of the "
            f"target covered · {self.mean_overlap:.2f}× mean overlap "
            f"(max {self.max_overlap})"
        )


@lru_cache(maxsize=8)
def _canada_geometry():
    """Canada's landmass as one prepared shapely geometry (lon/lat)."""
    import cartopy.io.shapereader as shpreader
    from shapely.ops import unary_union

    path = shpreader.natural_earth("50m", "cultural", "admin_0_countries")
    for rec in shpreader.Reader(path).records():
        name = rec.attributes.get("ADMIN") or rec.attributes.get("admin")
        if name == "Canada":
            return unary_union(rec.geometry)
    raise RuntimeError("Canada not found in the Natural Earth countries file")


@lru_cache(maxsize=16)
def target_samples(
    target: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    sample_km: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-area sample points of the region to be covered.

    Returns (lonlat Nx2, xy Nx2 in WORK_CRS). Sampling on the equal-area plane
    means every sample stands for the same number of square kilometres, so
    "fraction covered" is a fraction of area, not a fraction of degrees.
    """
    corner_lons = np.repeat(np.linspace(lon_min, lon_max, 120), 120)
    corner_lats = np.tile(np.linspace(lat_min, lat_max, 120), 120)
    box = WORK_CRS.transform_points(GEODETIC, corner_lons, corner_lats)[:, :2]
    box = box[np.isfinite(box).all(axis=1)]
    x0, y0 = box.min(axis=0)
    x1, y1 = box.max(axis=0)

    step = sample_km * 1000.0
    xs = np.arange(x0, x1 + step, step)
    ys = np.arange(y0, y1 + step, step)
    gx, gy = np.meshgrid(xs, ys)
    xy = np.column_stack([gx.ravel(), gy.ravel()])

    lonlat = GEODETIC.transform_points(WORK_CRS, xy[:, 0], xy[:, 1])[:, :2]
    ok = np.isfinite(lonlat).all(axis=1)
    lonlat, xy = lonlat[ok], xy[ok]

    inside = (
        (lonlat[:, 1] >= lat_min)
        & (lonlat[:, 1] <= lat_max)
        & (lonlat[:, 0] >= lon_min)
        & (lonlat[:, 0] <= lon_max)
    )
    lonlat, xy = lonlat[inside], xy[inside]

    if target in ("land", "canada"):
        from shapely.geometry import Point
        from shapely.prepared import prep

        if target == "canada":
            geom = prep(_canada_geometry())
        else:
            from shapely.geometry import MultiPolygon

            parts = list(cfeature.LAND.with_scale("50m").geometries())
            geom = prep(MultiPolygon([g for p in parts for g in getattr(p, "geoms", [p])]))
        keep = np.array([geom.contains(Point(lo, la)) for lo, la in lonlat], dtype=bool)
        lonlat, xy = lonlat[keep], xy[keep]

    return lonlat, xy


def _distance_matrix(centres_lonlat: np.ndarray, samples_lonlat: np.ndarray) -> np.ndarray:
    """Geodesic distance (m) from each centre to each sample: (C, S)."""
    out = np.empty((len(centres_lonlat), len(samples_lonlat)), dtype=float)
    for i, c in enumerate(centres_lonlat):
        res = np.asarray(GEOD.inverse(c[np.newaxis, :], samples_lonlat))
        out[i] = res[:, 0]
    return out


def hex_centres(
    radius_km: float,
    spacing_factor: float,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> np.ndarray:
    """Triangular-lattice centres (lon, lat) blanketing the working box."""
    corner_lons = np.repeat(np.linspace(lon_min, lon_max, 120), 120)
    corner_lats = np.tile(np.linspace(lat_min, lat_max, 120), 120)
    box = WORK_CRS.transform_points(GEODETIC, corner_lons, corner_lats)[:, :2]
    box = box[np.isfinite(box).all(axis=1)]
    x0, y0 = box.min(axis=0)
    x1, y1 = box.max(axis=0)

    d = radius_km * 1000.0 * spacing_factor  # centre-to-centre spacing
    row_h = d * SQRT3 / 2.0
    pad = d  # one ring of centres outside the box, so edges stay covered

    centres = []
    y = y0 - pad
    row = 0
    while y <= y1 + pad:
        offset = (d / 2.0) if row % 2 else 0.0
        x = x0 - pad + offset
        while x <= x1 + pad:
            centres.append((x, y))
            x += d
        y += row_h
        row += 1

    xy = np.asarray(centres)
    lonlat = GEODETIC.transform_points(WORK_CRS, xy[:, 0], xy[:, 1])[:, :2]
    return lonlat[np.isfinite(lonlat).all(axis=1)]


def _constrain(candidates: np.ndarray, constraint: str,
               lat_min: float, lat_max: float,
               lon_min: float, lon_max: float) -> np.ndarray:
    """Restrict where a circle centre may sit."""
    if constraint == "any":
        return candidates
    inside = (
        (candidates[:, 1] >= lat_min)
        & (candidates[:, 1] <= lat_max)
        & (candidates[:, 0] >= lon_min)
        & (candidates[:, 0] <= lon_max)
    )
    candidates = candidates[inside]
    if constraint == "land" and len(candidates):
        from shapely.geometry import Point
        from shapely.prepared import prep

        geom = prep(_canada_geometry())
        keep = np.array([geom.contains(Point(lo, la)) for lo, la in candidates], dtype=bool)
        candidates = candidates[keep]
    return candidates


@lru_cache(maxsize=32)
def _plan_cached(
    radius_km: float,
    spacing_factor: float,
    target: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    sample_km: float = 40.0,
    prune: bool = True,
    constraint: str = "any",
    optimize: bool = False,
) -> tuple[list[tuple[float, float]], Coverage]:
    """Hex-covering centres for the target area, plus its coverage statistics."""
    samples_lonlat, _ = target_samples(target, lat_min, lat_max, lon_min, lon_max, sample_km)
    if len(samples_lonlat) == 0:
        return [], Coverage(0.0, 0.0, 0, 0, 0)

    # A constrained centre cannot sit wherever the lattice wants it, so offer a
    # denser pool and let a greedy set-cover pick from it.
    greedy = optimize or constraint != "any"
    factor = spacing_factor / 2.0 if greedy else spacing_factor
    candidates = hex_centres(radius_km, factor, lat_min, lat_max, lon_min, lon_max)
    candidates = _constrain(candidates, constraint, lat_min, lat_max, lon_min, lon_max)
    if not len(candidates):
        return [], Coverage(0.0, 0.0, 0, len(samples_lonlat), len(samples_lonlat))

    dist = _distance_matrix(candidates, samples_lonlat)
    covers = dist <= radius_km * 1000.0

    keep = covers.any(axis=1)
    candidates, covers = candidates[keep], covers[keep]

    if greedy and len(candidates):
        # Greedy max-coverage: take the circle adding the most uncovered area,
        # repeat while it still adds something.
        chosen: list[int] = []
        uncovered = np.ones(covers.shape[1], dtype=bool)
        while uncovered.any():
            gains = covers[:, uncovered].sum(axis=1)
            gains[chosen] = 0
            best = int(np.argmax(gains))
            if gains[best] == 0:
                break
            chosen.append(best)
            uncovered &= ~covers[best]
        candidates, covers = candidates[chosen], covers[chosen]

    if prune and len(candidates):
        # Drop circles that cover nothing another kept circle does not, rarest
        # contribution first, so the survivors are the ones carrying the area.
        counts = covers.sum(axis=1)
        order = np.argsort(counts)
        alive = np.ones(len(candidates), dtype=bool)
        for i in order:
            alive[i] = False
            others = covers[alive]
            still = others.any(axis=0) if len(others) else np.zeros(covers.shape[1], bool)
            if not np.all(still[covers[i]]):
                alive[i] = True  # it was the only cover for something
        candidates, covers = candidates[alive], covers[alive]

    hits = covers.sum(axis=0)
    covered = hits > 0
    stats = Coverage(
        covered_fraction=float(covered.mean()),
        mean_overlap=float(hits[covered].mean()) if covered.any() else 0.0,
        max_overlap=int(hits.max()) if len(hits) else 0,
        n_circles=len(candidates),
        n_samples=len(samples_lonlat),
    )
    return tuple((float(lo), float(la)) for lo, la in candidates), stats


def plan(*args, **kwargs) -> tuple[list[tuple[float, float]], Coverage]:
    """Cached wrapper around the planner; see :func:`_plan_cached`."""
    centres, stats = _plan_cached(*args, **kwargs)
    return list(centres), stats


def evaluate(
    centres: list[tuple[float, float]],
    radius_km: float,
    target: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    sample_km: float = 40.0,
) -> Coverage:
    """Coverage statistics for an arbitrary set of centres."""
    samples_lonlat, _ = target_samples(target, lat_min, lat_max, lon_min, lon_max, sample_km)
    if not centres or len(samples_lonlat) == 0:
        return Coverage(0.0, 0.0, 0, len(centres), len(samples_lonlat))
    dist = _distance_matrix(np.asarray(centres, dtype=float), samples_lonlat)
    hits = (dist <= radius_km * 1000.0).sum(axis=0)
    covered = hits > 0
    return Coverage(
        covered_fraction=float(covered.mean()),
        mean_overlap=float(hits[covered].mean()) if covered.any() else 0.0,
        max_overlap=int(hits.max()),
        n_circles=len(centres),
        n_samples=len(samples_lonlat),
    )
