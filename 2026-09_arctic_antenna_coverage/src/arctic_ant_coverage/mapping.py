"""Rendering of the Canadian Arctic coverage map.

Grid points are every integer latitude/longitude that is a multiple of `step`
(5 by default, i.e. ``val % 5 == 0``); each one gets a geodesic circle of
`radius_km` drawn around it.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from functools import lru_cache

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
import numpy as np
from cartopy.geodesic import Geodesic

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon as MplPolygon  # noqa: E402

# Centre of the area of interest: roughly the middle of the Canadian Arctic
# Archipelago, used as the tangent point of the azimuthal projections.
CENTER_LON = -96.0
CENTER_LAT = 72.0

# All three are azimuthal projections centred on the region, which is what you
# want this far north -- Mercator-family projections tear the Arctic apart.
PROJECTIONS: dict[str, object] = {
    # Equal-area: circle areas are comparable across the whole map.
    "Lambert azimuthal equal-area": lambda: ccrs.LambertAzimuthalEqualArea(
        central_longitude=CENTER_LON, central_latitude=CENTER_LAT
    ),
    # Conformal, the classic sea-ice / polar chart projection.
    "Polar stereographic (70N true scale)": lambda: ccrs.NorthPolarStereo(
        central_longitude=CENTER_LON, true_scale_latitude=70
    ),
    # Distances from the centre point are true, so range rings read directly.
    "Azimuthal equidistant": lambda: ccrs.AzimuthalEquidistant(
        central_longitude=CENTER_LON, central_latitude=CENTER_LAT
    ),
}

GEOD = Geodesic()


@lru_cache(maxsize=32)
def _feature(name: str, resolution: str):
    """Scaled Natural Earth feature, reused across renders.

    Reuse matters: cartopy caches each feature's projected geometry against the
    feature object, so a fresh ``with_scale`` call per render costs ~6 s of
    reprojection that a cached one skips.
    """
    return getattr(cfeature, name).with_scale(resolution)


@dataclass
class Settings:
    lat_min: float = 60.0
    lat_max: float = 85.0
    lon_min: float = -141.0
    lon_max: float = -60.0
    step: int = 5
    radius_km: float = 1200.0
    layout: str = "hex"  # "hex" covering, or "grid" (every lat/lon % step == 0)
    spacing_factor: float = 1.73  # centre spacing in units of the radius
    target: str = "canada"  # what must be covered: "canada", "land" or "box"
    prune: bool = True
    constraint: str = "any"  # centre placement: "any", "box" or "land"
    optimize: bool = False  # greedy fewest circles instead of an even lattice
    projection: str = next(iter(PROJECTIONS))
    resolution: str = "50m"
    land_only: bool = False
    fill: bool = False
    fill_alpha: float = 0.05
    line_width: float = 0.8
    show_centers: bool = True
    show_labels: bool = False
    show_graticule: bool = True
    frame: str = "circles"  # "circles" fits every ring; "grid" fits the lat/lon box
    margin: float = 0.04
    dpi: int = 130
    figsize: tuple[float, float] = field(default=(11.0, 9.0))


_LAND_PREP = None


def _on_land(lon: float, lat: float) -> bool:
    """True if the point falls on land (Natural Earth 50m)."""
    global _LAND_PREP
    if _LAND_PREP is None:
        from shapely.geometry import MultiPolygon
        from shapely.prepared import prep

        geoms = list(cfeature.LAND.with_scale("50m").geometries())
        _LAND_PREP = prep(MultiPolygon([g for p in geoms for g in getattr(p, "geoms", [p])]))
    from shapely.geometry import Point

    return _LAND_PREP.contains(Point(lon, lat))


def points_for(s: Settings) -> tuple[list[tuple[float, float]], "object"]:
    """Circle centres for these settings, with their coverage statistics."""
    from . import layout as _layout

    if s.layout == "hex":
        return _layout.plan(
            radius_km=s.radius_km,
            spacing_factor=s.spacing_factor,
            target=s.target,
            lat_min=s.lat_min,
            lat_max=s.lat_max,
            lon_min=s.lon_min,
            lon_max=s.lon_max,
            prune=s.prune,
            constraint=s.constraint,
            optimize=s.optimize,
        )
    pts = grid_points(s)
    return pts, _layout.evaluate(
        pts, s.radius_km, s.target, s.lat_min, s.lat_max, s.lon_min, s.lon_max
    )


def grid_points(s: Settings) -> list[tuple[float, float]]:
    """Every (lon, lat) with both values integer multiples of ``s.step``."""
    step = max(1, int(s.step))
    lats = [v for v in range(-90, 91, step) if s.lat_min <= v <= s.lat_max]
    lons = [v for v in range(-180, 181, step) if s.lon_min <= v <= s.lon_max]
    points: list[tuple[float, float]] = []
    for lat in lats:
        if abs(lat) == 90:  # the pole is one point, not one per meridian
            points.append((float(lons[0] if lons else 0), float(lat)))
            continue
        points.extend((float(lon), float(lat)) for lon in lons)
    if s.land_only:
        points = [p for p in points if _on_land(*p)]
    return points


def _circle_xy(lon: float, lat: float, radius_m: float, proj) -> np.ndarray:
    """Geodesic circle around (lon, lat), in projection coordinates.

    Projecting the vertices ourselves (rather than handing cartopy a lon/lat
    polygon) keeps circles that swallow the pole or cross the dateline intact.
    """
    pts = np.asarray(GEOD.circle(lon, lat, radius_m, n_samples=361, endpoint=True))
    xyz = proj.transform_points(ccrs.Geodetic(), pts[:, 0], pts[:, 1])
    return xyz[:, :2]


def _label(lon: float, lat: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(int(lat))}°{ns} {abs(int(lon))}°{ew}"


def render_with_stats(s: Settings):
    """Render, and also hand back the coverage statistics."""
    points, stats = points_for(s)
    return render(s, points, stats), stats


def render(s: Settings, points=None, stats=None) -> bytes:
    """Draw the map and return it as PNG bytes."""
    proj = PROJECTIONS[s.projection]()
    if points is None or stats is None:
        points, stats = points_for(s)
    radius_m = s.radius_km * 1000.0

    fig = plt.figure(figsize=s.figsize, dpi=s.dpi)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_facecolor("#eaf2fa")

    ax.add_feature(_feature("OCEAN", s.resolution), facecolor="#dce9f5")
    ax.add_feature(_feature("LAND", s.resolution), facecolor="#f3efe4")
    ax.add_feature(_feature("LAKES", s.resolution), facecolor="#dce9f5",
                   edgecolor="#9fb6c9", linewidth=0.3)
    ax.add_feature(_feature("COASTLINE", s.resolution),
                   edgecolor="#5d6b78", linewidth=0.5)
    ax.add_feature(_feature("BORDERS", s.resolution),
                   edgecolor="#8d8577", linewidth=0.5, linestyle=":")

    cmap = plt.get_cmap("turbo")
    all_xy: list[np.ndarray] = []
    for i, (lon, lat) in enumerate(points):
        xy = _circle_xy(lon, lat, radius_m, proj)
        finite = xy[np.isfinite(xy).all(axis=1)]
        if len(finite) < 3:
            continue
        all_xy.append(finite)
        color = cmap((i % max(len(points), 1)) / max(len(points), 1))
        ax.add_patch(
            MplPolygon(
                finite,
                closed=True,
                facecolor=color if s.fill else "none",
                alpha=s.fill_alpha if s.fill else 1.0,
                edgecolor="none",
                zorder=3,
            )
        )
        ax.add_patch(
            MplPolygon(
                finite,
                closed=True,
                facecolor="none",
                edgecolor=color,
                linewidth=s.line_width,
                zorder=4,
            )
        )

    if points and s.show_centers:
        lons, lats = zip(*points)
        ax.plot(lons, lats, "o", transform=ccrs.PlateCarree(), markersize=2.6,
                color="#1a1a1a", linestyle="none", zorder=6)
    if points and s.show_labels:
        for lon, lat in points:
            ax.text(lon, lat, _label(lon, lat), transform=ccrs.PlateCarree(),
                    fontsize=5.5, ha="center", va="bottom", color="#1a1a1a",
                    zorder=7, clip_on=True)

    if s.show_graticule:
        fine = max(5, int(s.step))
        # Only draw meridians/parallels near the region: projecting the whole
        # globe's graticule is most of the render cost.
        lo0, lo1 = max(-180, s.lon_min - 40), min(180, s.lon_max + 40)
        la0, la1 = max(-90, s.lat_min - 30), min(90, s.lat_max + 8)
        xr = lambda st: range(int(lo0 // st * st), int(lo1) + st, st)  # noqa: E731
        yr = lambda st: range(int(la0 // st * st), int(la1) + st, st)  # noqa: E731
        if fine < 15:  # unlabelled fine grid; labelling every line is slow
            ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                         xlocs=xr(fine), ylocs=yr(fine),
                         linewidth=0.25, color="#7a8894", alpha=0.35, linestyle=":")
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            xlocs=xr(15),
            ylocs=yr(max(5, fine)),
            linewidth=0.35,
            color="#7a8894",
            alpha=0.55,
            linestyle="--",
        )
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 7, "color": "#43505c"}

    # Frame either on the circles (nothing clipped) or on the grid box itself.
    if s.frame == "grid" and points:
        lons = np.linspace(s.lon_min, s.lon_max, 60)
        lats = np.linspace(s.lat_min, s.lat_max, 60)
        gl_, ga_ = np.meshgrid(lons, lats)
        edge = proj.transform_points(ccrs.Geodetic(), gl_.ravel(), ga_.ravel())[:, :2]
        stacked_list = [edge[np.isfinite(edge).all(axis=1)]]
    else:
        stacked_list = all_xy
    if stacked_list:
        stacked = np.vstack(stacked_list)
        x0, y0 = stacked.min(axis=0)
        x1, y1 = stacked.max(axis=0)
        pad = s.margin * max(x1 - x0, y1 - y0)
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad)
    else:
        ax.set_global()

    if s.layout != "hex":
        placement = f"every {s.step}° lat/lon grid point"
    elif s.optimize or s.constraint != "any":
        placement = f"greedy covering, centres: {s.constraint}"
    else:
        placement = f"hexagonal covering, {s.spacing_factor:.2f}× radius spacing"
    # fig.suptitle rather than ax.set_title: cartopy's gridline labels can push
    # the axes title to an infinite y and it vanishes off the canvas.
    fig.suptitle(
        f"Canadian Arctic — {s.radius_km:g} km radius circles, {placement}\n"
        f"{stats.summary()} · {s.projection}",
        fontsize=11,
        y=0.955,
    )

    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.06)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue()
