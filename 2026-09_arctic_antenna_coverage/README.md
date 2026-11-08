# Arctic ANT coverage

Where would you put antennas in the Canadian arctic to capture as much data downlinked by cubesats as possible, without large overlap? This project aims to solve that question.

## Intro

A NiceGUI app that maps the Canadian Arctic and plans **1200 km radius coverage
circles** over it.

Two placement layouts:

* **Even covering (default)** — a hexagonal lattice built on an equal-area
  plane. Circles of radius R on a triangular lattice tile the plane without
  gaps when centres are `R·√3` apart, which is the minimum-overlap covering, so
  `1.73×` is the default spacing. Over Canadian land between 60–85° N and
  141–60° W that comes to **6 circles, 100% covered, 1.27× mean overlap**.
  Constrain the centres to the box and let it optimise and it drops to 4
  circles (2.0× overlap); constrain them to Canadian land and 3 circles reach
  94% of the area.
* **Grid** — the original: every integer lat/lon divisible by N (`val % 5 == 0`).

Coverage is measured, not assumed: the target area is sampled on an equal-area
grid (40 km), so "100% covered" is a fraction of area, and the overlap figure is
the mean number of circles over each covered sample.

## Run

```sh
uv run arctic-ant-coverage      # http://localhost:8080
```

`uv sync` first if you want the environment built explicitly. Coastlines come
from Natural Earth and are downloaded and cached by cartopy on first use, so the
first render needs network access.

## Projection

Web-Mercator map tiles are useless this far north, so the map is drawn with
matplotlib + cartopy in an azimuthal projection centred on the archipelago
(96° W, 72° N). Three choices:

| Projection | Why |
| --- | --- |
| Lambert azimuthal equal-area (default) | Areas are true, so coverage footprints are comparable anywhere on the map. |
| Polar stereographic, 70° N true scale | Conformal; the standard polar/sea-ice chart projection. |
| Azimuthal equidistant | Distance from the centre point is true, so ranges read directly off the map. |

Circles are **geodesic** (WGS84, via `cartopy.geodesic.Geodesic`), not circles in
projected space, and their vertices are projected individually — so rings that
swallow the pole or cross the antimeridian still close correctly.

## Controls

- Layout, centre spacing (1.0–2.0× radius), area to cover (Canadian land / any
  land / whole box), where centres may sit (anywhere / in the box / on Canadian
  land), and a greedy "fewest circles" mode.
- Grid step and lat/lon bounds; optional "land points only" filter for the grid layout.
- Circle radius (100–3000 km), interior shading, centre markers and labels, graticule.
- Framing: fit every circle, or fit the grid box.
- Coastline detail (110m / 50m / 10m) and a PNG download.

## Layout

- `src/arctic_ant_coverage/layout.py` — hex lattice, equal-area target sampling, greedy set-cover, coverage statistics.
- `src/arctic_ant_coverage/mapping.py` — settings and figure rendering (headless Agg).
- `src/arctic_ant_coverage/main.py` — NiceGUI page; rendering runs off the event loop with a debounce.
