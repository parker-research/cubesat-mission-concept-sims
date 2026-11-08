"""NiceGUI front end for the Canadian Arctic coverage map."""

from __future__ import annotations

import asyncio
import base64
import threading
from dataclasses import asdict

from nicegui import run, ui

from .mapping import PROJECTIONS, Settings, render_with_stats

SETTINGS = Settings()
RENDER_LOCK = threading.Lock()  # matplotlib's pyplot state is not thread-safe


def _render_locked(cfg: Settings):
    with RENDER_LOCK:
        return render_with_stats(cfg)


def _png_data_url(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode()


@ui.page("/")
def index() -> None:
    s = Settings(**asdict(SETTINGS))
    state: dict[str, bytes] = {}

    ui.colors(primary="#1f6feb")

    with ui.header().classes("items-center justify-between py-2"):
        ui.label("Canadian Arctic — coverage circles").classes("text-lg font-medium")
        status = ui.label("").classes("text-sm opacity-80")

    with ui.row().classes("w-full no-wrap gap-4 p-4"):
        with ui.column().classes("w-80 shrink-0 gap-3"):
            with ui.card().classes("w-full"):
                ui.label("Placement").classes("font-medium")
                layout = ui.select({"hex": "Even covering (hex lattice)",
                                    "grid": "Every N° lat/lon grid point"},
                                   value=s.layout, label="Layout").classes("w-full")
                ui.label("Spacing (× radius) — √3 ≈ 1.73 is the gapless minimum-overlap "
                         "covering; lower overlaps more, higher leaves holes.") \
                    .classes("text-xs opacity-70")
                spacing = ui.slider(min=1.0, max=2.0, step=0.01, value=s.spacing_factor) \
                    .props("label-always")
                target = ui.select({"canada": "Canadian land in the box",
                                    "land": "Any land in the box",
                                    "box": "The whole lat/lon box"},
                                   value=s.target, label="Area to cover").classes("w-full")
                constraint = ui.select({"any": "Anywhere (true lattice)",
                                        "box": "Inside the lat/lon box",
                                        "land": "On Canadian land"},
                                       value=s.constraint, label="Centres may sit") \
                    .classes("w-full")
                optimize = ui.switch("Fewest circles (greedy)", value=s.optimize)
                step = ui.number("Grid step (degrees)", value=s.step, min=1, max=45,
                                 step=1, format="%d").classes("w-full")
                ui.label("Grid layout only: points are every integer lat/lon divisible "
                         "by the step (5 → val % 5 == 0).").classes("text-xs opacity-70")
                with ui.row().classes("w-full no-wrap gap-2"):
                    lat_min = ui.number("Lat min", value=s.lat_min, min=-90, max=90).classes("w-1/2")
                    lat_max = ui.number("Lat max", value=s.lat_max, min=-90, max=90).classes("w-1/2")
                with ui.row().classes("w-full no-wrap gap-2"):
                    lon_min = ui.number("Lon min", value=s.lon_min, min=-180, max=180).classes("w-1/2")
                    lon_max = ui.number("Lon max", value=s.lon_max, min=-180, max=180).classes("w-1/2")
                land_only = ui.switch("Grid layout: keep land points only", value=s.land_only)

            with ui.card().classes("w-full"):
                ui.label("Circles").classes("font-medium")
                ui.label("Radius (km)").classes("text-xs opacity-70")
                radius = ui.slider(min=100, max=3000, step=25, value=s.radius_km) \
                    .props("label-always snap")
                fill = ui.switch("Shade interiors", value=s.fill)
                centers = ui.switch("Mark centres", value=s.show_centers)
                labels = ui.switch("Label centres", value=s.show_labels)
                graticule = ui.switch("Graticule", value=s.show_graticule)
                frame = ui.select({"circles": "Fit all circles", "grid": "Fit grid area"},
                                  value=s.frame, label="Framing").classes("w-full")

            with ui.card().classes("w-full"):
                ui.label("Projection").classes("font-medium")
                projection = ui.select(list(PROJECTIONS), value=s.projection).classes("w-full")
                ui.label("All choices are azimuthal and centred on the archipelago — "
                         "the right family this far north.").classes("text-xs opacity-70")
                resolution = ui.select({"110m": "Coarse (110m)", "50m": "Medium (50m)",
                                        "10m": "Fine (10m)"}, value=s.resolution,
                                       label="Coastline detail").classes("w-full")

            with ui.row().classes("w-full no-wrap gap-2"):
                ui.button("Redraw", icon="refresh", on_click=lambda: redraw()).classes("grow")
                ui.button("PNG", icon="download",
                          on_click=lambda: ui.download.content(
                              state.get("png", b""), "canadian-arctic-coverage.png")) \
                    .props("outline")

        with ui.column().classes("grow items-center"):
            spinner = ui.spinner(size="lg").classes("mt-8")
            image = ui.image().classes("w-full max-w-5xl rounded shadow").props("fit=contain")
            image.visible = False
            caption = ui.label("").classes("text-sm opacity-70 mt-2")

    def collect() -> Settings:
        return Settings(
            lat_min=float(lat_min.value or -90),
            lat_max=float(lat_max.value or 90),
            lon_min=float(lon_min.value or -180),
            lon_max=float(lon_max.value or 180),
            step=int(step.value or 5),
            layout=layout.value,
            spacing_factor=float(spacing.value or 1.73),
            target=target.value,
            constraint=constraint.value,
            optimize=bool(optimize.value),
            radius_km=float(radius.value or 1500),
            projection=projection.value,
            resolution=resolution.value,
            land_only=bool(land_only.value),
            fill=bool(fill.value),
            show_centers=bool(centers.value),
            show_labels=bool(labels.value),
            show_graticule=bool(graticule.value),
            frame=frame.value,
        )

    async def redraw() -> None:
        cfg = collect()
        spinner.visible = True
        status.text = "planning and rendering…"
        png, stats = await run.io_bound(_render_locked, cfg)
        state["png"] = png
        image.set_source(_png_data_url(png))
        image.visible = True
        spinner.visible = False
        status.text = f"{stats.n_circles} circles · {stats.covered_fraction * 100:.1f}% covered"
        caption.text = f"{stats.summary()} · {cfg.radius_km:g} km radius"

    pending: dict[str, asyncio.Task] = {}

    def schedule() -> None:
        """Coalesce bursts of control changes (slider drags) into one redraw."""
        if task := pending.pop("task", None):
            task.cancel()

        async def debounced() -> None:
            await asyncio.sleep(0.35)
            await redraw()

        pending["task"] = asyncio.create_task(debounced())

    for control in (step, lat_min, lat_max, lon_min, lon_max, radius, projection,
                    resolution, land_only, fill, centers, labels, graticule, frame,
                    layout, spacing, target, constraint, optimize):
        control.on_value_change(lambda _: schedule())

    ui.timer(0.1, redraw, once=True)


def main() -> None:
    ui.run(title="Arctic ANT Coverage", reload=False, port=8080, show=False)


if __name__ in {"__main__", "__mp_main__"}:
    main()
