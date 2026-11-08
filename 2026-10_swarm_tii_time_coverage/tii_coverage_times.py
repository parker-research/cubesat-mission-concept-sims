#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "viresclient",
#     "pandas",
#     "numpy",
#     "matplotlib",
# ]
# ///
"""
Swarm TII data coverage map: which UTC times have TII cross-track ion-flow
data on each day, per spacecraft.

Data source: ESA VirES for Swarm (https://vires.services), product
SW_EXPT_EFIx_TCT02 (2 Hz TII cross-track flow).

One-time setup (token from https://vires.services/accounts/tokens/):
    uvx viresclient set_token https://vires.services/ows

Examples:
    uv run tii_coverage.py --start 2014-01-01 --end 2015-01-01
    uv run tii_coverage.py --spacecraft A B --start 2023-01-01 --end 2024-01-01 --quality
    uv run tii_coverage.py --replot tii_coverage.csv          # re-plot without downloading

How it works: data are requested downsampled (default every 10 s) in monthly
chunks, and each chunk is reduced straight away to "how many samples fell in
each N-minute UTC bin of each day". Results are cached to the CSV so an
interrupted run resumes where it left off.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Bit for Viy in TCT Quality_flags (bit 0 Vixh, 1 Vixv, 2 Viy, 3 Viz; set = good).
# Double-check against the TCT release notes on the Swarm dissemination server.
VIY_GOOD_BIT = 2

VIRES_URL = "https://vires.services/ows"


def step_seconds(step):
    return pd.Timedelta(step).total_seconds()


def fetch_month(sc, t0, t1, step, quality):
    """Return a Series of timestamps (optionally good-Viy only) for one chunk."""
    from viresclient import SwarmRequest

    req = SwarmRequest(VIRES_URL)
    req.set_collection(f"SW_EXPT_EFI{sc}_TCT02")
    req.set_products(measurements=["Viy", "Quality_flags"], sampling_step=step)
    try:
        data = req.get_between(t0, t1, asynchronous=True, show_progress=False)
        df = data.as_dataframe()
    except Exception as exc:  # empty periods raise on some versions
        msg = str(exc).lower()
        if "no data" in msg or "empty" in msg:
            return pd.DatetimeIndex([])
        raise
    if df.empty:
        return pd.DatetimeIndex([])
    if quality:
        good = (df["Quality_flags"].astype(int) >> VIY_GOOD_BIT) & 1
        df = df[good == 1]
    return pd.DatetimeIndex(df.index)


def to_bins(times, bin_minutes):
    """Count samples per (day, UTC bin)."""
    if len(times) == 0:
        return pd.DataFrame(columns=["date", "bin", "count"])
    t = times.tz_localize(None) if times.tz is not None else times
    day = t.normalize()
    minute_of_day = (t - day).total_seconds() / 60.0
    b = (minute_of_day // bin_minutes).astype(int)
    out = (pd.DataFrame({"date": day, "bin": b})
           .groupby(["date", "bin"]).size().rename("count").reset_index())
    return out


def collect(args):
    cache = args.csv
    done = pd.DataFrame(columns=["spacecraft", "month", "date", "bin", "count"])
    if os.path.exists(cache):
        done = pd.read_csv(cache, parse_dates=["date"])
    months = pd.date_range(pd.Timestamp(args.start).to_period("M").to_timestamp(),
                           pd.Timestamp(args.end), freq="MS")
    frames = [done] if not done.empty else []
    for sc in args.spacecraft:
        have = set(done.loc[done["spacecraft"] == sc, "month"].astype(str))
        for m0 in months:
            key = m0.strftime("%Y-%m")
            if key in have:
                continue
            t0 = max(m0, pd.Timestamp(args.start))
            t1 = min(m0 + pd.offsets.MonthBegin(1), pd.Timestamp(args.end))
            print(f"Swarm {sc} {key} ...", end=" ", flush=True)
            times = fetch_month(sc, t0.to_pydatetime(), t1.to_pydatetime(),
                                args.step, args.quality)
            b = to_bins(times, args.bin_minutes)
            print(f"{len(times):,} samples")
            if b.empty:  # placeholder row so the month counts as done
                b = pd.DataFrame({"date": [m0], "bin": [-1], "count": [0]})
            b.insert(0, "month", key)
            b.insert(0, "spacecraft", sc)
            frames.append(b)
            pd.concat(frames, ignore_index=True).to_csv(cache, index=False)
    return pd.concat(frames, ignore_index=True)


def plot(cov, args):
    cov = cov.astype({"bin": int, "count": int, "spacecraft": str})
    cov = cov[cov["bin"] >= 0].copy()
    cov["date"] = pd.to_datetime(cov["date"])
    nbins = int(24 * 60 // args.bin_minutes)
    expected = args.bin_minutes * 60 / step_seconds(args.step)
    sats = [s for s in args.spacecraft if s in set(cov["spacecraft"])]
    if not sats:
        sys.exit("No TII data found for the requested spacecraft/time range.")

    days = pd.date_range(pd.Timestamp(args.start).normalize(),
                         pd.Timestamp(args.end).normalize() - pd.Timedelta(days=1), freq="D")
    fig, axes = plt.subplots(2 * len(sats), 1, figsize=(14, 4.2 * len(sats)),
                             sharex=True, squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1] * len(sats)})
    axes = axes[:, 0]
    x_edges = mdates.date2num(list(days) + [days[-1] + pd.Timedelta(days=1)])
    y_edges = np.arange(nbins + 1) * args.bin_minutes / 60.0

    for i, sc in enumerate(sats):
        sub = cov[cov["spacecraft"] == sc]
        grid = np.zeros((nbins, len(days)))
        col = days.get_indexer(sub["date"])
        ok = col >= 0
        grid[sub["bin"].values[ok], col[ok]] = np.clip(sub["count"].values[ok] / expected, 0, 1)
        grid_masked = np.ma.masked_equal(grid, 0)

        ax = axes[2 * i]
        mesh = ax.pcolormesh(x_edges, y_edges, grid_masked, cmap="viridis",
                             vmin=0, vmax=1, shading="flat")
        ax.set_ylim(0, 24)
        ax.set_yticks(range(0, 25, 3))
        ax.set_ylabel("UTC hour")
        label = "good-quality Viy" if args.quality else "any TII record"
        ax.set_title(f"Swarm {sc}: TII cross-track flow coverage ({label})")
        ax.set_facecolor("#eeeeee")
        fig.colorbar(mesh, ax=ax, pad=0.01, label="fraction of bin covered")

        hours = grid.sum(axis=0) * args.bin_minutes / 60.0
        axd = axes[2 * i + 1]
        axd.bar(x_edges[:-1], hours, width=1.0, align="edge", color="tab:blue")
        axd.set_ylim(0, 24)
        axd.set_ylabel("hours/day")
        fig.colorbar(mesh, ax=axd, pad=0.01).ax.set_visible(False)  # keep widths aligned

    loc = mdates.AutoDateLocator()
    axes[-1].xaxis.set_major_locator(loc)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    axes[-1].set_xlim(x_edges[0], x_edges[-1])
    fig.tight_layout()
    fig.savefig(args.png, dpi=150)
    print(f"Saved {args.png}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spacecraft", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    p.add_argument("--start", default="2014-01-01")
    p.add_argument("--end", default="2014-04-01", help="exclusive")
    p.add_argument("--step", default="PT10S", help="VirES sampling step (ISO 8601)")
    p.add_argument("--bin-minutes", type=float, default=5)
    p.add_argument("--quality", action="store_true", help="count only good-quality Viy")
    p.add_argument("--csv", default="tii_coverage.csv", help="cache / output table")
    p.add_argument("--png", default="tii_coverage.png")
    p.add_argument("--replot", metavar="CSV", help="plot an existing CSV, no download")
    args = p.parse_args()

    if args.replot:
        cov = pd.read_csv(args.replot, parse_dates=["date"])
        valid = cov[cov["bin"] >= 0]
        if "--start" not in sys.argv:
            args.start = str(valid["date"].min().date())
        if "--end" not in sys.argv:
            args.end = str((valid["date"].max() + pd.Timedelta(days=1)).date())
    else:
        cov = collect(args)
    plot(cov, args)


if __name__ == "__main__":
    main()

