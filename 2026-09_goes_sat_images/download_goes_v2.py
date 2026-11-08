#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests>=2.31",
#     "tyro>=0.9",
# ]
# ///
"""
Download GOES-18 ABI Full Disk GeoColor images from the NOAA STAR CDN
directory listing:

    https://cdn.star.nesdis.noaa.gov/GOES18/ABI/FD/GEOCOLOR/

The directory contains many images per timestamp at different resolutions,
e.g.:
    20261750100_GOES18-ABI-FD-GEOCOLOR-339x339.jpg       ~120 KB
    20261750100_GOES18-ABI-FD-GEOCOLOR-678x678.jpg       ~420 KB   <- default
    20261750110_GOES18-ABI-FD-GEOCOLOR-1808x1808.jpg     ~2.3 MB
    20261750100_GOES18-ABI-FD-GEOCOLOR-5424x5424.jpg     ~15 MB
    20261750110_GOES18-ABI-FD-GEOCOLOR-10848x10848.jpg   ~16 MB
    20261750110_GOES18-ABI-FD-GEOCOLOR-21696x21696.jpg   ~45-50 MB (largest)

This script:
  1. Fetches and parses the directory listing (HTML <a href> links).
  2. Groups files by their timestamp/base name.
  3. Picks the file matching the requested `--resolution` for each
     timestamp (default 678x678 -- small and fast, good for testing).
     Pass `--resolution highest` to always grab the biggest file
     published for each timestamp instead of a fixed size.
  4. Downloads the selected files. Like `wget -c`, it checks any existing
     local file against the server before touching the network transfer:
     matching size is left alone, a short/partial file is resumed with an
     HTTP Range request, and a mismatched file (e.g. local is somehow
     larger than remote) is re-downloaded from scratch. The final size is
     verified against Content-Length.

Run directly (uv will read the inline `/// script` block and set up an
ephemeral environment automatically):

    ./download_goes18_geocolor.py --out ./goes18_geocolor

Or with a plain interpreter, after `pip install requests tyro`:

    python download_goes18_geocolor.py --out ./goes18_geocolor

Examples:
    # Download everything currently listed (defaults to 678x678 -- small,
    # good for testing)
    ./download_goes18_geocolor.py --out ./goes18_geocolor

    # Grab full native resolution instead
    ./download_goes18_geocolor.py --out ./goes18_geocolor --resolution 21696x21696

    # Only download the most recent 20 timestamps
    ./download_goes18_geocolor.py --out ./goes18_geocolor --limit 20

    # Run forever, checking for new images every 5 minutes
    ./download_goes18_geocolor.py --out ./goes18_geocolor --daemon --interval 300
"""

from __future__ import annotations

import concurrent.futures
import re
import signal
import time
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urljoin

import requests
import tyro

BASE_URL = "https://cdn.star.nesdis.noaa.gov/GOES18/ABI/FD/GEOCOLOR/"

# Resolutions NOAA publishes for GOES-18 Full Disk GeoColor, smallest to
# largest (confirmed against the live directory listing). "highest" is a
# synthetic option meaning "whatever the biggest available file is" --
# useful since the largest resolution occasionally lags a minute or two
# behind the smaller ones, or is occasionally missing outright.
RESOLUTIONS = ["339x339", "678x678", "1808x1808", "5424x5424", "10848x10848", "21696x21696"]
ResolutionChoice = Literal[
    "339x339", "678x678", "1808x1808", "5424x5424", "10848x10848", "21696x21696", "highest", "1800x1080"
]

# Matches an optional trailing "-WIDTHxHEIGHT" right before the extension.
RES_RE = re.compile(r"-(\d+)x(\d+)(?=\.\w+$)")

CHUNK_SIZE = 1 << 16  # 64 KiB


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


class LinkParser(HTMLParser):
    """Minimal HTML parser to pull href values out of an Apache/nginx-style
    directory listing, without needing BeautifulSoup as a dependency."""

    def __init__(self):
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            for name, value in attrs:
                if name.lower() == "href" and value:
                    self.hrefs.append(value)


def fetch_listing(url: str, session: requests.Session) -> list[str]:
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    parser = LinkParser()
    parser.feed(resp.text)

    files = []
    for href in parser.hrefs:
        if href.startswith("?") or href in ("../", "./"):
            continue
        if href.startswith("http"):
            if not href.startswith(url):
                continue
            name = href[len(url):]
        else:
            name = href
        name = name.strip("/")
        if not name:
            continue
        if not re.search(r"\.(jpg|jpeg|png|gif)$", name, re.IGNORECASE):
            continue
        files.append(name)
    return files


def base_key(filename: str) -> str:
    """Strip a trailing '-WIDTHxHEIGHT' (if present) to get the group key
    that identifies 'the same shot at some resolution'."""
    return RES_RE.sub("", filename)


def resolution_of(filename: str) -> float:
    """Return a comparable resolution score. Files with no WxH suffix are
    assumed to be the native/full-resolution version and score highest."""
    m = RES_RE.search(filename)
    if not m:
        return float("inf")  # no suffix -> full/native resolution
    w, h = int(m.group(1)), int(m.group(2))
    return float(w * h)


def resolution_str_of(filename: str) -> Optional[str]:
    """Return the 'WIDTHxHEIGHT' string encoded in the filename, or None if
    the filename has no such suffix (e.g. a full-res file with no tag)."""
    m = RES_RE.search(filename)
    return f"{m.group(1)}x{m.group(2)}" if m else None


def pick_highest_res(files: list[str]) -> list[str]:
    groups: dict[str, list[str]] = {}
    for f in files:
        groups.setdefault(base_key(f), []).append(f)
    return [max(variants, key=resolution_of) for variants in groups.values()]


def select_by_resolution(files: list[str], resolution: str) -> list[str]:
    """Pick one file per timestamp matching the requested resolution.

    `resolution="highest"` keeps the previous behavior of picking the
    largest available file per timestamp. Any concrete "WIDTHxHEIGHT"
    value only keeps files that carry exactly that suffix -- timestamps
    missing that particular size (e.g. the largest resolution sometimes
    lags behind, or occasionally isn't published) are simply skipped.
    """
    if resolution == "highest":
        return pick_highest_res(files)
    return [f for f in files if resolution_str_of(f) == resolution]


def remote_size(session: requests.Session, url: str) -> Optional[int]:
    """HEAD the URL to get Content-Length, like wget does before deciding
    whether/how much to fetch."""
    try:
        resp = session.head(url, timeout=15, allow_redirects=True)
        resp.raise_for_status()
        length = resp.headers.get("Content-Length")
        return int(length) if length is not None else None
    except (requests.RequestException, TypeError, ValueError):
        return None


def download_one(session: requests.Session, url: str, dest: Path, retries: int = 3) -> str:
    """Download `url` to `dest`, behaving like `wget -c`:

    - If a local file already exists and its size matches the server's
      Content-Length, it's left alone (verified match, no transfer).
    - If the local file is shorter than the server's copy, resume via a
      Range request appended to the existing bytes.
    - If the local file is longer than (or otherwise doesn't fit) the
      server's copy, it's discarded and re-downloaded from scratch.
    - After transfer, the resulting file size is verified against the
      expected total size.
    """
    expected_total = remote_size(session, url)

    if dest.exists():
        local_size = dest.stat().st_size
        if expected_total is not None and local_size == expected_total:
            return f"SKIP (verified match): {dest.name}"
        elif expected_total is not None and 0 < local_size < expected_total:
            pass  # will resume below
        else:
            # Local file is bigger than remote, or remote size unknown, or
            # local file is empty -- safest to restart clean.
            dest.unlink()

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            resume_from = dest.stat().st_size if dest.exists() else 0
            headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
            mode = "ab" if resume_from else "wb"

            with session.get(url, headers=headers, stream=True, timeout=60) as resp:
                if resume_from and resp.status_code == 200:
                    # Server ignored our Range request (doesn't support resume) --
                    # it's sending the whole file, so start over from scratch.
                    resume_from = 0
                    mode = "wb"
                elif resume_from and resp.status_code != 206:
                    resp.raise_for_status()

                resp.raise_for_status()

                tmp = dest.with_suffix(dest.suffix + ".part")
                if mode == "ab" and dest.exists():
                    dest.rename(tmp)
                with open(tmp, mode) as f:
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                tmp.replace(dest)

            final_size = dest.stat().st_size
            if expected_total is not None and final_size != expected_total:
                last_err = RuntimeError(
                    f"size mismatch after download: got {final_size}, expected {expected_total}"
                )
                # Corrupt/incomplete -- drop it and retry from scratch next loop.
                dest.unlink(missing_ok=True)
                time.sleep(1.5 * attempt)
                continue

            verb = "RESUMED" if resume_from else "OK"
            return f"{verb}: {dest.name}"
        except requests.RequestException as e:
            last_err = e
            time.sleep(1.5 * attempt)

    return f"FAILED: {dest.name} ({last_err})"


# Set by the SIGINT/SIGTERM handler so a running daemon loop can exit
# cleanly between cycles (and mid-sleep) instead of being killed abruptly.
_shutdown_requested = False


def _handle_shutdown_signal(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    log(f"Received signal {signum}, will stop after the current cycle...")


def run_once(
    url: str,
    out: Path,
    resolution: str,
    limit: Optional[int],
    workers: int,
    session: requests.Session,
    quiet_skips: bool,
) -> tuple[int, int, int, int]:
    """Fetch the listing, pick files matching `resolution`, download/verify each.
    Returns (ok, resumed, skip, fail) counts."""
    log(f"Fetching directory listing: {url}")
    try:
        files = fetch_listing(url, session)
    except requests.RequestException as e:
        log(f"ERROR: could not fetch listing: {e}")
        return (0, 0, 0, 0)

    if not files:
        log("No image files found in listing. The page structure may have changed.")
        return (0, 0, 0, 0)

    log(f"Found {len(files)} total image files in listing.")

    selected = select_by_resolution(files, resolution)
    selected.sort()  # filenames are timestamp-prefixed, so this sorts chronologically

    if not selected:
        log(f"No files matched resolution '{resolution}'. Available resolutions: {', '.join(RESOLUTIONS)}")
        return (0, 0, 0, 0)

    if limit:
        selected = selected[-limit:]

    tasks = [(urljoin(url, name), out / name) for name in selected]
    log(f"Selected {len(tasks)} file(s) at resolution '{resolution}' to check/download.")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(download_one, session, u, d): d.name for u, d in tasks}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            result = fut.result()
            results.append(result)
            if not (quiet_skips and result.startswith("SKIP")):
                log(f"[{i}/{len(tasks)}] {result}")

    ok = sum(1 for r in results if r.startswith("OK"))
    resumed = sum(1 for r in results if r.startswith("RESUMED"))
    skip = sum(1 for r in results if r.startswith("SKIP"))
    fail = sum(1 for r in results if r.startswith("FAILED"))
    return (ok, resumed, skip, fail)


def main(
    out: Path = Path("goes18_geocolor"),
    resolution: ResolutionChoice = "678x678",
    url: str = BASE_URL,
    limit: Optional[int] = None,
    workers: int = 4,
    daemon: bool = False,
    interval: int = 300,
) -> None:
    """Download GOES-18 ABI Full Disk GeoColor images.

    Args:
        out: Output directory.
        resolution: Which resolution to download. One of: 339x339 (~120 KB,
            good for quick tests), 678x678 (~420 KB, default -- still small
            but has visible detail), 1808x1808 (~2.3 MB), 5424x5424
            (~15 MB), 10848x10848 (~16 MB), 21696x21696 (~45-50 MB, the
            largest available). "highest" picks whatever the biggest file
            published for each timestamp happens to be.
        url: Directory listing URL to scrape.
        limit: Only download the N most recent timestamps (by filename sort).
        workers: Parallel download workers.
        daemon: Keep running, polling the listing repeatedly instead of
            exiting after one pass.
        interval: Seconds to wait between polls in --daemon mode.
    """
    out.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; goes18-downloader/1.0)"})

    if not daemon:
        ok, resumed, skip, fail = run_once(
            url, out, resolution, limit, workers, session, quiet_skips=False
        )
        log(f"Done. Downloaded: {ok}, Resumed: {resumed}, Already had: {skip}, Failed: {fail}")
        log(f"Files saved to: {out.resolve()}")
        return

    # --- Daemon mode ---
    signal.signal(signal.SIGINT, _handle_shutdown_signal)
    signal.signal(signal.SIGTERM, _handle_shutdown_signal)

    log(f"Starting daemon mode. Resolution: {resolution}. Polling every {interval}s. Press Ctrl+C to stop.")
    cycle = 0
    while not _shutdown_requested:
        cycle += 1
        log(f"--- Cycle {cycle} ---")
        try:
            ok, resumed, skip, fail = run_once(
                url, out, resolution, limit, workers, session, quiet_skips=True
            )
            log(f"Cycle {cycle} done. New: {ok}, Resumed: {resumed}, Already had: {skip}, Failed: {fail}")
        except Exception as e:
            # Never let a transient error (network blip, parsing hiccup, etc.)
            # kill the daemon -- log it and keep going.
            log(f"ERROR during cycle {cycle}: {e}")

        if _shutdown_requested:
            break

        # Sleep in small increments so a shutdown signal is honored promptly
        # instead of waiting out the full interval.
        slept = 0
        while slept < interval and not _shutdown_requested:
            time.sleep(min(1, interval - slept))
            slept += 1

    log("Daemon stopped.")


if __name__ == "__main__":
    tyro.cli(main)


