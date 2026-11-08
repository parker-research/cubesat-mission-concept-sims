#!/usr/bin/env python3
"""
Yoshimura Spiral Crease Pattern Generator
==========================================

Generates a printable crease pattern for a Yoshimura fold:
  - Solid diagonal lines forming a diamond/crosshatch mesh -> MOUNTAIN folds
  - Dashed horizontal lines at each row boundary            -> VALLEY folds,
    running parallel to the paper's long axis (a.k.a. "hotdog" folds)
  - Dotted vertical lines at internal column boundaries     -> also folds,
    running parallel to the paper's short axis (a.k.a. "hamburger" folds)
  - Solid border around the whole sheet
  - Column numbers along the top, row numbers down the left side

Defaults are set to match a landscape layout: 5 columns x 8 rows on US
Letter paper turned sideways (11" x 8.5"), so the diamonds are long along
the same axis as the paper (horizontal), with 7 hotdog folds (the 7
internal row boundaries) and 4 hamburger folds (the 4 internal column
boundaries). Every parameter below can be changed.

Usage:
    python3 yoshimura_pattern.py
    python3 yoshimura_pattern.py --cols 5 --rows 8 --out my_pattern.pdf
    python3 yoshimura_pattern.py --orientation portrait --rows 16
"""

# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "matplotlib",
# ]
# ///

import argparse
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Page size presets (width, height) in inches, portrait orientation.
# Use `orientation="landscape"` to swap width/height at draw time.
# ---------------------------------------------------------------------------
PAGE_SIZES = {
    "letter": (8.5, 11.0),
    "a4": (8.27, 11.69),
    "legal": (8.5, 14.0),
}


def draw_yoshimura_pattern(
    cols=5,
    rows=8,
    page_size="letter",
    orientation="landscape",
    margin=0.0,
    mountain_lw=1.3,
    valley_lw=0.9,
    border_lw=1.8,
    vertical_lines=4,
    vertical_lw=0.7,
    label_fontsize_top=14,
    label_fontsize_side=11,
    show_labels=False,
    out_path="yoshimura_pattern.pdf",
):
    """
    Draw a Yoshimura crease pattern and save it to `out_path`.

    Parameters
    ----------
    cols, rows : int
        Number of columns / rows of diamond cells in the grid. Columns run
        across the width, rows run down the height. In the default
        landscape orientation, width is the paper's long axis, so cols=5,
        rows=8 makes each diamond cell wider than it is tall (i.e. the
        diamonds are long along the same axis as the paper).
    page_size : str or (float, float)
        Either a key into PAGE_SIZES ("letter", "a4", "legal") or an
        explicit (width_in, height_in) tuple, given in portrait
        (width, height) order regardless of `orientation`.
    orientation : str
        "portrait" (use page_size as-is) or "landscape" (swap width and
        height so the sheet is turned sideways).
    margin : float
        Margin (inches) left blank on all four sides of the page.
    mountain_lw, valley_lw, border_lw : float
        Line widths for diagonal (mountain), horizontal (valley), and
        border lines respectively.
    vertical_lines : int
        Total number of extra dotted vertical lines drawn across the whole
        grid, evenly spaced by column count so they fall on the internal
        column boundaries (i.e. through the middle of each full diamond).
        These run parallel to the short axis of the paper ("hamburger"
        folds). With the default cols=5, vertical_lines=4 lands exactly on
        the 4 internal column dividers. Set to 0 to disable.

        The dashed horizontal valley lines (one per internal row boundary,
        rows-1 of them) run parallel to the long axis of the paper
        ("hotdog" folds); with the default rows=8 that's 7 of them.
    vertical_lw : float
        Line width for the extra dotted vertical lines.
    show_labels : bool
        Whether to draw the 1..cols / 1..rows index labels.
    out_path : str
        Output file path. Extension determines format (.pdf, .png, .svg, ...).
    """
    if isinstance(page_size, str):
        if page_size not in PAGE_SIZES:
            raise ValueError(f"Unknown page_size '{page_size}'. Choose from {list(PAGE_SIZES)}")
        page_w, page_h = PAGE_SIZES[page_size]
    else:
        page_w, page_h = page_size

    if orientation == "landscape":
        page_w, page_h = page_h, page_w
    elif orientation != "portrait":
        raise ValueError(f"Unknown orientation '{orientation}'. Choose 'portrait' or 'landscape'.")

    grid_w = page_w - 2 * margin
    grid_h = page_h - 2 * margin
    if grid_w <= 0 or grid_h <= 0:
        raise ValueError("Margin is too large for the chosen page size.")

    cell_w = grid_w / cols
    cell_h = grid_h / rows

    x0 = margin
    y_top = page_h - margin  # top edge of the grid (page coords, y grows upward)

    fig = plt.figure(figsize=(page_w, page_h))
    ax = fig.add_axes([0, 0, 1, 1])  # axes fill the whole page, no border padding
    ax.set_xlim(0, page_w)
    ax.set_ylim(0, page_h)
    ax.set_aspect("equal")
    ax.axis("off")

    # ---- Diagonal "mountain" lines: an X inside every cell ----------------
    for r in range(rows):
        row_y_top = y_top - r * cell_h
        row_y_bot = y_top - (r + 1) * cell_h
        for c in range(cols):
            col_x_left = x0 + c * cell_w
            col_x_right = x0 + (c + 1) * cell_w
            ax.plot([col_x_left, col_x_right], [row_y_top, row_y_bot],
                     color="black", lw=mountain_lw, solid_capstyle="round")
            ax.plot([col_x_left, col_x_right], [row_y_bot, row_y_top],
                     color="black", lw=mountain_lw, solid_capstyle="round")

    # ---- Horizontal "valley" lines (dashed), at every internal row line ---
    for r in range(1, rows):
        y = y_top - r * cell_h
        ax.plot([x0, x0 + grid_w], [y, y],
                 color="black", lw=valley_lw, linestyle=(0, (6, 4)))

    # ---- Extra dotted vertical lines, evenly spaced across the whole width -
    if vertical_lines > 0:
        n_strips = vertical_lines + 1
        for k in range(1, n_strips):
            x = x0 + k * (grid_w / n_strips)
            ax.plot([x, x], [y_top, y_top - grid_h],
                     color="black", lw=vertical_lw, linestyle=(0, (1, 3)))

    # ---- Solid outer border -------------------------------------------------
    ax.plot(
        [x0, x0 + grid_w, x0 + grid_w, x0, x0],
        [y_top, y_top, y_top - grid_h, y_top - grid_h, y_top],
        color="black", lw=border_lw,
    )

    # ---- Labels -------------------------------------------------------------
    if show_labels:
        for c in range(cols):
            xc = x0 + (c + 0.5) * cell_w
            ax.text(xc, y_top + margin * 0.28, str(c + 1),
                     ha="center", va="bottom", fontsize=label_fontsize_top)
        for r in range(rows):
            yc = y_top - (r + 0.5) * cell_h
            ax.text(x0 - margin * 0.22, yc, str(r + 1),
                     ha="right", va="center", fontsize=label_fontsize_side)

    fig.savefig(out_path, dpi=300, bbox_inches=None)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Generate a Yoshimura spiral crease pattern.")
    parser.add_argument("--cols", type=int, default=5, help="Number of columns (default: 5)")
    parser.add_argument("--rows", type=int, default=8, help="Number of rows (default: 8)")
    parser.add_argument("--page-size", default="letter", choices=list(PAGE_SIZES.keys()),
                         help="Page size preset, given in portrait order (default: letter)")
    parser.add_argument("--orientation", default="landscape", choices=["portrait", "landscape"],
                         help="Page orientation (default: landscape)")
    parser.add_argument("--margin", type=float, default=0.0, help="Page margin in inches (default: 0.0, i.e. pattern runs edge-to-edge)")
    parser.add_argument("--vertical-lines", type=int, default=4,
                         help="Total extra dotted vertical lines across the whole grid, evenly spaced (default: 4)")
    parser.add_argument("--out", default="yoshimura_pattern.pdf", help="Output file path")
    args = parser.parse_args()

    path = draw_yoshimura_pattern(
        cols=args.cols,
        rows=args.rows,
        page_size=args.page_size,
        orientation=args.orientation,
        margin=args.margin,
        vertical_lines=args.vertical_lines,
        out_path=args.out,
    )
    print(f"Saved crease pattern to {path}")


if __name__ == "__main__":
    main()
