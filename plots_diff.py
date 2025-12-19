#!/usr/bin/env python3
"""
plots_diff.py

Reads diffs.csv (first row: usernames as header; below: continuous diff values by column), generates per-user PNGs with:
 - full-range histogram
 - zoom-range histogram

Configuration: all adjustable values and plot windows at the top.
"""
from __future__ import annotations

import csv
import math
import os
import re
import sys
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from diff import STREAM_ID

# -------- Parameters --------
CSV_FILE = "diffs.csv"         # Path to diffs CSV
OUTDIR = "plots"               # Output directory for PNGs
MIN_COUNT = 1                 # Minimum samples per user to plot
NO_COMBINED = False            # If True, don't generate combined plot
SHOW_PLOT = False              # If True, call plt.show() after saving figures

FULL_MIN = 0.0                 # Left edge for full plot
FULL_MAX = 20.0                # Right edge for full plot
FULL_MAX_BINS = 40             # Max bins for full plot

ZOOM_BINS = 40                 # Max bins for zoom plot
# zoom values. this varies depending on the stream.
if STREAM_ID == 'OxzHQ546YQY':  # chatvote
    ZOOM_MIN = 6.0
    ZOOM_MAX = 9.0
elif STREAM_ID == '-fesy2kdDxo': # chatvotecombo
    ZOOM_MIN = 10
    ZOOM_MAX = 15
else: # chatvotecombo / unknown
    ZOOM_MIN = 3.0
    ZOOM_MAX = 7.0

CHAT_COOLDOWN = 3.0            # X value for vertical line in full plot
CHAT_COOLDOWN_LABEL = "Chat Cooldown"
# ----------------------------

def read_csv_columns(filename: str) -> Tuple[List[str], List[List[str]]]:
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")

    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return [], []

        columns: List[List[str]] = [[] for _ in header]
        for row in reader:
            for i in range(len(header)):
                cell = row[i] if i < len(row) else ""
                columns[i].append(cell if cell is not None else "")
    return header, columns

def parse_column_to_floats(col: List[str]) -> List[float]:
    out: List[float] = []
    for s in col:
        if s is None:
            continue
        t = str(s).strip()
        if t == "":
            continue
        try:
            v = float(t)
            out.append(v)
            continue
        except ValueError:
            pass
        try:
            v = float(t.replace(",", "."))
            out.append(v)
        except ValueError:
            print(f"Warning: could not parse value: '{t}', skipped.", file=sys.stderr)
            continue
    return out

def safe_filename(s: str) -> str:
    s2 = s.strip()
    s2 = re.sub(r"[^A-Za-z0-9._-]+", "_", s2)
    if not s2:
        s2 = "user"
    return s2

def auto_bin_count(n_samples: int, max_bins: int) -> int:
    return min(max_bins, max(10, int(np.sqrt(n_samples)*2)))

def plot_user_distribution(username: str,
                           arr: np.ndarray,
                           outpath: Path,
                           show_plot: bool = SHOW_PLOT) -> None:
    if arr.size == 0:
        raise ValueError("No data to plot.")

    mu = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=0))

    # --- Full histogram ---
    full_min = FULL_MIN
    full_max = FULL_MAX
    n_bins_full = auto_bin_count(len(arr), FULL_MAX_BINS)
    counts_full, edges_full = np.histogram(arr, bins=n_bins_full, range=(full_min, full_max), density=True)
    centers_full = 0.5 * (edges_full[:-1] + edges_full[1:])
    mode_x = centers_full[np.argmax(counts_full)] if counts_full.size > 0 else np.nan

    # --- Zoom histogram ---
    zoom_min = ZOOM_MIN
    zoom_max = ZOOM_MAX
    arr_in_zoom = arr[(arr >= zoom_min) & (arr <= zoom_max)]
    if arr_in_zoom.size == 0:
        arr_in_zoom = arr
    counts_zoom, edges_zoom = np.histogram(arr_in_zoom, bins=ZOOM_BINS, range=(zoom_min, zoom_max), density=True)
    centers_zoom = 0.5 * (edges_zoom[:-1] + edges_zoom[1:])

    # --- Plotting ---
    fig, axes = plt.subplots(ncols=2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
    ax_full, ax_zoom = axes

    # Full
    ax_full.hist(arr, bins=n_bins_full, range=(full_min, full_max), density=True, alpha=0.65, color="#4C72B0", edgecolor="black",
                 label=f"Histogram (bins={n_bins_full})")
    ax_full.axvline(CHAT_COOLDOWN, color="red", lw=2, linestyle=":", label=CHAT_COOLDOWN_LABEL)
    ax_full.axvline(mode_x, color="k", lw=1.2, linestyle=":", label="Mode")
    ax_full.text(mode_x, ax_full.get_ylim()[1] * 0.9, "Mode", rotation=90, verticalalignment="top",
                 horizontalalignment="right", fontsize=9, color="k")
    ax_full.set_xlabel("Time Difference (seconds)")
    ax_full.set_ylabel("Density")
    ax_full.set_xlim(full_min, full_max)
    ax_full.set_title(f"{username} — Full Distribution (N={len(arr)})")
    ax_full.grid(alpha=0.25)
    ax_full.legend()

    # Zoom
    bar_widths = (edges_zoom[1:] - edges_zoom[:-1]) * 0.85
    ax_zoom.bar(centers_zoom, counts_zoom, width=bar_widths, align="center", alpha=0.75,
                color="#4C72B0", edgecolor="black", label=f"Histogram (bins={ZOOM_BINS})")
    ax_zoom.set_xlim(zoom_min, zoom_max)
    ax_zoom.set_xlabel("Time Difference (seconds)")
    ax_zoom.set_title(f"{username} - Zoom [{ZOOM_MIN}s to {ZOOM_MAX}s]")
    stats = f"N={len(arr)}\nμ={mu:.3f}\nσ={sigma:.3f}\nBins(full)={n_bins_full}\nBins(zoom)={ZOOM_BINS}"
    ax_zoom.text(0.98, 0.95, stats, transform=ax_zoom.transAxes, fontsize=9,
                 verticalalignment="top", horizontalalignment="right",
                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax_zoom.legend()

    plt.suptitle(f"Time Differences for '{username}'", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=150)
    print(f"Saved: {outpath}")
    if show_plot:
        try:
            plt.show()
        except Exception:
            pass
    plt.close(fig)

def plot_combined(users_data: Dict[str, np.ndarray],
                  outpath: Path,
                  show_plot: bool = SHOW_PLOT) -> None:
    if not users_data:
        return

    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab10.colors
    i = 0
    modes = {}
    for username, arr in users_data.items():
        if arr.size == 0:
            continue
        bins = auto_bin_count(len(arr), FULL_MAX_BINS)
        counts, bin_edges = np.histogram(arr, bins=bins, range=(FULL_MIN, FULL_MAX), density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        mode_x = bin_centers[np.argmax(counts)] if counts.size > 0 else np.nan
        plt.step(bin_centers, counts, where="mid", color=colors[i % len(colors)], lw=1.8, label=username)
        modes[username] = mode_x
        i += 1

    for idx, (username, mode_x) in enumerate(modes.items()):
        plt.axvline(mode_x, color="k", lw=0.6, linestyle="--")
        plt.text(mode_x, plt.ylim()[1] * (0.9 - 0.03 * (idx % 6)), username, rotation=90, fontsize=8, verticalalignment="top")

    plt.xlabel("Time Difference (seconds)")
    plt.ylabel("Density")
    plt.xlim(FULL_MIN, FULL_MAX)
    plt.title("Comparison of User Histograms")
    plt.grid(alpha=0.2)
    plt.legend(fontsize=8, loc="upper right")
    plt.tight_layout()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=150)
    print(f"Saved: {outpath}")
    if show_plot:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()

def main() -> None:
    try:
        header, raw_columns = read_csv_columns(CSV_FILE)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(2)

    users_data: Dict[str, np.ndarray] = {}
    for name, col in zip(header, raw_columns):
        floats = parse_column_to_floats(col)
        users_data[name] = np.array(floats, dtype=float)

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    plotted_users = 0
    for name, arr in users_data.items():
        if arr.size < MIN_COUNT:
            print(f"Skipped '{name}': only {arr.size} measurements (<{MIN_COUNT})")
            continue
        fname = safe_filename(name)
        outpath = outdir / f"diffs_{fname}.png"
        try:
            plot_user_distribution(name, arr, outpath, show_plot=SHOW_PLOT)
            plotted_users += 1
        except Exception as e:
            print(f"Error while plotting '{name}': {e}", file=sys.stderr)

    if plotted_users == 0:
        print("No users with sufficient measurements for plotting found.", file=sys.stderr)
    else:
        print(f"Created {plotted_users} per-user plots in: {outdir}")

    if not NO_COMBINED:
        combined_path = outdir / "diffs_all_users.png"
        combined_data = {n: a for n, a in users_data.items() if a.size >= MIN_COUNT}
        if combined_data:
            plot_combined(combined_data, combined_path, show_plot=SHOW_PLOT)
        else:
            print("No combined plot generated: no users with sufficient measurements.", file=sys.stderr)

if __name__ == "__main__":
    main()