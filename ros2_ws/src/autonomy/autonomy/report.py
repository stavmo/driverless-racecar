"""
report.py - turns the recorded run into results: numbers (summary.json),
a csv log, and the charts used in the README.

Charts:
  track_stats.png       dataset statistics: track width and bend sharpness along the lap
  lap_times.png         lap time of each lap
  track_map.png         lap 1 vs lap 2 driven path, colored by speed
  speed_vs_distance.png actual speed along the lap, lap 1 vs lap 2
  mapping_accuracy.png  how far each mapped cone is from its true position
"""
import csv
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.cm import ScalarMappable         # noqa: E402
from scipy.spatial import cKDTree                # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
LAP_COLORS = {1: "#1baf7a", 2: "#2a78d6"}   # lap 1 aqua, lap 2 blue
CONE_BLUE, CONE_YELLOW = "#2a78d6", "#eda100"
SPEED_CMAP = LinearSegmentedColormap.from_list("speed", ["#e9e4fb", "#6250d6", "#1f1663"])
LAP_NAMES = {1: "Lap 1 · explore", 2: "Lap 2 · fast lap"}


def _style(ax, title=None):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold")


def _fig(w, h):
    return plt.figure(figsize=(w, h), facecolor=SURFACE, dpi=110)


def _laps_from_log(log):
    """Split the brain's log into laps (brain lap counter: 0 = lap 1, 1 = lap 2, ...)."""
    arr = np.array([r[:9] for r in log], dtype=float)
    laps = {}
    for k in (0, 1):
        rows = arr[arr[:, 8] == k]
        if len(rows) < 2:
            continue
        xy = rows[:, 1:3]
        dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        laps[k + 1] = dict(t=rows[:, 0], xy=xy, v=rows[:, 4], dist=dist, xte=rows[:, 7])
    return laps


def mapping_accuracy(map_xy, map_color, truth):
    """Distance from each mapped cone to the nearest TRUE cone of the same color."""
    errors, unmatched_truth = [], 0
    for color in (0, 1, 2):
        m = map_xy[map_color == color]
        t = truth[truth[:, 2] == color, :2]
        if len(m) == 0 or len(t) == 0:
            continue
        errors.append(cKDTree(t).query(m)[0])
        d_truth, _ = cKDTree(m).query(t)
        unmatched_truth += int(np.sum(d_truth > 1.0))
    errors = np.concatenate(errors) if errors else np.zeros(0)
    return errors, unmatched_truth


def generate_report(data, results_dir, sim_summary=None, truth=None, log=print):
    os.makedirs(results_dir, exist_ok=True)
    plans = data["plans"]
    laps = _laps_from_log(data["log"])
    map_xy, map_color = np.asarray(data["map_xy"]), np.asarray(data["map_color"])
    summary = {}

    # -------- csv log of every control step --------
    with open(os.path.join(results_dir, "run_log.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "yaw", "speed", "steering", "accel", "cross_track_err", "lap", "mode"])
        for r in data["log"]:
            w.writerow([f"{r[0]:.2f}", f"{r[1]:.3f}", f"{r[2]:.3f}", f"{r[3]:.4f}", f"{r[4]:.3f}",
                        f"{r[5]:.4f}", f"{r[6]:.3f}", f"{r[7]:.3f}", r[8], r[9]])

    # -------- numbers --------
    if sim_summary:
        summary["official"] = sim_summary
    summary["brain"] = {
        "mapped_cones": int(len(map_xy)),
        "planned_lap_time_centerline_s": _planned_time(plans["center"], plans["center_speed"]),
        "centerline_length_m": _loop_length(plans["center"]),
        "max_cross_track_error_m": {LAP_NAMES[k]: round(float(l["xte"].max()), 2)
                                    for k, l in laps.items() if k > 1},
        "top_speed_kmh": round(float(max(l["v"].max() for l in laps.values()) * 3.6), 1),
    }
    if truth is not None and len(truth):
        err, missed = mapping_accuracy(map_xy, map_color, truth)
        good = err[err < 1.0]
        summary["mapping"] = {
            "true_cones": int(len(truth)), "mapped_cones": int(len(map_xy)),
            "mean_error_cm": round(float(good.mean() * 100), 1),
            "p95_error_cm": round(float(np.percentile(good, 95) * 100), 1),
            "duplicate_or_ghost_cones": int(np.sum(err >= 1.0)),
            "true_cones_never_mapped": missed,
        }
        _plot_mapping(good, os.path.join(results_dir, "mapping_accuracy.png"))
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log(f"summary: {json.dumps(summary)}")

    # -------- charts --------
    _plot_track_stats(plans, os.path.join(results_dir, "track_stats.png"))
    lap_times = (sim_summary or {}).get("lap_times_s") or [float(l["t"][-1] - l["t"][0]) for l in laps.values()]
    _plot_lap_times(lap_times, os.path.join(results_dir, "lap_times.png"))
    _plot_track_map(laps, map_xy, map_color, lap_times, os.path.join(results_dir, "track_map.png"))
    _plot_speed(laps, os.path.join(results_dir, "speed_vs_distance.png"))
    return summary


def _loop_length(p):
    return round(float(np.sum(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1))), 1)


def _planned_time(path, v):
    ds = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)
    return round(float(np.sum(ds / np.maximum(v, 0.1))), 2)


def _plot_track_stats(plans, out):
    c, wl, wr = plans["center"], plans["width_left"], plans["width_right"]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])
    radius = 1 / np.maximum(np.abs(plans["center_curv"]), 1e-4)
    fig = _fig(10, 5.6)
    ax1, ax2 = fig.add_subplot(2, 1, 1), fig.add_subplot(2, 1, 2, sharex=None)
    ax1.plot(s, wl + wr, color=LAP_COLORS[2], lw=2)
    _style(ax1, "Track width along the lap")
    ax1.set_ylabel("width [m]", color=INK_2)
    ax2.plot(s, np.minimum(radius, 500), color="#eb6834", lw=2)
    ax2.set_yscale("log")
    ax2.set_yticks([20, 50, 100, 200, 500])
    ax2.set_yticklabels(["20", "50", "100", "200", "500+"])
    ax2.minorticks_off()
    _style(ax2, "Corner radius along the lap (lower = sharper bend, capped at 500 m)")
    ax2.set_ylabel("radius [m]", color=INK_2)
    ax2.set_xlabel("distance along centerline [m]", color=INK_2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def _plot_lap_times(lap_times, out):
    fig = _fig(8, 3.2)
    ax = fig.add_subplot(1, 1, 1)
    names = [LAP_NAMES[i + 1] for i in range(len(lap_times))][::-1]
    vals = lap_times[::-1]
    cols = [LAP_COLORS[i + 1] for i in range(len(lap_times))][::-1]
    bars = ax.barh(names, vals, color=cols, height=0.55, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + max(vals) * 0.01, b.get_y() + b.get_height() / 2, f"{v:.1f} s",
                va="center", color=INK, fontsize=10)
    _style(ax, "Lap times (official, measured by the simulator)")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("seconds", color=INK_2)
    ax.set_xlim(0, max(vals) * 1.15)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def _plot_track_map(laps, map_xy, map_color, lap_times, out):
    """Driven path of lap 1 and lap 2 side by side, colored by the ACTUAL speed."""
    vmax = max(l["v"].max() for l in laps.values()) * 3.6
    norm = Normalize(0, vmax)
    fig = _fig(12, 6.2)
    for k, lap in enumerate((1, 2)):
        if lap not in laps:
            continue
        ax = fig.add_axes([0.02 + 0.45 * k, 0.04, 0.43, 0.84])
        ax.scatter(*map_xy[map_color == 0].T, s=2, color=CONE_BLUE, alpha=0.5, linewidths=0)
        ax.scatter(*map_xy[map_color == 1].T, s=2, color=CONE_YELLOW, alpha=0.6, linewidths=0)
        xy, v = laps[lap]["xy"], laps[lap]["v"] * 3.6
        ax.scatter(xy[:, 0], xy[:, 1], c=v, cmap=SPEED_CMAP, norm=norm, s=3, linewidths=0)
        ax.set_aspect("equal")
        ax.axis("off")
        t = f" · {lap_times[lap - 1]:.1f} s" if len(lap_times) >= lap else ""
        ax.set_title(LAP_NAMES[lap] + t, loc="left", color=INK, fontsize=13, fontweight="bold")
    cax = fig.add_axes([0.92, 0.2, 0.015, 0.55])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=SPEED_CMAP), cax=cax)
    cb.set_label("speed [km/h]", color=INK_2)
    cb.ax.tick_params(colors=INK_2)
    cb.outline.set_visible(False)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def _plot_speed(laps, out):
    fig = _fig(11, 4.2)
    ax = fig.add_subplot(1, 1, 1)
    for k in (2, 1):
        if k in laps:
            ax.plot(laps[k]["dist"], laps[k]["v"] * 3.6, color=LAP_COLORS[k], lw=2, label=LAP_NAMES[k])
    _style(ax, "Speed along the lap: with the full map, lap 2 brakes only where the corners require it")
    ax.set_xlabel("distance driven in the lap [m]", color=INK_2)
    ax.set_ylabel("speed [km/h]", color=INK_2)
    leg = ax.legend(frameon=False, loc="lower right", fontsize=10)
    for txt in leg.get_texts():
        txt.set_color(INK)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def _plot_mapping(err, out):
    fig = _fig(8, 3.6)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(err * 100, bins=40, color=LAP_COLORS[2], edgecolor=SURFACE, linewidth=1)
    _style(ax, f"Mapping accuracy: mapped cone vs. true position (mean {err.mean() * 100:.1f} cm)")
    ax.set_xlabel("position error [cm]", color=INK_2)
    ax.set_ylabel("number of cones", color=INK_2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
