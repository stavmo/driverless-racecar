"""
video.py - turns the recorded run into an animated video.

For each recorded moment - draw one picture, then use
matplotlib's animation to stitch the pictures together.

Left panel  : bird's-eye view of the whole track. Cones appear only once the
              car has seen them, so you watch the map being built on lap 1.
Right panel : camera "chasing" the car's in its own frame,
              showing what the sensor sees right now and the planned path.
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib import animation                       # noqa: E402
from matplotlib.cm import ScalarMappable                # noqa: E402
from matplotlib.collections import LineCollection      # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402
from matplotlib.patches import Polygon, Wedge         # noqa: E402

BG = "#0f1117"
# speed colors: one hue, dim (slow) -> bright (fast) on the dark background
SPEED_CMAP = LinearSegmentedColormap.from_list("speed", ["#2d2463", "#7c6cf0", "#ece9ff"])
CONE_COLORS = {0: "#3b82f6", 1: "#facc15", 2: "#fb923c"}
MODE_COLORS = {"EXPLORE": "#22c55e", "FAST": "#38bdf8", "FINISH": "#a3a3a3", "DONE": "#a3a3a3"}
MODE_TEXT = {"EXPLORE": "LAP 1 · EXPLORING + MAPPING", "FAST": "LAP 2 · FAST LAP ON THE MAP",
             "FINISH": "FINISHED · BRAKING", "DONE": "DONE"}


def _pick_frames(snapshots, explore_dt, fast_dt):
    frames, next_t = [], -1.0
    for s in snapshots:
        if s["t"] >= next_t:
            frames.append(s)
            next_t = s["t"] + (explore_dt if s["mode"] == "EXPLORE" else fast_dt)
    return frames


def _to_car_frame(points, x, y, yaw):
    """World -> car frame rotated so the car points UP the screen."""
    if len(points) == 0:
        return np.zeros((0, 2))
    d = points[:, :2] - [x, y]
    c, s = np.cos(-yaw), np.sin(-yaw)
    fx = c * d[:, 0] - s * d[:, 1]   # forward
    fy = s * d[:, 0] + c * d[:, 1]   # left
    return np.column_stack([-fy, fx])  # screen: x = right, y = forward


def render_video(data, out_path, explore_dt=1.0, fast_dt=0.5, fps=20, dpi=80, view=60.0, car_name=None):
    snaps = _pick_frames(data["snapshots"], explore_dt, fast_dt)
    map_xy, map_color = np.asarray(data["map_xy"]), np.asarray(data["map_color"])
    first_seen = np.asarray(data["map_first_seen"])
    log_t = np.array([r[0] for r in data["log"]])
    log_xy = np.array([[r[1], r[2]] for r in data["log"]])
    log_v = np.array([r[4] for r in data["log"]])
    log_lap = np.array([r[8] for r in data["log"]])
    plans = data.get("plans") or {}

    fig = plt.figure(figsize=(13.5, 7.2), facecolor=BG)
    ax_map = fig.add_axes([0.01, 0.03, 0.56, 0.86], facecolor=BG)
    ax_cam = fig.add_axes([0.60, 0.03, 0.39, 0.86], facecolor=BG)
    for ax in (ax_map, ax_cam):
        ax.set_aspect("equal")
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#2a2f3a")
    pad = 40
    ax_map.set_xlim(map_xy[:, 0].min() - pad, map_xy[:, 0].max() + pad)
    ax_map.set_ylim(map_xy[:, 1].min() - pad, map_xy[:, 1].max() + pad)
    ax_cam.set_xlim(-view / 2, view / 2)
    ax_cam.set_ylim(-view * 0.15, view * 0.85)

    title = fig.text(0.01, 0.95, "", color="white", fontsize=17, fontweight="bold", family="monospace")
    hud = fig.text(0.99, 0.95, "", color="#d1d5db", fontsize=12, family="monospace", ha="right")
    fig.text(0.99, 0.005, "BGRacing Autonomous · Stav Mordekhai's home assignment", color="#4b5563",
             fontsize=8, ha="right")

    # --- static/overview artists ---
    cones_sc = ax_map.scatter([], [], s=4, linewidths=0)
    old_trail, = ax_map.plot([], [], "-", color="#4b5563", lw=0.8, alpha=0.6)
    speed_norm = Normalize(0, 160)  # km/h
    trail = LineCollection([], cmap=SPEED_CMAP, norm=speed_norm, linewidths=2.2)
    ax_map.add_collection(trail)
    car_dot, = ax_map.plot([], [], "o", color="red", ms=7, mec="black", zorder=5)
    center_ln, = ax_map.plot([], [], "--", color="#38bdf8", lw=0.8, alpha=0.6)
    cbar_ax = fig.add_axes([0.03, 0.06, 0.15, 0.015])
    cb = fig.colorbar(ScalarMappable(norm=speed_norm, cmap=SPEED_CMAP), cax=cbar_ax, orientation="horizontal")
    cb.set_label("speed [km/h]", color="#9ca3af", fontsize=8)
    cb.ax.tick_params(colors="#9ca3af", labelsize=7)

    # --- chase camera artists ---
    fov = Wedge((0, 0), 40, 90 - 75, 90 + 75, color="#22c55e", alpha=0.06)
    ax_cam.add_patch(fov)
    known_sc = ax_cam.scatter([], [], s=30, linewidths=0, alpha=0.35)
    det_sc = ax_cam.scatter([], [], s=70, edgecolors="white", linewidths=0.8)
    path_ln, = ax_cam.plot([], [], "-", lw=3, alpha=0.9)
    car_poly = Polygon([[-0.8, -1.0], [0.8, -1.0], [0.8, 2.6], [0, 3.3], [-0.8, 2.6]],
                       closed=True, color="red", zorder=6)
    ax_cam.add_patch(car_poly)
    ax_cam.text(0, -view * 0.12, "car view · sensor range 40 m · big dots = seen right now", color="#6b7280",
                fontsize=9, ha="center", va="bottom")

    def draw(k):
        s = snaps[k]
        t, x, y, yaw, v, mode = s["t"], s["x"], s["y"], s["yaw"], s["v"], s["mode"]
        mcol = MODE_COLORS[mode]
        title.set_text(f"{car_name.upper()} · {MODE_TEXT[mode]}" if car_name else MODE_TEXT[mode])
        title.set_color(mcol)
        hud.set_text(f"t={t:6.1f}s  speed={v * 3.6:5.1f} km/h  cones mapped={s['map_size']:4d}")

        seen = first_seen <= t
        cones_sc.set_offsets(map_xy[seen])
        cones_sc.set_color([CONE_COLORS[c] for c in map_color[seen]])
        upto = np.searchsorted(log_t, t)
        lap_start = np.searchsorted(log_lap, s["lap"])      # first log row of this lap
        old_trail.set_data(log_xy[:lap_start, 0], log_xy[:lap_start, 1])
        pts = log_xy[lap_start:upto]
        if len(pts) > 1:
            trail.set_segments(np.stack([pts[:-1], pts[1:]], axis=1)[::2])
            trail.set_array(log_v[lap_start:upto - 1][::2] * 3.6)
        car_dot.set_data([x], [y])
        if mode != "EXPLORE" and "center" in plans:   # show the planned centerline after lap 1
            c = plans["center"]
            center_ln.set_data(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]])

        # chase camera
        near = np.linalg.norm(map_xy - [x, y], axis=1) < view
        near &= seen
        known_sc.set_offsets(_to_car_frame(map_xy[near], x, y, yaw))
        known_sc.set_color([CONE_COLORS[c] for c in map_color[near]])
        det = s["detections"]
        if len(det):
            det_sc.set_offsets(_to_car_frame(det, x, y, yaw))
            det_sc.set_color([CONE_COLORS[int(c)] for c in det[:, 2]])
        else:
            det_sc.set_offsets(np.zeros((0, 2)))
        p = _to_car_frame(s["path"], x, y, yaw)
        path_ln.set_data(p[:, 0], p[:, 1]) if len(p) else path_ln.set_data([], [])
        path_ln.set_color(mcol)
        return []

    anim = animation.FuncAnimation(fig, draw, frames=len(snaps), blit=False)
    if out_path.endswith(".mp4"):
        anim.save(out_path, writer=animation.FFMpegWriter(fps=fps, bitrate=2500), dpi=dpi)
    else:
        anim.save(out_path, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return len(snaps)
