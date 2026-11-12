"""
track.py - turns BrandsHatchLayout.csv file into a track.

The BrandsHatchLayout.csv only has about 156 points per side, spaced ~25 m apart. 
We mark cones every few meters, so we:
  1. read the left and right edge points,
  2. draw a smooth closed curve through each edge (a "periodic spline"),
  3. place a cone every cone_spacing (5) meters along each curve.
Left edge  -> blue cones   (blue on the left)
Right edge -> yellow cones (yellow on the right)
Two orange cones on each side mark the start/finish line.
"""
import csv

import numpy as np
from scipy.interpolate import splev, splprep

BLUE, YELLOW, ORANGE = 0, 1, 2
COLOR_NAMES = {BLUE: "blue", YELLOW: "yellow", ORANGE: "orange"}


def load_edges(csv_path):
    """Read the csv and return two (N, 2) arrays: left edge and right edge."""
    left, right = [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if not row["x"] or not row["y"]:  # empty separator rows
                continue
            point = (float(row["x"]), float(row["y"]))
            (left if row["side"].strip() == "left" else right).append(point)
    return np.array(left), np.array(right)


def closed_spline(points, n_samples=4000):
    """Smooth closed curve through the points, densely sampled."""
    # per=1 (scipy ): the curve is a loop (end connects back to start)
    tck, _ = splprep([points[:, 0], points[:, 1]], s=0, per=1)
    u = np.linspace(0, 1, n_samples, endpoint=False)
    x, y = splev(u, tck)
    return np.column_stack([x, y])


def resample_by_distance(curve, spacing):
    """Pick points along a closed curve that are `spacing` meters apart."""
    closed = np.vstack([curve, curve[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])  # distance along the curve
    n = int(s[-1] // spacing)
    targets = np.linspace(0, s[-1], n, endpoint=False)
    x = np.interp(targets, s, closed[:, 0])
    y = np.interp(targets, s, closed[:, 1])
    return np.column_stack([x, y])


class Track:
    """all the simulator knows about the 'real world' track."""

    def __init__(self, csv_path, cone_spacing=5.0):
        self.left_raw, self.right_raw = load_edges(csv_path)
        self.left = closed_spline(self.left_raw)
        self.right = closed_spline(self.right_raw)

        blue = resample_by_distance(self.left, cone_spacing)
        yellow = resample_by_distance(self.right, cone_spacing)

        # start/finish line: between the first left and first right csv points
        self.start_left = self.left_raw[0]
        self.start_right = self.right_raw[0]
        start_mid = (self.start_left + self.start_right) / 2
        nxt_mid = (self.left_raw[1] + self.right_raw[1]) / 2
        d = nxt_mid - start_mid
        self.start_pose = (start_mid[0], start_mid[1], float(np.arctan2(d[1], d[0])))

        # two orange cones on each side of the start line
        direction = d / np.linalg.norm(d)
        orange = []
        for edge_pt in (self.start_left, self.start_right):
            for offset in (-2.0, 2.0):
                orange.append(edge_pt + direction * offset)
        orange = np.array(orange)

        self.cones = np.vstack([
            np.column_stack([blue, np.full(len(blue), BLUE)]),
            np.column_stack([yellow, np.full(len(yellow), YELLOW)]),
            np.column_stack([orange, np.full(len(orange), ORANGE)]),
        ])  # columns: x, y, color

        # ground-truth centerline, used only by the simulator for judging
        # (off-track detection). The autonomy side never sees it.
        from scipy.spatial import cKDTree
        idx = cKDTree(self.right).query(self.left)[1]
        self.centerline = (self.left + self.right[idx]) / 2
        self.half_width = np.linalg.norm(self.left - self.right[idx], axis=1) / 2

    def stats(self):
        """Numbers for the README 'dataset statistics' section."""
        def length(c):
            return float(np.sum(np.linalg.norm(np.diff(np.vstack([c, c[:1]]), axis=0), axis=1)))
        width = 2 * self.half_width
        return {
            "csv_points_left": int(len(self.left_raw)),
            "csv_points_right": int(len(self.right_raw)),
            "left_edge_length_m": round(length(self.left), 1),
            "right_edge_length_m": round(length(self.right), 1),
            "centerline_length_m": round(length(self.centerline), 1),
            "track_width_min_m": round(float(width.min()), 1),
            "track_width_mean_m": round(float(width.mean()), 1),
            "track_width_max_m": round(float(width.max()), 1),
            "blue_cones": int(np.sum(self.cones[:, 2] == BLUE)),
            "yellow_cones": int(np.sum(self.cones[:, 2] == YELLOW)),
            "orange_cones": int(np.sum(self.cones[:, 2] == ORANGE)),
        }
