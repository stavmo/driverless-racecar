"""
path_planning.py- finding the middle of the road

Two planners:

1) local_centerline()  - used on the explore lap, when the car only knows
   the cones it has seen so far.
     * pair every blue cone with its nearest yellow cone (and vice versa);
       the point halfway between them is in the middle of the road,
     * connect these midpoints into a path starting at the car, always
       taking the closest next midpoint that is roughly in front of us.

2) global_centerline() - used after lap 1, when the whole track is mapped.
   Same midpoint idea, for the entire map, then:
     * put the midpoints in driving order (using the path we drove on lap 1),
     * fit a smooth closed curve through them (spline),
     * measure how much room there is to the left/right at each point
       (used for the track statistics: how wide the road is everywhere).
"""
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree

BLUE, YELLOW = 0, 1
MAX_TRACK_WIDTH = 40.0  # [m] never pair cones further apart than this


def cone_midpoints(blue, yellow, max_width=MAX_TRACK_WIDTH):
    """Midpoints between each cone and the nearest cone of the other color."""
    if len(blue) == 0 or len(yellow) == 0:
        return np.zeros((0, 2))
    mids = []
    for a, b in ((blue, yellow), (yellow, blue)):
        d, j = cKDTree(b).query(a)
        ok = d < max_width
        mids.append((a[ok] + b[j[ok]]) / 2)
    return np.vstack(mids)


def local_centerline(cone_map, x, y, yaw, horizon=45.0, max_step=14.0, max_turn_deg=55.0):
    """Short path (list of points) through the middle of the road ahead of the car."""
    pos = np.array([x, y])
    heading = np.array([np.cos(yaw), np.sin(yaw)])
    blue = cone_map.cones_of(BLUE)
    yellow = cone_map.cones_of(YELLOW)

    def nearby_ahead(c):
        if len(c) == 0:
            return c
        rel = c - pos
        return c[(np.linalg.norm(rel, axis=1) < horizon) & (rel @ heading > -5.0)]

    mids = cone_midpoints(nearby_ahead(blue), nearby_ahead(yellow))
    path = [pos]
    direction = heading
    used = np.zeros(len(mids), bool)
    cos_limit = np.cos(np.radians(max_turn_deg))
    for _ in range(40):
        rel = mids - path[-1]
        dist = np.linalg.norm(rel, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_angle = (rel @ direction) / dist
        ok = (~used) & (dist > 1.0) & (dist < max_step) & (cos_angle > cos_limit)
        used |= dist <= 1.0            # drop near-duplicates of the current point
        if not ok.any():
            break
        k = int(np.argmin(np.where(ok, dist, np.inf)))
        used[k] = True
        direction = rel[k] / dist[k]
        path.append(mids[k])
    return np.array(path)


def resample_closed(points, spacing):
    closed = np.vstack([points, points[:1]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], max(3, int(round(s[-1] / spacing))), endpoint=False)  # exactly even spacing
    return np.column_stack([np.interp(t, s, closed[:, 0]), np.interp(t, s, closed[:, 1])])


def global_centerline(cone_map, lap1_trajectory, spacing=2.0, smoothing=4.0):
    """Closed, smooth centerline of the whole mapped track + room on each side."""
    blue = cone_map.cones_of(BLUE)
    yellow = cone_map.cones_of(YELLOW)
    mids = cone_midpoints(blue, yellow)

    # order midpoints by where they are along the lap-1 driven path
    traj = np.asarray(lap1_trajectory)
    d, idx = cKDTree(traj).query(mids)
    mids, idx = mids[d < 15.0], idx[d < 15.0]           # drop bad pairings
    order = np.argsort(idx, kind="stable")
    mids = mids[order]
    mids = mids[np.concatenate([[True], np.linalg.norm(np.diff(mids, axis=0), axis=1) > 0.5])]

    # smooth closed spline (s = how much we allow it to deviate from noisy points)
    pts = np.vstack([mids, mids[:1]])
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=smoothing * len(pts), per=1)
    dense = np.column_stack(splev(np.linspace(0, 1, 20 * len(pts), endpoint=False), tck))
    center = resample_closed(dense, spacing)

    width_left = cKDTree(blue).query(center)[0]     # distance to blue (left) cones
    width_right = cKDTree(yellow).query(center)[0]  # distance to yellow (right) cones
    return center, width_left, width_right


def curvature_closed(path):
    """Signed curvature (1/radius) of a closed path. + = turning left."""
    d1 = (np.roll(path, -1, axis=0) - np.roll(path, 1, axis=0)) / 2
    d2 = np.roll(path, -1, axis=0) - 2 * path + np.roll(path, 1, axis=0)
    num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    den = np.hypot(d1[:, 0], d1[:, 1]) ** 3 + 1e-9
    return num / den


def curvature_open(path):
    if len(path) < 3:
        return np.zeros(len(path))
    d1 = np.gradient(path, axis=0)
    d2 = np.gradient(d1, axis=0)
    num = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
    den = np.hypot(d1[:, 0], d1[:, 1]) ** 3 + 1e-9
    return num / den

