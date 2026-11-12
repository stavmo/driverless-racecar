"""
speed_planning.py- how fast I can go at each point of the path

Step 1- Corner limit. In a bend of radius R the tires must provide a
         sideways (lateral) acceleration of v^2 / R. They can give at most
         a_lat_max, so:   v_corner = sqrt(a_lat_max * R) = sqrt(a_lat_max / |curvature|)

Step 2- Braking limit (backward pass). You must already be slow enough
         before the corner. Walking the path backwards from each corner:
             v[i] <= sqrt(v[i+1]^2 + 2 * a_brake * ds)
         (formula v^2 = v0^2 + 2*a*d).

Step 3- Accelararion limit (forward pass). The engine can only add speed
         so fast coming out of a corner:
             v[i+1] <= sqrt(v[i]^2 + 2 * a_accel * ds)

The result is a speed for every point
(flat out on straights, brake into corners, accelerate out of them).
"""
import numpy as np


def corner_speed(curvature, a_lat_max, v_max):
    with np.errstate(divide="ignore"):
        v = np.sqrt(a_lat_max / np.maximum(np.abs(curvature), 1e-6))
    return np.minimum(v, v_max)


def speed_profile_closed(path, curvature, a_lat_max=12.0, a_accel=6.0, a_brake=10.0, v_max=44.0):
    """Speed profile for a closed loop (the full track)."""
    ds = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)
    v = corner_speed(curvature, a_lat_max, v_max)
    n = len(v)
    for _ in range(2):  # two rounds so the loop 'wraps around' correctly
        for i in range(2 * n):              # forward pass: acceleration limit
            a, b = i % n, (i + 1) % n
            v[b] = min(v[b], np.sqrt(v[a] ** 2 + 2 * a_accel * ds[a]))
        for i in range(2 * n, 0, -1):       # backward pass: braking limit
            a, b = (i - 1) % n, i % n
            v[a] = min(v[a], np.sqrt(v[b] ** 2 + 2 * a_brake * ds[a]))
    return v


def safe_speed_ahead(path, curvature, a_lat_max, a_brake, v_max):
    """For an open path starting at the car: fastest speed right now that still
    lets us brake in time for every bend we can see."""
    if len(path) < 3:
        return 3.0  #can't see the road: creep forward
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    v_corner = corner_speed(curvature, a_lat_max, v_max)
    # can't see past the end of the path-> assume we may need to stop there
    v_corner[-1] = min(v_corner[-1], 3.0)
    allowed_now = np.sqrt(v_corner ** 2 + 2 * a_brake * s)
    return float(min(v_max, allowed_now.min()))
