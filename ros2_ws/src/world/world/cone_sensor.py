"""
cone_sensor.py - pretends to be the car's perception system (like a camera + LiDAR).

    * range limit      - only cones closer than `max_range`
    * field of view    - only cones within +-fov/2 of straight ahead
    * miss probability - each cone has a chance of not being detected
    * noise            - measured position is off by a few cm (more when far)
Detections are reported in the car's coordinate frame:
    x = meters in front of the car, y = meters to the left.
"""
import numpy as np


class ConeSensor:
    def __init__(self, cones, max_range=40.0, fov_deg=150.0,
                 detect_prob=0.95, noise_base=0.02, noise_per_m=0.005, seed=0):
        self.cones = cones                      # (N, 3): x, y, color (world frame)
        self.max_range = max_range
        self.half_fov = np.radians(fov_deg) / 2
        self.detect_prob = detect_prob
        self.noise_base = noise_base
        self.noise_per_m = noise_per_m
        self.rng = np.random.default_rng(seed)  # fixed seed gives reproducible runs

    def measure(self, x, y, yaw):
        # 1. vector from car to every cone, rotated into the car's frame
        dx = self.cones[:, 0] - x
        dy = self.cones[:, 1] - y
        c, s = np.cos(-yaw), np.sin(-yaw)
        local_x = c * dx - s * dy
        local_y = s * dx + c * dy

        # 2. keep only cones the sensor could physically see
        dist = np.hypot(local_x, local_y)
        bearing = np.arctan2(local_y, local_x)
        visible = (dist < self.max_range) & (np.abs(bearing) < self.half_fov)
        visible &= self.rng.random(len(dist)) < self.detect_prob

        # 3. add measurement noise (bigger for far cones)
        sigma = self.noise_base + self.noise_per_m * dist[visible]
        meas_x = local_x[visible] + self.rng.normal(0, 1, visible.sum()) * sigma
        meas_y = local_y[visible] + self.rng.normal(0, 1, visible.sum()) * sigma
        colors = self.cones[visible, 2].astype(int)
        return np.column_stack([meas_x, meas_y, colors])
