"""
cone_map.py - the car's memory of where the cones are (MAPPING).

Every time the sensor reports cones, we:
  1. convert each detection from the car's frame to the world frame
     (rotate by the car's heading, then shift by the car's position),
  2. look for a cone of the same color already in the map within
     `match_radius` meters, deciding "is this the cone I saw before, or a new one?"
  3. if it matches: average the new measurement in (each sighting makes the
     cone's position more accurate, because random noise averages out),
     if not: add it as a new cone.

The average is weighted: the sensor is less precise for far cones (its
error grows with distance), so a sighting from 5 m away counts much more than
one from 35 m away. Weight = 1 / variance = 1 / sigma^2, the standard way to
combine measurements of different quality. Very far detections (beyond
`max_range`) are not used for mapping at all.
"""
import numpy as np


class ConeMap:
    def __init__(self, match_radius=2.5, max_range=32.0, sigma_base=0.02, sigma_per_m=0.005):
        self.match_radius = match_radius
        self.max_range = max_range
        self.sigma_base, self.sigma_per_m = sigma_base, sigma_per_m  # from the sensor's spec
        self.xy = np.zeros((0, 2))     # estimated cone positions
        self.color = np.zeros(0, int)  # cone colors
        self.count = np.zeros(0, int)  # how many times each cone was seen
        self.weight = np.zeros(0)      # total weight (confidence) of each estimate

    @staticmethod
    def to_world(detections, x, y, yaw):
        c, s = np.cos(yaw), np.sin(yaw)
        wx = x + c * detections[:, 0] - s * detections[:, 1]
        wy = y + s * detections[:, 0] + c * detections[:, 1]
        return np.column_stack([wx, wy])

    def update(self, detections, x, y, yaw):
        if len(detections) == 0:
            return
        rng = np.hypot(detections[:, 0], detections[:, 1])
        detections = detections[rng < self.max_range]
        rng = rng[rng < self.max_range]
        world = self.to_world(detections, x, y, yaw)
        colors = detections[:, 2].astype(int)
        weights = 1.0 / (self.sigma_base + self.sigma_per_m * rng) ** 2
        new_xy, new_color, new_w = [], [], []
        for p, col, w in zip(world, colors, weights):
            if len(self.xy):
                d = np.linalg.norm(self.xy - p, axis=1)
                d[self.color != col] = np.inf      # only match same color
                j = int(np.argmin(d))
                if d[j] < self.match_radius:
                    # weighted running average: mean += (sample - mean) * w / total_weight
                    self.count[j] += 1
                    self.weight[j] += w
                    self.xy[j] += (p - self.xy[j]) * w / self.weight[j]
                    continue
            new_xy.append(p)
            new_color.append(col)
            new_w.append(w)
        if new_xy:
            self.xy = np.vstack([self.xy, new_xy])
            self.color = np.concatenate([self.color, new_color])
            self.count = np.concatenate([self.count, np.ones(len(new_xy), int)])
            self.weight = np.concatenate([self.weight, new_w])

    def cones_of(self, color):
        return self.xy[self.color == color]

    def __len__(self):
        return len(self.xy)
