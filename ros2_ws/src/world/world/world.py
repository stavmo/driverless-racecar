"""
world.py - the simulated world, with no ROS code in it.

Keeping the logic separate from ROS means we can test it with Python
(tools/offline_run.py) and the ROS node (world_node.py) is just a thin wrapper
that moves messages in and out.

The world also acts as the race official: it times laps, counts knocked-over
cones (usually +2 s per cone) and counts off-track moments.
"""
import math

import numpy as np
from scipy.spatial import cKDTree

from .cone_sensor import ConeSensor
from .track import Track
from .vehicle import Vehicle, VehicleParams

CONE_HIT_RADIUS = 1.0   # [m] car body touching a cone
CONE_PENALTY_S = 2.0    # rule: +2 s per cone down


class World:
    def __init__(self, csv_path, seed=0, sensor_kwargs=None, odom_noise=(0.03, 0.002)):
        self.track = Track(csv_path)
        x0, y0, yaw0 = self.track.start_pose
        # start 6 m behind the line so lap 1 begins with a line crossing
        self.car = Vehicle(x0 - 6 * math.cos(yaw0), y0 - 6 * math.sin(yaw0), yaw0, VehicleParams())
        self.sensor = ConeSensor(self.track.cones, seed=seed, **(sensor_kwargs or {}))
        self.rng = np.random.default_rng(seed + 1)
        self.odom_noise = odom_noise
        self.t = 0.0

        self._cone_tree = cKDTree(self.track.cones[:, :2])
        self._center_tree = cKDTree(self.track.centerline)
        self.hit_cones = set()
        self.off_track_time = 0.0

        # lap timing
        sl, sr = self.track.start_left, self.track.start_right
        self._line_a, self._line_b = np.array(sl), np.array(sr)
        self._prev_side = self._side_of_line(self.car.x, self.car.y)
        self.lap_start_times = []   # time of each line crossing
        self.lap_times = []
        self.lap_cone_hits = []
        self._hits_at_lap_start = 0

    # ---------- physics ----------
    def step(self, steer_cmd, accel_cmd, dt):
        self.car.step(steer_cmd, accel_cmd, dt)
        self.t += dt
        self._judge(dt)

    def _side_of_line(self, x, y):
        a, b = self._line_a, self._line_b
        return np.sign((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]))

    def _judge(self, dt):
        car = self.car
        # cones touched
        for i in self._cone_tree.query_ball_point([car.x, car.y], CONE_HIT_RADIUS):
            self.hit_cones.add(i)
        # off track: further from the centerline than the half width
        d, i = self._center_tree.query([car.x, car.y])
        if d > self.track.half_width[i]:
            self.off_track_time += dt
        # start/finish line crossing (only counts near the line itself)
        side = self._side_of_line(car.x, car.y)
        near = np.linalg.norm(np.array([car.x, car.y]) - (self._line_a + self._line_b) / 2) < 25
        if near and side != self._prev_side and side != 0:
            if self.lap_start_times:
                self.lap_times.append(self.t - self.lap_start_times[-1])
                self.lap_cone_hits.append(len(self.hit_cones) - self._hits_at_lap_start)
            self.lap_start_times.append(self.t)
            self._hits_at_lap_start = len(self.hit_cones)
        if side != 0:
            self._prev_side = side

    # ---------- sensors ----------
    def odometry(self):
        """Car pose and speed as the car's own localization reports it (a little noisy)."""
        pos_sigma, yaw_sigma = self.odom_noise
        c = self.car
        return (c.x + self.rng.normal(0, pos_sigma), c.y + self.rng.normal(0, pos_sigma),
                c.yaw + self.rng.normal(0, yaw_sigma), c.v)

    def detect_cones(self):
        c = self.car
        return self.sensor.measure(c.x, c.y, c.yaw)

    def summary(self):
        return {
            "lap_times_s": [round(t, 2) for t in self.lap_times],
            "cones_hit_per_lap": self.lap_cone_hits,
            "cones_hit_total": len(self.hit_cones),
            "off_track_time_s": round(self.off_track_time, 2),
            "sim_time_s": round(self.t, 2),
        }
