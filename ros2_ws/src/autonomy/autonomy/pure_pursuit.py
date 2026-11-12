"""
pure_pursuit.py - steering + throttle control.

  1. find the point on the path that is `L` meters ahead of the car
     (L = ahead distance; we make it longer at high speed for smoothness)
  2. alpha = angle between where the car points and that target point
  3. the circle that passes through the car and the target point needs
         steering angle  delta = atan( 2 * wheelbase * sin(alpha) / L )
     (geometry: a circle through two points, tangent to the car's heading)

Proportional speed control:
         accel = Kp * (target_speed - current_speed)
  too slow -> accelerate, too fast -> brake, harder the bigger the error.
"""
import math

import numpy as np


class PurePursuit:
    def __init__(self, wheelbase=1.55, k_lookahead=0.30, min_lookahead=4.0,
                 max_lookahead=20.0, kp_speed=2.5, max_accel=7.0, max_brake=12.0):
        self.wheelbase = wheelbase
        self.k = k_lookahead
        self.min_l, self.max_l = min_lookahead, max_lookahead
        self.kp = kp_speed
        self.max_accel, self.max_brake = max_accel, max_brake

    def lookahead(self, v):
        return min(self.max_l, max(self.min_l, self.k * v + 2.0))

    def steer_to(self, x, y, yaw, target):
        dx, dy = target[0] - x, target[1] - y
        alpha = math.atan2(dy, dx) - yaw
        L = max(math.hypot(dx, dy), 0.1)
        return math.atan2(2 * self.wheelbase * math.sin(alpha), L)

    def speed_cmd(self, v, v_target):
        a = self.kp * (v_target - v)
        return max(-self.max_brake, min(self.max_accel, a))


class TrackFollower:
    """Pure pursuit on a full closed path, remembering our progress along it."""

    def __init__(self, path, speeds, controller):
        self.path = path
        self.speeds = speeds
        self.ctrl = controller
        self.n = len(path)
        self.ds = float(np.mean(np.linalg.norm(np.diff(path, axis=0), axis=1)))
        self.idx = None

    def _nearest(self, x, y):
        p = np.array([x, y])
        if self.idx is None:        # first call: search the whole path
            self.idx = int(np.argmin(np.linalg.norm(self.path - p, axis=1)))
        else:                       # afterwards: search just ahead of last position
            window = (self.idx + np.arange(-5, 60)) % self.n
            self.idx = int(window[np.argmin(np.linalg.norm(self.path[window] - p, axis=1))])
        return self.idx

    def command(self, x, y, yaw, v):
        i = self._nearest(x, y)
        L = self.ctrl.lookahead(v)
        target = self.path[(i + int(round(L / self.ds))) % self.n]
        steer = self.ctrl.steer_to(x, y, yaw, target)
        # look ~0.4 s ahead on the speed profile so braking starts on time
        j = (i + int(round(max(v, 1.0) * 0.4 / self.ds))) % self.n
        accel = self.ctrl.speed_cmd(v, self.speeds[j])
        cross_track = float(np.linalg.norm(self.path[i] - np.array([x, y])))
        return steer, accel, cross_track
