"""
brain.py - the autonomy "brain", with no ROS code in it (can be tested alone).

   EXPLORE  (lap 1)  drive carefully using only cones seen so far, build the map
      |              -> at the start line: build the full centerline of the map
      v                 and plan a speed for every point of it
   FAST     (lap 2)  drive fast along the centerline, following the speed plan
      |
      v
   FINISH            brake to a stop, mission complete

Inputs:  on_cones(detections)      - from the perception sensor
         on_odom(x, y, yaw, v, t)   - the car's position and speed
Output:  on_odom returns (steering, acceleration) for the car.
"""
import math

import numpy as np

from .cone_map import ConeMap
from .path_planning import curvature_closed, curvature_open, global_centerline, local_centerline
from .pure_pursuit import PurePursuit, TrackFollower
from .speed_planning import safe_speed_ahead, speed_profile_closed

EXPLORE, FAST, FINISH, DONE = "EXPLORE", "FAST", "FINISH", "DONE"


class Brain:
    def __init__(self, explore_speed=14.0, explore_a_lat=8.0,
                 fast_a_lat=12.5, fast_a_accel=6.0, fast_a_brake=10.0, fast_v_max=44.0):
        self.map = ConeMap()
        self.ctrl = PurePursuit()
        self.mode = EXPLORE
        self.explore_speed = explore_speed
        self.explore_a_lat = explore_a_lat
        self.fast_params = dict(a_lat_max=fast_a_lat, a_accel=fast_a_accel,
                                a_brake=fast_a_brake, v_max=fast_v_max)
        self.pose = None
        self.start = None
        self.lap = 0
        self.dist_since_line = 0.0
        self._prev_pos = None
        self._prev_ahead = None
        self.lap1_trajectory = []
        self.follower = None
        self.local_path = np.zeros((0, 2))
        self.plans = {}       # after lap 1 (centerline + speed plan)
        self.log = []         # one row per control step (for plots)
        self.snapshots = []   # occasional bigger records (for the video)
        self._last_snapshot_t = -1e9
        self._last_detections = np.zeros((0, 3))
        self.first_seen_t = []  # time each map cone was first seen
        self._pose_history = []  # recent (t, x, y, yaw, v) for time synchronization

    # ------------------------------------------------------------------ inputs
    def on_cones(self, detections, t):
        pose = self._pose_at(t)
        if pose is None:
            return
        x, y, yaw = pose
        n_before = len(self.map)
        self.map.update(detections, x, y, yaw)
        self.first_seen_t += [t] * (len(self.map) - n_before)
        world = ConeMap.to_world(detections, x, y, yaw) if len(detections) else np.zeros((0, 2))
        self._last_detections = np.column_stack([world, detections[:, 2]]) if len(world) else np.zeros((0, 3))

    def _pose_at(self, t):
        """TIME SYNCHRONIZATION (I explained it also in the README): where was the car at the moment the cones were seen?
        Sensor and odometry messages arrive at different times. At 40 m/s, using
        a pose that is just 0.05 s old puts every cone 2 m in the wrong place.
        So we keep a short history of poses and interpolate to the sensor's timestamp."""
        h = self._pose_history
        if not h:
            return None
        if t >= h[-1][0]:     # newer than our last pose: extrapolate with speed
            t0, x, y, yaw, v = h[-1]
            dt = t - t0
            return x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt, yaw
        for (ta, xa, ya, yawa, _), (tb, xb, yb, yawb, _) in zip(reversed(h[:-1]), reversed(h[1:])):
            if ta <= t <= tb:
                w = (t - ta) / max(tb - ta, 1e-9)
                dyaw = math.atan2(math.sin(yawb - yawa), math.cos(yawb - yawa))
                return xa + w * (xb - xa), ya + w * (yb - ya), yawa + w * dyaw
        return h[0][1:4]

    def on_odom(self, x, y, yaw, v, t):
        self.pose = (x, y, yaw, v)
        self._pose_history.append((t, x, y, yaw, v))
        if len(self._pose_history) > 40:
            self._pose_history.pop(0)
        if self.start is None:
            self.start = (x, y, yaw)
        self._track_progress(x, y)

        if self.mode == EXPLORE:
            steer, accel, xte = self._explore(x, y, yaw, v)
        elif self.mode == FAST:
            steer, accel, xte = self.follower.command(x, y, yaw, v)
        else:  # FINISH / DONE: keep steering on the line, brake
            steer, _, xte = self.follower.command(x, y, yaw, v)
            accel = -8.0
            if v < 0.2:
                self.mode = DONE

        self.log.append((t, x, y, yaw, v, steer, accel, xte, self.lap, self.mode))
        if t - self._last_snapshot_t >= 0.25:
            self._snapshot(t)
        return steer, accel

    # ------------------------------------------------------------ lap logic
    def _track_progress(self, x, y):
        p = np.array([x, y])
        if self._prev_pos is not None:
            self.dist_since_line += float(np.linalg.norm(p - self._prev_pos))
        self._prev_pos = p
        if self.mode == EXPLORE and (not self.lap1_trajectory or
                                     np.linalg.norm(p - self.lap1_trajectory[-1]) > 1.0):
            self.lap1_trajectory.append(p)

        # crossed our own starting line (moving forward, after a real lap)?
        sx, sy, syaw = self.start
        ahead = (x - sx) * math.cos(syaw) + (y - sy) * math.sin(syaw)
        lateral = abs(-(x - sx) * math.sin(syaw) + (y - sy) * math.cos(syaw))
        crossed = self._prev_ahead is not None and self._prev_ahead < 0 <= ahead
        self._prev_ahead = ahead
        if crossed and lateral < 25 and self.dist_since_line > 500:
            self.dist_since_line = 0.0
            self.lap += 1
            self._on_lap_complete()

    def _on_lap_complete(self):
        if self.mode == EXPLORE:
            self._plan_full_track()
            self.mode = FAST
            self.follower = TrackFollower(self.plans["center"], self.plans["center_speed"], self.ctrl)
        elif self.mode == FAST:
            self.mode = FINISH   # keep following the centerline while braking

    # ------------------------------------------------------------ planning
    def _explore(self, x, y, yaw, v):
        path = local_centerline(self.map, x, y, yaw)
        self.local_path = path
        if len(path) < 2:
            return 0.0, self.ctrl.speed_cmd(v, 2.0), 0.0
        # resample to even 1.5 m spacing so curvature estimates are meaningful
        seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
        s = np.concatenate([[0], np.cumsum(seg)])
        si = np.arange(0, s[-1], 1.5)
        even = np.column_stack([np.interp(si, s, path[:, 0]), np.interp(si, s, path[:, 1])])
        L = self.ctrl.lookahead(v)
        target = even[min(len(even) - 1, int(L / 1.5))]
        steer = self.ctrl.steer_to(x, y, yaw, target)
        v_target = safe_speed_ahead(even, curvature_open(even), self.explore_a_lat,
                                    a_brake=6.0, v_max=self.explore_speed)
        return steer, self.ctrl.speed_cmd(v, v_target), 0.0

    def _plan_full_track(self):
        center, wl, wr = global_centerline(self.map, self.lap1_trajectory)
        k_c = curvature_closed(center)
        v_c = speed_profile_closed(center, k_c, **self.fast_params)
        self.plans = dict(center=center, center_speed=v_c, center_curv=k_c,
                          width_left=wl, width_right=wr)

    # ------------------------------------------------------------ recording
    def _snapshot(self, t):
        self._last_snapshot_t = t
        x, y, yaw, v = self.pose
        if self.mode == EXPLORE:
            path = self.local_path
        else:
            i = self.follower.idx or 0
            path = self.follower.path[(i + np.arange(0, 40)) % self.follower.n]
        det = self._last_detections
        self.snapshots.append(dict(t=t, x=x, y=y, yaw=yaw, v=v, mode=self.mode, lap=self.lap,
                                   map_size=len(self.map), path=np.array(path), detections=det))

    @property
    def finished(self):
        return self.mode == DONE
