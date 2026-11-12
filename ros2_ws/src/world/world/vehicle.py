"""
vehicle.py - how the car moves.

We use the Bicycle Model: 2 wheels instead of 4-wheel car  
(one front wheel that steers, one rear wheel). Given speed v and steering
angle delta, the car drives along a circle of radius R = wheelbase / tan(delta).

State (what we track every time step):
    x, y   - position of the rear axle [m]
    yaw    - which way the car points [rad]
    v      - speed [m/s]
Inputs (what the autonomy sends us):
    steer  - requested steering angle [rad]
    accel  - requested acceleration [m/s^2] (negative = braking)

Design choices on top of the basic model:
    * the steering wheel can't jump instantly (first-order lag + limits)
    * engine/brake limits on acceleration, plus air drag
    * tire grip limit: a car can only corner at lateral acceleration
      v^2 / R <= mu * g. If the autonomy asks for a tighter turn than the
      tires allow, the car can't follow it and runs wide (understeer).
"""
import math
from dataclasses import dataclass


@dataclass
class VehicleParams:
    wheelbase: float = 1.55          # [m] typical formula style car size
    max_steer: float = 0.45          # [rad] ~26 degrees
    steer_time_constant: float = 0.08  # [s] how quickly steering responds
    max_accel: float = 7.0           # [m/s^2] engine/traction limit
    max_brake: float = 12.0          # [m/s^2]
    max_lateral_accel: float = 14.0  # [m/s^2] tire grip limit (~1.4 g with aero)
    drag_coeff: float = 0.0015       # drag decel = coeff * v^2
    max_speed: float = 45.0          # [m/s] ~160 km/h (Brands Hatch is a full-size
                                     # circuit, so we give the car more top speed
                                     # than a real car's ~120 km/h)


class Vehicle:
    def __init__(self, x, y, yaw, params=None):
        self.p = params or VehicleParams()
        self.x, self.y, self.yaw = x, y, yaw
        self.v = 0.0
        self.steer = 0.0           # actual steering angle (after lag)
        self.lateral_accel = 0.0   # for logging
        self.sliding = False       # True when the tires are at their limit

    def step(self, steer_cmd, accel_cmd, dt):
        p = self.p
        # --- steering actuator: limited and slightly delayed ---
        steer_cmd = max(-p.max_steer, min(p.max_steer, steer_cmd))
        self.steer += (steer_cmd - self.steer) * min(1.0, dt / p.steer_time_constant)

        # --- longitudinal: engine/brake limits and drag ---
        accel = max(-p.max_brake, min(p.max_accel, accel_cmd))
        accel -= p.drag_coeff * self.v ** 2
        self.v = max(0.0, min(p.max_speed, self.v + accel * dt))

        # --- lateral: curvature asked for vs. curvature the tires allow ---
        curvature = math.tan(self.steer) / p.wheelbase      # 1 / R
        if self.v > 1.0:
            grip_curvature = p.max_lateral_accel / self.v ** 2
            self.sliding = abs(curvature) > grip_curvature
            if self.sliding:  # understeer: can't turn as tightly as asked
                curvature = math.copysign(grip_curvature, curvature)
        else:
            self.sliding = False
        self.lateral_accel = self.v ** 2 * curvature

        # --- move the car (bicycle model, rear-axle reference) ---
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self.yaw += self.v * curvature * dt
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))  # keep in [-pi, pi]
