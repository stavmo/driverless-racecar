"""
world_node.py - ROS2 node for CONTAINER 1 ("the world").

It wraps world.py (Python) with ROS2 communication:

  publishes  /perception/cones       msgs/ConeArray   cones the sensor sees (car frame), 10 Hz
  publishes  /vehicle/odometry       nav_msgs/Odometry    car position, heading and speed, 20 Hz
  subscribes /control/command        msgs/ControlCommand   steering + acceleration from the brain
  subscribes /mission/finished       std_msgs/Bool        the brain says it is done
  publishes  /world/summary            std_msgs/String      official results (lap times, cones hit) as JSON
  publishes  /world/ground_truth_cones msgs/ConeArray   true cone positions - for scoring the map only

Lockstep simulation: every 0.05 s of simulated time the world publishes the
odometry, then waits for the brain's command for that exact timestamp before
moving on, taking turns. This makes the result identical on every
computer, no matter how fast or slow it is which gives reproducible results.
Set the parameter real_time_factor (like 1.0) to also slow it down to real time.
"""
import json
import math
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String

from msgs.msg import Cone, ConeArray, ControlCommand

from .world import World

PHYSICS_DT = 0.01      # [s] physics step (100 Hz)
CONTROL_EVERY = 5      # odometry + control at 20 Hz
SENSOR_EVERY = 10      # cone detections at 10 Hz

RELIABLE = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)  # late joiners still get it


def to_stamp(t):
    # Convert to whole nanoseconds first, then split. (Splitting first breaks for
    # values like t = 4.9999999999 -> sec=4, nanosec rounds to 1e9 -> time jumps back.)
    sec, nanosec = divmod(int(round(t * 1e9)), 1_000_000_000)
    return Time(sec=sec, nanosec=nanosec)


class SimNode(Node):
    def __init__(self):
        super().__init__("world_node")
        self.declare_parameter("csv_path", "/data/BrandsHatchLayout.csv")
        self.declare_parameter("results_dir", "/results")
        self.declare_parameter("seed", 0)
        self.declare_parameter("max_sim_time", 900.0)
        self.declare_parameter("real_time_factor", 0.0)  # 0 = as fast as possible
        self.declare_parameter("sensor_noise_scale", 1.0)
        self.declare_parameter("detection_probability", 0.95)
        self.declare_parameter("car_name", "Lightning McQueen")

        p = lambda name: self.get_parameter(name).value  # noqa: E731
        noise = float(p("sensor_noise_scale"))
        self.world = World(p("csv_path"), seed=int(p("seed")), sensor_kwargs=dict(
            noise_base=0.02 * noise, noise_per_m=0.005 * noise,
            detect_prob=float(p("detection_probability"))))
        self.results_dir = p("results_dir")
        self.max_sim_time = float(p("max_sim_time"))
        self.rtf = float(p("real_time_factor"))
        self.car_name = p("car_name")

        self.cone_pub = self.create_publisher(ConeArray, "/perception/cones", RELIABLE)
        self.odom_pub = self.create_publisher(Odometry, "/vehicle/odometry", RELIABLE)
        self.summary_pub = self.create_publisher(String, "/world/summary", LATCHED)
        self.truth_pub = self.create_publisher(ConeArray, "/world/ground_truth_cones", LATCHED)
        self.create_subscription(ControlCommand, "/control/command", self.on_command, RELIABLE)
        self.create_subscription(Bool, "/mission/finished", self.on_finished, LATCHED)

        self.cmd = (0.0, 0.0)
        self.cmd_stamp = None
        self.finished = False
        stats = self.world.track.stats()
        self.get_logger().info(f"Track loaded: {json.dumps(stats)}")

    # ---------------- callbacks ----------------
    def on_command(self, msg):
        self.cmd = (msg.steering, msg.acceleration)
        self.cmd_stamp = (msg.header.stamp.sec, msg.header.stamp.nanosec)

    def on_finished(self, msg):
        if msg.data:
            self.finished = True

    # ---------------- publishing helpers ----------------
    def publish_cones(self, stamp):
        msg = ConeArray()
        msg.header.stamp, msg.header.frame_id = stamp, "base_link"
        for x, y, c in self.world.detect_cones():
            msg.cones.append(Cone(x=float(x), y=float(y), color=int(c)))
        self.cone_pub.publish(msg)

    def publish_odom(self, stamp):
        x, y, yaw, v = self.world.odometry()
        msg = Odometry()
        msg.header.stamp, msg.header.frame_id = stamp, "map"
        msg.child_frame_id = "base_link"
        msg.pose.pose.position.x, msg.pose.pose.position.y = float(x), float(y)
        # ROS stores rotations as 4 numbers (a "quaternion"). For a car on flat ground we only
        # turn around the vertical axis, so only z and w are needed: z = sin(yaw/2), w = cos(yaw/2).
        msg.pose.pose.orientation.z = math.sin(yaw / 2)
        msg.pose.pose.orientation.w = math.cos(yaw / 2)
        msg.twist.twist.linear.x = float(v)
        self.odom_pub.publish(msg)

    def publish_ground_truth(self):
        msg = ConeArray()
        msg.header.frame_id = "map"
        for x, y, c in self.world.track.cones:
            msg.cones.append(Cone(x=float(x), y=float(y), color=int(c)))
        self.truth_pub.publish(msg)

    # ---------------- main loop ----------------
    def wait_for_autonomy(self):
        self.get_logger().info("Waiting for the autonomy node to connect...")
        while rclpy.ok() and (self.cone_pub.get_subscription_count() == 0 or
                              self.count_publishers("/control/command") == 0):
            rclpy.spin_once(self, timeout_sec=0.2)
        time.sleep(1.0)  # give discovery a moment to finish on both sides
        self.get_logger().info(f"Autonomy connected - {self.car_name} is starting the race!")

    def wait_for_command(self, stamp, timeout=60.0):
        wanted = (stamp.sec, stamp.nanosec)
        start = time.time()
        while rclpy.ok() and self.cmd_stamp != wanted and not self.finished:
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() - start > timeout:
                self.get_logger().warn("No command from autonomy - using the last one")
                return

    def run(self):
        self.publish_ground_truth()
        step, wall_start, last_log = 0, time.time(), -10.0
        laps_seen = 0
        while rclpy.ok() and not self.finished and self.world.t < self.max_sim_time:
            if step % SENSOR_EVERY == 0:
                self.publish_cones(to_stamp(self.world.t))
            if step % CONTROL_EVERY == 0:
                stamp = to_stamp(self.world.t)
                self.publish_odom(stamp)
                self.wait_for_command(stamp)
            self.world.step(*self.cmd, PHYSICS_DT)
            step += 1

            if len(self.world.lap_times) > laps_seen:
                laps_seen = len(self.world.lap_times)
                self.get_logger().info(f"{self.car_name} finished LAP {laps_seen}: {self.world.lap_times[-1]:.2f} s, "
                                       f"cones hit this lap: {self.world.lap_cone_hits[-1]}")
            if self.world.t - last_log >= 20.0:
                last_log = self.world.t
                self.get_logger().info(f"sim t={self.world.t:6.1f}s  speed={self.world.car.v * 3.6:5.1f} km/h")
            if self.rtf > 0:  # this is optional: slow down to (a multiple of) real time
                ahead = self.world.t / self.rtf - (time.time() - wall_start)
                if ahead > 0:
                    time.sleep(ahead)

        summary = self.world.summary()
        summary["track"] = self.world.track.stats()
        self.get_logger().info(f"Mission over. Official results: {json.dumps(summary)}")
        self.summary_pub.publish(String(data=json.dumps(summary)))
        os.makedirs(self.results_dir, exist_ok=True)
        with open(os.path.join(self.results_dir, "sim_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        end = time.time() + 5.0  # keep answering for a moment so the summary gets delivered
        while rclpy.ok() and time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)


def main():
    rclpy.init()
    node = SimNode()
    try:
        node.wait_for_autonomy()
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
