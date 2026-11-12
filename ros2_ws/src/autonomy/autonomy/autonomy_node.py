"""
autonomy_node.py - ROS2 node for CONTAINER 2.

It wraps brain.py (which is pure Python) with ROS2 communication:

  subscribes /perception/cones       msgs/ConeArray       what the sensor sees (car frame)
  subscribes /vehicle/odometry       nav_msgs/Odometry        where the car is, how fast it goes
  publishes  /control/command        msgs/ControlCommand  steering + acceleration
  publishes  /mission/finished       std_msgs/Bool            "I'm done" after the last lap
  publishes  /autonomy/map           msgs/ConeArray       the cone map built so far (1 Hz, for debugging)
  publishes  /autonomy/mode           std_msgs/String       the current mode (EXPLORE/FAST/FINISH/DONE), sent when it changes
  subscribes /world/summary            std_msgs/String          official results from the simulator
  subscribes /world/ground_truth_cones msgs/ConeArray       true cones- used only in the end to score the map
                                                              (in the final report, not for driving)

When the mission is complete it writes the results (charts, video, summary) into /results
"""
import json
import math
import os
import pickle
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from nav_msgs.msg import Odometry
from std_msgs.msg import Bool, String

from msgs.msg import Cone, ConeArray, ControlCommand

from .brain import Brain

RELIABLE = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE)
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


def stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class AutonomyNode(Node):
    def __init__(self):
        super().__init__("autonomy_node")
        self.declare_parameter("results_dir", "/results")
        self.declare_parameter("make_video", True)
        self.declare_parameter("car_name", "Lightning McQueen")
        self.declare_parameter("author", "Stav Mordekhai")
        self.results_dir = self.get_parameter("results_dir").value
        self.make_video = bool(self.get_parameter("make_video").value)
        self.car_name = self.get_parameter("car_name").value
        self.author = self.get_parameter("author").value

        self.brain = Brain()
        self.cmd_pub = self.create_publisher(ControlCommand, "/control/command", RELIABLE)
        self.done_pub = self.create_publisher(Bool, "/mission/finished", LATCHED)
        self.map_pub = self.create_publisher(ConeArray, "/autonomy/map", RELIABLE)
        self.mode_pub = self.create_publisher(String, "/autonomy/mode", RELIABLE)
        self.create_subscription(ConeArray, "/perception/cones", self.on_cones, RELIABLE)
        self.create_subscription(Odometry, "/vehicle/odometry", self.on_odom, RELIABLE)
        self.create_subscription(String, "/world/summary", self.on_summary, LATCHED)
        self.create_subscription(ConeArray, "/world/ground_truth_cones", self.on_truth, LATCHED)
        self.create_timer(1.0, self.publish_map)

        self.sim_summary = None
        self.truth = None
        self.finished_at = None
        self.complete = False
        self._last_mode = None
        self.get_logger().info("Autonomy brain ready, waiting for sensor data...")

    # ---------------- callbacks ----------------
    def on_cones(self, msg):
        det = np.array([[c.x, c.y, c.color] for c in msg.cones]) if msg.cones else np.zeros((0, 3))
        self.brain.on_cones(det, stamp_to_sec(msg.header.stamp))

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        yaw = 2 * math.atan2(q.z, q.w)          #get which way the car is facing (in radians) from the message
        t = stamp_to_sec(msg.header.stamp)
        steer, accel = self.brain.on_odom(p.x, p.y, yaw, msg.twist.twist.linear.x, t)

        cmd = ControlCommand()
        cmd.header.stamp = msg.header.stamp      # answer this odometry message (lockstep)
        cmd.steering, cmd.acceleration = float(steer), float(accel)
        self.cmd_pub.publish(cmd)

        if self.brain.mode != self._last_mode:
            self._last_mode = self.brain.mode
            self.get_logger().info(f"[t={t:6.1f}s] {self.car_name}: mode -> {self.brain.mode}")
            self.mode_pub.publish(String(data=self.brain.mode))
            if self.brain.plans and self.brain.mode == "FAST":
                pl = self.brain.plans
                self.get_logger().info(
                    f"Full track mapped: {len(self.brain.map)} cones. Centerline {len(pl['center'])} pts, "
                    f"planned lap time {self._planned(pl['center'], pl['center_speed']):.1f}s")
        if self.brain.finished and self.finished_at is None:
            self.finished_at = time.time()
            self.done_pub.publish(Bool(data=True))
            self.get_logger().info("Mission complete - car stopped.")

    @staticmethod
    def _planned(path, v):
        ds = np.linalg.norm(np.roll(path, -1, axis=0) - path, axis=1)
        return float(np.sum(ds / np.maximum(v, 0.1)))

    def on_summary(self, msg):
        self.sim_summary = json.loads(msg.data)

    def on_truth(self, msg):
        self.truth = np.array([[c.x, c.y, c.color] for c in msg.cones])

    def publish_map(self):
        m = ConeArray()
        m.header.frame_id = "map"
        m.header.stamp = self.get_clock().now().to_msg()
        for (x, y), c in zip(self.brain.map.xy, self.brain.map.color):
            m.cones.append(Cone(x=float(x), y=float(y), color=int(c)))
        self.map_pub.publish(m)

    # ---------------- results ----------------
    def ready_for_report(self):
        if self.finished_at is None:
            return False
        return self.sim_summary is not None or time.time() - self.finished_at > 15.0

    def write_results(self):
        from .report import generate_report
        os.makedirs(self.results_dir, exist_ok=True)
        b = self.brain
        data = dict(log=b.log, snapshots=b.snapshots, plans=b.plans, map_xy=b.map.xy,
                    map_color=b.map.color, map_first_seen=b.first_seen_t)
        with open(os.path.join(self.results_dir, "run_data.pkl"), "wb") as f:
            pickle.dump(data, f)
        self.get_logger().info("Writing charts and summary...")
        generate_report(data, self.results_dir, self.sim_summary, self.truth,
                        log=self.get_logger().info)
        if self.make_video:
            from .video import render_video
            self.get_logger().info("Rendering the video (takes 1-3 minutes)...")
            render_video(data, os.path.join(self.results_dir, "drive.gif"),
                         explore_dt=3.0, fast_dt=1.0, fps=12, dpi=65, car_name=self.car_name)
            try:
                render_video(data, os.path.join(self.results_dir, "drive.mp4"),
                             explore_dt=1.0, fast_dt=0.5, fps=20, dpi=100, car_name=self.car_name)
            except Exception as e:  # ffmpeg missing etc- then the GIF is enough
                self.get_logger().warn(f"MP4 skipped: {e}")
        self.get_logger().info(f"All results saved to {self.results_dir}")
        from .finish_banner import finish_banner
        print("\n" + finish_banner(self.car_name, self.author, self.sim_summary) + "\n", flush=True)


def main():
    rclpy.init()
    node = AutonomyNode()
    try:
        while rclpy.ok() and not node.ready_for_report():
            rclpy.spin_once(node, timeout_sec=0.1)
        if rclpy.ok():
            node.write_results()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
