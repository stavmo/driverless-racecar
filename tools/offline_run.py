"""
offline_run.py - run the whole pipeline WITHOUT ROS or Docker.

Same simulator + same brain code as the ROS nodes, just wired together
directly in one Python loop. it's useful for fast development and testing.
    python3 tools/offline_run.py

Optional environment variables for experiments:
    NOISE=2.0    multiply the sensor noise
    DETECT=0.8   probability that the sensor detects a visible cone
    SEED=1       random seed
"""
import os
import pickle
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "world"))
sys.path.insert(0, os.path.join(ROOT, "ros2_ws", "src", "autonomy"))

from world.world import World          # noqa: E402
from autonomy.brain import Brain     # noqa: E402

PHYSICS_DT = 0.01     # 100 Hz physics
CONTROL_EVERY = 5     # 20 Hz odometry + control
SENSOR_EVERY = 10     # 10 Hz cone detections
MAX_SIM_TIME = float(os.environ.get("MAX_T", 1200))


def main():
    noise = float(os.environ.get("NOISE", 1.0))
    world = World(os.path.join(ROOT, "data", "BrandsHatchLayout.csv"), seed=int(os.environ.get("SEED", 0)),
                  sensor_kwargs=dict(noise_base=0.02 * noise, noise_per_m=0.005 * noise,
                                     detect_prob=float(os.environ.get("DETECT", 0.95))))
    brain = Brain()
    steer, accel = 0.0, 0.0
    wall = time.time()
    step = 0
    while not brain.finished and world.t < MAX_SIM_TIME:
        if step % SENSOR_EVERY == 0:
            brain.on_cones(world.detect_cones(), world.t)
        if step % CONTROL_EVERY == 0:
            steer, accel = brain.on_odom(*world.odometry(), world.t)
        world.step(steer, accel, PHYSICS_DT)
        step += 1
    print("wall time %.1fs" % (time.time() - wall))
    print("brain mode:", brain.mode, "laps:", brain.lap, "map cones:", len(brain.map))
    print(world.summary())
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    out = os.environ.get("OUT", os.path.join(ROOT, "results", "offline_run.pkl"))
    with open(out, "wb") as f:
        pickle.dump(dict(log=brain.log, snapshots=brain.snapshots, plans=brain.plans,
                         map_xy=brain.map.xy, map_color=brain.map.color,
                         map_first_seen=brain.first_seen_t,
                         gt_cones=world.track.cones, summary=world.summary()), f)


if __name__ == "__main__":
    main()
