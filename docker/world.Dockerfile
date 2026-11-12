# CONTAINER 1 - "the world" (simulator)
# Built on Ubuntu 24.04 (noble) and ships Python 3.12.
FROM ros:jazzy-ros-base

# Python libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-numpy python3-scipy \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws
COPY ros2_ws/src/msgs src/msgs
COPY ros2_ws/src/world  src/world
RUN . /opt/ros/jazzy/setup.sh && colcon build --event-handlers console_direct-

# Network settings so the two containers find each other
COPY docker/fastdds.xml /fastdds.xml
COPY docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "run", "world", "world_node"]
