# CONTAINER 2 - "the brain" (autonomy)
# Official ROS2 Jazzy image. It is built on Ubuntu 24.04 (noble) and ships Python 3.12.
FROM ros:jazzy-ros-base

# numpy/scipy for the math, matplotlib for charts + video, ffmpeg to write the MP4
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-numpy python3-scipy python3-matplotlib python3-pil ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ros2_ws
COPY ros2_ws/src/msgs     src/msgs
COPY ros2_ws/src/autonomy src/autonomy
RUN . /opt/ros/jazzy/setup.sh && colcon build --event-handlers console_direct-

COPY docker/fastdds.xml /fastdds.xml
COPY docker/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "run", "autonomy", "autonomy_node"]
