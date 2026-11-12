# A list of cones, everything the sensor sees in one scan.
# header.stamp    = simulation time of the measurement
# header.frame_id = "base_link" (car frame: x forward, y left) or "map" (world frame)
std_msgs/Header header
Cone[] cones
