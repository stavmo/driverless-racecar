# what the autonomy wants the car to do
# header.stamp echoes the odometry message this command answers (lockstep simulation).
std_msgs/Header header
float64 steering      # [rad] front wheel angle, + = turn left
float64 acceleration  # [m/s^2] + = throttle, - = brake
