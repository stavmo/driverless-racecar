cmake_minimum_required(VERSION 3.8)
project(msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# Turn the .msg text files into real Python (and C++) classes
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Cone.msg"
  "msg/ConeArray.msg"
  "msg/ControlCommand.msg"
  DEPENDENCIES std_msgs
)

ament_package()
