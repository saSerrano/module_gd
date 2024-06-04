cmake_minimum_required(VERSION 2.8.12.2)
project(module_gd)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  std_msgs
  tf
  tf2
  message_generation
)

## Generate messages in the 'msg' folder
add_message_files(
   FILES
   Vector3Array.msg
   Vector3Mat.msg
   CheckedVector3Array.msg
   CheckedVector3Mat.msg
   PointedLocation.msg
 )

## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   geometry_msgs
   std_msgs
 )

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES module_gd
  CATKIN_DEPENDS geometry_msgs roscpp rospy std_msgs tf tf2 message_runtime
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(gesture_detector_node src/gesture_detector.cpp)
target_link_libraries(gesture_detector_node ${catkin_LIBRARIES})
