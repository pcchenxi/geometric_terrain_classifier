cmake_minimum_required(VERSION 2.8.3)
project(terrain_classifier)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(PCL 1.7 REQUIRED COMPONENTS)
find_package(OpenCV 2.4 REQUIRED)
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  sensor_msgs
  std_msgs
  tf
  laser_geometry

  pcl_ros
  cv_bridge
  image_transport
  image_geometry
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES pluto_costmap_generator
#  CATKIN_DEPENDS pcl_ros roscpp rospy sensor_msgs std_msgs tf
#  DEPENDS system_lib
)

include_directories(
  /home/xi/catkin_rcv/src/terrain_feature/terrain/src
  /home/xi/catkin_rcv/src/cloud_img_mapper/src

  ${OpenCV_INCLUDE_DIRS}
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(
  ${OpenCV_LIBRARY_DIRS}
  ${PCL_LIBRARY_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

add_executable(pluto_terrain_classification src/pluto_terrain_classification.cpp)

target_link_libraries(pluto_terrain_classification
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
