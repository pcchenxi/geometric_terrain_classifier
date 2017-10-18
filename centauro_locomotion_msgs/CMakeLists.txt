
cmake_minimum_required(VERSION 2.8)

project(centauro_locomotion_msgs)

find_package(catkin REQUIRED COMPONENTS 
	std_msgs 
	message_generation 
	genmsg
	actionlib
	actionlib_msgs
)
	
add_message_files(
	FILES
	DrivingMovement.msg
	TerrainClassMap.msg
)

add_action_files(
	FILES	
	LegMovement.action
)

generate_messages(
	DEPENDENCIES
	std_msgs
	actionlib_msgs
)

catkin_package(
	CATKIN_DEPENDS message_runtime
)

include_directories(${catkin_INCLUDE_DIRS} include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")