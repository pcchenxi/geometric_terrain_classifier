
cmake_minimum_required(VERSION 2.8)
project(centauro_costmap)

find_package(catkin REQUIRED COMPONENTS
	nodelet
	message_generation
)

add_message_files(FILES
	CostMap.msg
)

generate_messages(DEPENDENCIES
	std_msgs
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")

