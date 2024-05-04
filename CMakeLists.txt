cmake_minimum_required(VERSION 3.5)
project(my-first-detector)
find_package( OpenCV REQUIRED )
find_package(realsense2 REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${TensorFlow_INCLUDE_DIRS})
add_executable(my-first-detector my-first-detector.cpp)
target_link_libraries(my-first-detector ${OpenCV_LIBS} ${realsense2_LIBRARY})

