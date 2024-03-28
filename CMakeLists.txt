cmake_minimum_required(VERSION 3.5)
project(my-first-tensorflow)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${TensorFlow_INCLUDE_DIRS})
add_executable(my-first-tensorflow my-first-tensorflow.cpp)
target_link_libraries(my-first-tensorflow libtensorflow.so libtensorflow.so.2 ${OpenCV_LIBS})

