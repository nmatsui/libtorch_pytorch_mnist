cmake_minimum_required(VERSION 3.1)

project (libtorch_mnist CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_subdirectory("pimage")
add_subdirectory("pcamera")
