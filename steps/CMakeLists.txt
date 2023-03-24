cmake_minimum_required(VERSION 3.17)

project(step)

set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/local/include)
include_directories(/opt/homebrew/Cellar/boost/1.81.0_1/include)

# step01 executable
add_executable(step01 step01.cpp)
add_executable(step02 step02.cpp)