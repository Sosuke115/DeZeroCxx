cmake_minimum_required(VERSION 3.17)

project(step)

set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/local/include)
include_directories(/opt/homebrew/Cellar/boost/1.81.0_1/include)

add_executable(step01 step01.cpp)
add_executable(step02 step02.cpp)
add_executable(step03 step03.cpp)
add_executable(step04 step04.cpp)