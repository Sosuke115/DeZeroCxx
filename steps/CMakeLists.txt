cmake_minimum_required(VERSION 3.17)

project(step)

set(CMAKE_CXX_STANDARD 17)

include_directories(/usr/local/include)

# local用
include_directories(/opt/homebrew/include)

add_custom_target(valgrind
    COMMAND valgrind --tool=memcheck --leak-check=yes ./step09
    COMMENT "Running valgrind"
)

# add_executable(test test.cpp)
# add_executable(step01 step01.cpp)
# add_executable(step02 step02.cpp)
# add_executable(step03 step03.cpp)
# add_executable(step04 step04.cpp)
# add_executable(step06 step06.cpp)
# add_executable(step07 step07.cpp)
# add_executable(step08 step08.cpp)
add_executable(step09 step09.cpp)
