cmake_minimum_required(VERSION 3.17)

# Create project
set(ProjectName "DeZeroCxx")
project(${ProjectName})

set(CMAKE_CXX_STANDARD 17)

enable_testing()
add_subdirectory(tests)
