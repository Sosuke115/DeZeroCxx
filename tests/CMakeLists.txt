cmake_minimum_required(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)

find_package(GTest REQUIRED)
include(GoogleTest)

set(pwd ${CMAKE_CURRENT_SOURCE_DIR})
set(root_dir ${pwd}/..)

set(target "unittest")

set(test_sources 
    ${root_dir}/steps/step09.cpp
    ${pwd}/test_square.cpp
)


add_executable(${target}
    ${test_sources}
)

target_link_libraries(${target} GTest::GTest GTest::Main
)

target_include_directories(${target} PRIVATE 
    ${GTEST_INCLUDE_DIRS}
    /usr/local/include
    /opt/homebrew/include
    ${root_dir}/steps
)

# Google Testの各テストケースごとにCTestのテストを作成する
gtest_add_tests(TARGET ${target})

# ctest用(ctest -T memcheck でメモリチェック)
# https://stackoverflow.com/questions/40325957/how-do-i-add-valgrind-tests-to-my-cmake-test-target
find_program(MEMORYCHECK_COMMAND valgrind)
set(MEMORYCHECK_COMMAND_OPTIONS "--leak-check=yes --error-exitcode=1")
include (CTest)
