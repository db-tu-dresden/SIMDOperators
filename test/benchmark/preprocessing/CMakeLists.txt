
cmake_minimum_required(VERSION 3.20)
# project(Benchmark)

# if(NOT DEFINED SIMDOperators_INCLUDE_DIRECTORY)
# set(SIMDOPS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
set(CMAKE_CXX_STANDARD 20)

# add_subdirectory("${SIMDOPS_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/simdops")

add_executable(preprocessing preprocessing.cpp)

target_link_libraries(preprocessing PUBLIC SIMDOperators)

target_include_directories(preprocessing PUBLIC "${SIMDOperators_INCLUDE_DIRECTORY}")

if(${CMAKE_BUILD_TYPE} STREQUAL "Release")
    target_compile_options(preprocessing PUBLIC -O3 -g)
else()
    target_compile_options(preprocessing PUBLIC -O0 -g)
    target_compile_definitions(preprocessing PUBLIC -DDEBUG)
endif()

