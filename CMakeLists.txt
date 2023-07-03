cmake_minimum_required(VERSION 3.22)
project(SimdOperators)

set(CMAKE_CXX_STANDARD 20)

# /// create the TSL library
include(${PROJECT_SOURCE_DIR}/tools/tslgen/tsl.cmake)
create_tsl(
    TSLGENERATOR_DIRECTORY ${PROJECT_SOURCE_DIR}/tools/tslgen
    DESTINATION ${PROJECT_SOURCE_DIR}/libs/tsl
)

include_directories(${TSL_INCLUDE_DIRECTORY})

# /// create the SIMDOperators library
add_subdirectory(src)

# /// create the tests
add_subdirectory(test)


