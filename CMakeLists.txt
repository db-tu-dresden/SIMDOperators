cmake_minimum_required(VERSION 3.22)
project(SimdOperators)

set(CMAKE_CXX_STANDARD 20)

# /// create the TSL library
include(${PROJECT_SOURCE_DIR}/tools/tslgen/tsl.cmake)
create_tsl(
  USE_CONCEPTS
  TSLGENERATOR_DIRECTORY ${PROJECT_SOURCE_DIR}/tools/tslgen
  DESTINATION ${PROJECT_SOURCE_DIR}/libs/tsl
)

include_directories(${TSL_INCLUDE_DIRECTORY})

add_definitions(${CMAKE_DEFINITIONS})

set(SIMDOperators_INCLUDE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src)
# /// create the SIMDOperators library
add_subdirectory(src)

# /// create the examples
add_subdirectory(examples)

# /// create the tests
if (ENABLE_TESTING)
    add_subdirectory(test)
endif()

