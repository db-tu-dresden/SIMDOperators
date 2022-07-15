cmake_minimum_required(VERSION 3.22)
project(SimdOperators)

set(CMAKE_CXX_STANDARD 20)

add_subdirectory(${PROJECT_SOURCE_DIR}/libs/tvl)
include_directories(${PROJECT_SOURCE_DIR}/dbops/utils/include)
include_directories(${PROJECT_SOURCE_DIR}/dbops/datastructure/include)
include_directories(${PROJECT_SOURCE_DIR}/dbops/operators/include)


add_subdirectory(dbops/operators/src)
add_subdirectory(dbops/operators/tests)



