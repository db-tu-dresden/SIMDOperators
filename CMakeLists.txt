cmake_minimum_required(VERSION 3.15)
project(SimdOperators)


if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL "${PROJECT_SOURCE_DIR}")
  set(SIMDOPS_TESTING ON)
  set(SIMDOPS_BENCHMARKING ON)
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/simdops.cmake)


#[[
  If this is the top level CMakeLists.txt file, we should include building tests and benchmarks, regardless of anything else.
  If not, benchmarks can be explicitly included by setting the SIMDOPS_BENCHMARKS variable to ON.
  Same applies to tests (setting SIMDOPS_TESTING to ON).
]]
if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL "${PROJECT_SOURCE_DIR}")
  if(NOT CMAKE_BUILD_TYPE)
    message(FATAL_ERROR "Specifying CMAKE_BUILD_TYPE")
  else()
    message(STATUS "CMAKE_BUILD_TYPE = ${CMAKE_BUILD_TYPE}")
  endif()
  add_subdirectory(test)
  add_subdirectory(benchmarks)
  add_subdirectory(examples)  
else()
  if((ENABLE_TESTING) OR (SIMDOPS_TESTING))
    add_subdirectory(test)
  endif()
  if(SIMDOPS_BENCHMARKING)
    add_subdirectory(benchmarks)
  endif()
endif() 