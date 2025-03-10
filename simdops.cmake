cmake_minimum_required(VERSION 3.15)

set(SIMDOPS_CXX_STANDARD 20)
set(SIMDOPS_ROOT ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "Root directory of the SIMDOPS library")
set(SIMDOPS_INCLUDE ${SIMDOPS_ROOT}/include CACHE STRING "Include directory of the SIMDOPS library")
set(SIMDOPS_TSLGEN_ROOT ${SIMDOPS_ROOT}/tools/tslgen CACHE STRING "Root directory of the TSL generator")
set(SIMDOPS_TSL_ROOT ${SIMDOPS_ROOT}/lib/tsl CACHE STRING "Root directory of the TSL library")
set(SIMDOPS_BIN_ROOT ${SIMDOPS_ROOT}/bin CACHE STRING "Root directory of the SIMDOPS binaries")

set(SIMDOPS_CATCH2_ROOT ${SIMDOPS_ROOT}/tools/Catch2 CACHE STRING "Root directory of the Catch2 repository")

set(SIMDOPS_CXX_FLAGS_DEBUG -O0 -g)
set(SIMDOPS_CXX_FLAGS_RELEASE -O2)


include(${SIMDOPS_TSLGEN_ROOT}/tsl.cmake)

if((ENABLE_TESTING) OR (SIMDOPS_TESTING))
  add_subdirectory(${SIMDOPS_CATCH2_ROOT})
endif()
if((ENABLE_TESTING) OR (TSL_TESTING))
  
  create_tsl(
    USE_CONCEPTS CREATE_TESTS
    TSLGENERATOR_DIRECTORY ${SIMDOPS_TSLGEN_ROOT}
    DESTINATION ${SIMDOPS_TSL_ROOT}
  )
else()
  create_tsl(
    USE_CONCEPTS 
    TSLGENERATOR_DIRECTORY ${SIMDOPS_TSLGEN_ROOT}
    DESTINATION ${SIMDOPS_TSL_ROOT}
  )
endif()

function(create_target)
  set(oneValueArgs TARGET_NAME OUT_PATH)
  set(multiValueArgs SRC_FILES INC_DIRECTORIES LIBRARIES COMPILE_DEFS COMPILE_OPTS LINK_OPTS)
  cmake_parse_arguments(PARAMS "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if((NOT DEFINED PARAMS_TARGET_NAME) OR (NOT PARAMS_TARGET_NAME))
    message(FATAL_ERROR "create_target: TARGET_NAME is not set")
  endif()
  add_executable(${PARAMS_TARGET_NAME} ${PARAMS_SRC_FILES})
  set_target_properties(${PARAMS_TARGET_NAME} PROPERTIES 
    CXX_STANDARD ${SIMDOPS_CXX_STANDARD} 
    CXX_STANDARD_REQUIRED ON 
    RUNTIME_OUTPUT_DIRECTORY ${PARAMS_OUT_PATH}
    INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  target_include_directories(${PARAMS_TARGET_NAME} PRIVATE 
    ${SIMDOPS_INCLUDE} ${TSL_INCLUDE_DIRECTORY} ${TSL_INCLUDE_DIRECTORY_ROOT} ${PARAMS_INC_DIRECTORIES})
  target_compile_definitions(${PARAMS_TARGET_NAME} PRIVATE 
    ${PARAMS_COMPILE_DEFS})
  target_compile_options(
    ${PARAMS_TARGET_NAME} PRIVATE 
      ${PARAMS_COMPILE_OPTS} 
      $<$<CONFIG:Release>:${SIMDOPS_CXX_FLAGS_RELEASE}> 
      $<$<CONFIG:Debug>:${SIMDOPS_CXX_FLAGS_DEBUG}>
  )
  target_link_options(${PARAMS_TARGET_NAME} PRIVATE 
    ${PARAMS_LINK_OPTS})
  target_link_libraries(${PARAMS_TARGET_NAME} PRIVATE 
    tsl ${PARAMS_LIBRARIES})
endfunction()

function(create_test)
  set(oneValueArgs TARGET_NAME)
  set(multiValueArgs SRC_FILES INC_DIRECTORIES LIBRARIES COMPILE_DEFS COMPILE_OPTS LINK_OPTS)
  cmake_parse_arguments(PARAMS "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  create_target(
    TARGET_NAME ${PARAMS_TARGET_NAME} 
    OUT_PATH ${SIMDOPS_BIN_ROOT}/tests
    SRC_FILES ${PARAMS_SRC_FILES} 
    INC_DIRECTORIES ${PARAMS_INC_DIRECTORIES} 
    LIBRARIES ${PARAMS_LIBRARIES} Catch2::Catch2WithMain 
    COMPILE_DEFS ${PARAMS_COMPILE_DEFS} 
    COMPILE_OPTS ${PARAMS_COMPILE_OPTS} 
    LINK_OPTS ${PARAMS_LINK_OPTS})
endfunction()

function(create_benchmark)
  set(oneValueArgs TARGET_NAME)
  set(multiValueArgs SRC_FILES INC_DIRECTORIES LIBRARIES COMPILE_DEFS COMPILE_OPTS LINK_OPTS)
  cmake_parse_arguments(PARAMS "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  create_target(
    TARGET_NAME ${PARAMS_TARGET_NAME} 
    OUT_PATH ${SIMDOPS_BIN_ROOT}/benchmarks
    SRC_FILES ${PARAMS_SRC_FILES} 
    INC_DIRECTORIES ${PARAMS_INC_DIRECTORIES} 
    LIBRARIES ${PARAMS_LIBRARIES} 
    COMPILE_DEFS ${PARAMS_COMPILE_DEFS} 
    COMPILE_OPTS ${PARAMS_COMPILE_OPTS} 
    LINK_OPTS ${PARAMS_LINK_OPTS})
endfunction()


function(create_example)
  set(oneValueArgs TARGET_NAME)
  set(multiValueArgs SRC_FILES INC_DIRECTORIES LIBRARIES COMPILE_DEFS COMPILE_OPTS LINK_OPTS)
  cmake_parse_arguments(PARAMS "" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  create_target(
    TARGET_NAME ${PARAMS_TARGET_NAME} 
    OUT_PATH ${SIMDOPS_BIN_ROOT}/examples
    SRC_FILES ${PARAMS_SRC_FILES} 
    INC_DIRECTORIES ${PARAMS_INC_DIRECTORIES} 
    LIBRARIES ${PARAMS_LIBRARIES} 
    COMPILE_DEFS ${PARAMS_COMPILE_DEFS} 
    COMPILE_OPTS ${PARAMS_COMPILE_OPTS} 
    LINK_OPTS ${PARAMS_LINK_OPTS})
endfunction()