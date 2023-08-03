# Add Catch2 as dependency
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.3.2 # or a later release
)

FetchContent_MakeAvailable(Catch2)


# Adds a test executable with corresponding sources and links it to TSL, SIMDOperators and Catch2
# Usage: add_test(<target> <source1> <source2> ...)
function(add_test target)
  message("add_test for ${target} with ${ARGN}")
  foreach(arg IN LISTS ARGN)
    set( ALL_TEST_FILES ${ALL_TEST_FILES} ${arg})
  endforeach()
  set( ALL_TEST_FILES ${ALL_TEST_FILES} PARENT_SCOPE)
  add_executable(${target} ${ARGN})
  target_include_directories(${target} PUBLIC ${TSL_INCLUDE_DIRECTORY} ${SIMDOperators_INCLUDE_DIRECTORY})
  target_link_libraries(${target} tsl SIMDOperators Catch2::Catch2WithMain)
endfunction()


# add_subdirectory(column_based)

# add_executable(simpleTest simpleTest.cpp)
# target_include_directories(simpleTest PUBLIC ${TSL_INCLUDE_DIRECTORY} ${SIMDOperators_INCLUDE_DIRECTORY})
# target_link_libraries(simpleTest tsl SIMDOperators)


add_test(selectTest ${CMAKE_CURRENT_SOURCE_DIR}/operators/select_test.cpp)
add_test(calcTest ${CMAKE_CURRENT_SOURCE_DIR}/operators/calc_test.cpp)


add_executable(AllTests ${ALL_TEST_FILES})
target_include_directories(AllTests PUBLIC ${TSL_INCLUDE_DIRECTORY} ${SIMDOperators_INCLUDE_DIRECTORY})
target_link_libraries(AllTests tsl SIMDOperators Catch2::Catch2WithMain)


add_executable(joinTest joinTest.cpp)
target_include_directories(joinTest PUBLIC ${TSL_INCLUDE_DIRECTORY} ${SIMDOperators_INCLUDE_DIRECTORY})
target_link_libraries(joinTest tsl SIMDOperators Catch2::Catch2WithMain)

message(STATUS "SIMDOperators_INCLUDE_DIRECTORY: ${SIMDOperators_INCLUDE_DIRECTORY}")
message(STATUS "TSL_INCLUDE_DIRECTORY: ${TSL_INCLUDE_DIRECTORY}")
