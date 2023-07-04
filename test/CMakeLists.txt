# Add Catch2 as dependency
Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.3.2 # or a later release
)

FetchContent_MakeAvailable(Catch2)



add_subdirectory(column_based)

add_executable(simpleTest simpleTest.cpp)
target_include_directories(simpleTest PUBLIC ${TSL_INCLUDE_DIRECTORY} ${SIMDOperators_INCLUDE_DIRECTORY})
target_link_libraries(simpleTest tsl SIMDOperators)

add_executable(operatorTest operatorTest.cpp)
target_include_directories(operatorTest PUBLIC ${TSL_INCLUDE_DIRECTORY} ${SIMDOperators_INCLUDE_DIRECTORY})
target_link_libraries(operatorTest tsl SIMDOperators Catch2::Catch2WithMain)

add_executable(joinTest joinTest.cpp)
target_include_directories(joinTest PUBLIC ${TSL_INCLUDE_DIRECTORY} ${SIMDOperators_INCLUDE_DIRECTORY})
target_link_libraries(joinTest tsl SIMDOperators Catch2::Catch2WithMain)

message(STATUS "SIMDOperators_INCLUDE_DIRECTORY: ${SIMDOperators_INCLUDE_DIRECTORY}")
message(STATUS "TSL_INCLUDE_DIRECTORY: ${TSL_INCLUDE_DIRECTORY}")
