add_subdirectory(column_based)

add_executable(simpleTest simpleTest.cpp)
target_include_directories(simpleTest PUBLIC ${TSL_INCLUDE_DIRECTORY} ${SIMDOperators_INCLUDE_DIRECTORY})
target_link_libraries(simpleTest tsl SIMDOperators)

message(STATUS "SIMDOperators_INCLUDE_DIRECTORY: ${SIMDOperators_INCLUDE_DIRECTORY}")
message(STATUS "TSL_INCLUDE_DIRECTORY: ${TSL_INCLUDE_DIRECTORY}")
