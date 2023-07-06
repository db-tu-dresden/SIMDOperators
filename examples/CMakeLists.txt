
# HelloTSL
add_executable(HelloTSL HelloTSL/HelloTSL.cpp)
target_include_directories(HelloTSL PUBLIC ${TSL_INCLUDE_DIRECTORY} ${SIMDOperators_INCLUDE_DIRECTORY})
target_link_libraries(HelloTSL tsl SIMDOperators)
