add_executable(FilterTest filter_test.cpp)
target_compile_features(FilterTest PUBLIC cxx_std_20)
target_link_libraries(FilterTest SIMDOperators)
target_include_directories(FilterTest PUBLIC ${PROJECT_SOURCE_DIR}/../include)