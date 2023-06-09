
# /// Create SIMDOperators library
add_library(SIMDOperators INTERFACE)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/SIMDOperators)
set(SIMDOperators_INCLUDE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

# /// Add all include directories in SIMDOperators_includePaths
target_include_directories(SIMDOperators INTERFACE ${TSL_INCLUDE_DIRECTORY} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SIMDOperators INTERFACE tsl)


target_sources(SIMDOperators 
    INTERFACE 
        ${SRC_DIR}/datastructure/column.hpp
        ${SRC_DIR}/utils/types.hpp
        ${SRC_DIR}/operators/project.hpp
)   


