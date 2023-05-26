
# /// Create SIMDOperators library
add_library(SIMDOperators INTERFACE)

set(SIMDOperators_INCLUDE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)

# /// Add all include directories in SIMDOperators_includePaths
target_include_directories(SIMDOperators INTERFACE ${TSL_INCLUDE_DIRECTORY} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(SIMDOperators INTERFACE tsl)

target_sources(SIMDOperators 
    INTERFACE 
        ${CMAKE_CURRENT_SOURCE_DIR}/datastructure/column.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/types.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/operators/project.hpp
)   


