# create_test(
# TARGET_NAME select_test
# SRC_FILES algorithms/dbops/select_test.cpp
# INC_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}
# )

create_test(
  TARGET_NAME arithmetic_add_test
  SRC_FILES algorithms/dbops/arithmetic_add_test.cpp
  INC_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}
  LIBRARIES pthread
)

create_test(
  TARGET_NAME arithmetic_div_test
  SRC_FILES algorithms/dbops/arithmetic_div_test.cpp
  INC_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}
  LIBRARIES pthread
)

create_test(
  TARGET_NAME arithmetic_mul_test
  SRC_FILES algorithms/dbops/arithmetic_mul_test.cpp
  INC_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}
  LIBRARIES pthread
)

create_test(
  TARGET_NAME arithmetic_sub_test
  SRC_FILES algorithms/dbops/arithmetic_sub_test.cpp
  INC_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}
  LIBRARIES pthread
)

create_test(
  TARGET_NAME groupby_test
  SRC_FILES algorithms/dbops/groupby_test.cpp
  INC_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}
  LIBRARIES pthread
)

create_test(
  TARGET_NAME groupby_sum_test
  SRC_FILES algorithms/dbops/groupsum_test.cpp
  INC_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}
  LIBRARIES pthread
)

create_test(
  TARGET_NAME join_test
  SRC_FILES algorithms/dbops/join_test.cpp
  INC_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}
  LIBRARIES pthread
)

create_test(
  TARGET_NAME sort_direct_test
  SRC_FILES algorithms/dbops/sort_direct_test.cpp
  INC_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}
  LIBRARIES pthread
)

create_test(
  TARGET_NAME sort_indirect_test
  SRC_FILES algorithms/dbops/sort_indirect_test.cpp
  INC_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}
  LIBRARIES pthread
)

create_test(
  TARGET_NAME sort_indirect_clustered_test
  SRC_FILES algorithms/dbops/sort_with_clusters.cpp
  INC_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}
  LIBRARIES pthread
)