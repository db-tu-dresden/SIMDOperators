# create_test(
#   TARGET_NAME select_test
#   SRC_FILES algorithms/dbops/select_test.cpp
#   INC_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}
# )

create_test(
  TARGET_NAME groupby_test
  SRC_FILES algorithms/dbops/groupby_test.cpp
  INC_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}
  LIBRARIES pthread
)