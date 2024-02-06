#include "algorithms/dbops/select.hpp"

#include <catch2/catch_test_macros.hpp>

#include "tslCPUrt.hpp"
#include "tslintrin.hpp"

using namespace tuddbs;

TEST_CASE("filter equality for uint8_t", "[select_eq]") {
  using cpu_executor = tsl::executor<tsl::runtime::cpu>;
  cpu_executor exec;
  for (auto const parallel_degree : cpu_executor::template available_parallelism<uint8_t>()) {
    std::cerr << "parallel_degree: " << parallel_degree << std::endl;
  }
}