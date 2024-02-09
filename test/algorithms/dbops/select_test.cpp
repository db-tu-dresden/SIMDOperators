#include "algorithms/dbops/select.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>

#include "simdops_testing.hpp"
#include "tslCPUrt.hpp"
#include "tslintrin.hpp"

using namespace tuddbs;

void prepare_data(uint64_t* output, uint64_t* input, size_t element_count, size_t* positions, size_t selectivity,
                  uint64_t predicate_true, uint64_t predicate_false) {
  for (auto i = 0; i < element_count; ++i) {
    input[i] = predicate_false;
    output[i] = 0;
  }
  for (auto i = 0; i < (element_count * selectivity / 100); ++i) {
    input[positions[i]] = predicate_true;
  }
}

void init_positions(size_t* positions, size_t element_count) {
  for (auto i = 0; i < element_count; ++i) {
    positions[i] = i;
  }
  std::shuffle(positions, positions + element_count, std::mt19937{std::random_device{}()});
}

TEST_CASE("filter equality for uint64_t", "[cpu][select_eq_bm][uint64_t]") {
  using cpu_executor = tsl::executor<tsl::runtime::cpu>;
  cpu_executor exec;

  auto scalar_reference = [](uint64_t* output, uint64_t* input, size_t element_count, uint64_t predicate) {
    for (auto i = 0; i < element_count; i += 64) {
      std::bitset<64> mask{0};
      for (auto j = 0; j < 64; ++j) {
        mask[j] = input[i + j] == predicate;
      }
      output[i / 64] = mask.to_ullong();
    }
    if ((element_count % 64) != 0) {
      auto remainder = 64 - (element_count % 64);
      std::bitset<64> mask{0};
      for (auto j = 0; j < remainder; ++j) {
        mask[j] = input[element_count - remainder + j] == predicate;
      }
      output[element_count / 64] = mask.to_ullong();
    }
  };

  auto predicate_true = 1;
  auto predicate_false = 0;
  {
    size_t element_count = 128;
    auto input = exec.allocate<uint64_t>(element_count, 64);
    auto output = exec.allocate<uint64_t>(element_count, 64);
    auto output_reference = exec.allocate<uint64_t>(element_count, 64);
    auto positions = exec.allocate<size_t>(element_count, 64);
    init_positions(positions, element_count);

    auto operators = tuddbs::instantiate_simdops<uint64_t, tuddbs::FilterEQ_BM, tsl::runtime::cpu>(predicate_true);

    for (size_t selectivity = 0; selectivity <= 100; ++selectivity) {
      prepare_data(output, input, element_count, positions, selectivity, predicate_true, predicate_false);
      scalar_reference(output_reference, input, element_count, predicate_true);

      // auto fn = tuddbs::FilterEQ_BM<tsl::simd<uint64_t, tsl::avx2>>(predicate_true);
      // exec.submit(fn, reinterpret_cast<typename decltype(fn)::result_base_type*>(output), input, element_count,
      // tuddbs::bit_mask{});
      for (auto& op : operators) {
        auto sink = std::visit(
          [output](auto&& _arg) { return reinterpret_cast<std::decay_t<decltype(_arg)>::result_base_type*>(output); },
          op);
        exec.submit(op, sink, input, element_count, tuddbs::bit_mask{});
        // exec.submit(std::bind(&decltype(op)::finalize, &op));
        REQUIRE(std::memcmp(output, output_reference, element_count * sizeof(uint64_t)) == 0);
      }
    }
  }
}