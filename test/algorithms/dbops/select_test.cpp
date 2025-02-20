#include <catch2/catch_test_macros.hpp>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <random>

#include "algorithms/dbops/filter/filter.hpp"
#include "simdops_testing.hpp"
#include "tslCPUrt.hpp"
#include "tsl.hpp"

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

auto single_predicate_scalar_reference = []<typename T, int VectorElementCountToTest, template <typename> class Comp>(
                                           void* output, T* input, size_t element_count, T predicate) {
  Comp<T> comp;
  using result_type =
    std::conditional_t<VectorElementCountToTest <= 8, uint8_t,
                       std::conditional_t<VectorElementCountToTest <= 16, uint16_t,
                                          std::conditional_t<VectorElementCountToTest <= 32, uint32_t, uint64_t> > >;
  auto* output_reference = reinterpret_cast<result_type*>(output);
  for (auto i = 0; i < element_count; i += VectorElementCountToTest) {
    result_type mask = 0;
    for (auto j = 0; j < VectorElementCountToTest; ++j) {
      mask |= comp(input[i + j], predicate) ? 1 << j : 0;
    }
    *output_reference++ = mask;
  }

  if ((element_count % VectorElementCountToTest) != 0) {
    auto remainder = VectorElementCountToTest - (element_count % VectorElementCountToTest);
    result_type mask = 0;
    for (auto j = 0; j < remainder; ++j) {
      mask |= comp(input[element_count - remainder + j], predicate) ? 1 << j : 0;
    }
    *output_reference++ = mask;
  }
};

TEST_CASE("filter equality for uint64_t", "[cpu][select_eq_bm][uint64_t]") {
  using cpu_executor = tsl::executor<tsl::runtime::cpu>;
  cpu_executor exec;

  using tester = simdops_tester<cpu_executor, uint64_t, FilterEQ_BM>;

  auto predicate_true = 1;
  auto predicate_false = 0;
  {
    size_t element_count = 128;
    auto input = exec.allocate<uint64_t>(element_count, 64);
    auto output = exec.allocate<uint64_t>(element_count, 64);
    auto output_reference = exec.allocate<uint64_t>(element_count, 64);
    auto positions = exec.allocate<size_t>(element_count, 64);
    init_positions(positions, element_count);

    for (size_t selectivity = 0; selectivity <= 100; ++selectivity) {
      prepare_data(output, input, element_count, positions, selectivity, predicate_true, predicate_false);

      tester::run_tests_without_partitioning(

        single_predicate_scalar_reference.operator()<std::equal_to>(output_reference, input, element_count,
                                                                    predicate_true),

        std::make_tuple(output, input, element_count), std::make_tuple(predicate_true)

          )

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