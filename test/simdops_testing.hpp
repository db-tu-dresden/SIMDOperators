// ------------------------------------------------------------------- //
/*
   This file is part of the SimdOperators Project.
   Author(s): Johannes Pietrzyk.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, version 3.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
// ------------------------------------------------------------------- //
/**
 * @file simdops_testing.hpp
 * @brief
 */
#ifndef SIMDOPS_TEST_SIMDOPS_TESTING_HPP
#define SIMDOPS_TEST_SIMDOPS_TESTING_HPP

#include <cxxabi.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <vector>

template <class T>
std::string type_name() {
  typedef typename std::remove_reference<T>::type TR;
  std::unique_ptr<char, void (*)(void*)> own(abi::__cxa_demangle(typeid(TR).name(), nullptr, nullptr, nullptr),
                                             std::free);
  std::string r = own != nullptr ? own.get() : typeid(TR).name();
  if (std::is_const<TR>::value) {
    r += " const";
  }
  if (std::is_volatile<TR>::value) {
    r += " volatile";
  }
  if (std::is_lvalue_reference<T>::value) {
    r += "&";
  } else if (std::is_rvalue_reference<T>::value) {
    r += "&&";
  }
  return r;
}
#define TYPENAME(x) type_name<decltype(x)>()

struct scalar {};
struct sse {};
struct avx {};

template <typename T, typename Extension>
struct simd {
  using base_type = T;
};

class Runtime {
 public:
  using extension = std::tuple<scalar, sse, avx>;
  template <typename T>
  using simdStyles = std::tuple<simd<T, scalar>, simd<T, sse>, simd<T, avx>>;
};

namespace details {
  namespace types {
    template <class, class>
    struct tuple_type_cat;
    template <class... First, class... Second>
    struct tuple_type_cat<std::tuple<First...>, std::tuple<Second...>> {
      using type = std::tuple<First..., Second...>;
    };

    template <class PartitionFunction, class PartitionArgsTuple>
    struct partition_result;
    template <class PartitionFunction, class... PartitionArgs>
    struct partition_result<PartitionFunction, std::tuple<PartitionArgs...>> {
      using type = decltype(std::declval<PartitionFunction>()(std::declval<size_t>(), std::declval<size_t>(),
                                                              std::declval<PartitionArgs>()...));
    };
    template <class PartitionFunction, class PartitionArgsTuple>
    using partition_result_t = typename partition_result<PartitionFunction, PartitionArgsTuple>::type;

    template <class ConcreteSimdOp, class PartitionResultT, class SimdOperatorCallArgsTuple>
    struct partition_and_call_result;
    template <class ConcreteSimdOp, typename... PartitionResultTs, typename... SimdOperatorCallArgs>
    struct partition_and_call_result<ConcreteSimdOp, std::tuple<PartitionResultTs...>,
                                     std::tuple<SimdOperatorCallArgs...>> {
      using type = decltype(std::declval<ConcreteSimdOp>()(std::declval<PartitionResultTs>()...,
                                                           std::declval<SimdOperatorCallArgs>()...));
    };

  }  // namespace types
  template <class ConcreteSimdOp, class PartitionFunction, class PartitionArgsTuple, typename... SimdOperatorCallArgs>
  struct partition_and_call_result {
    using partition_fun_type = types::partition_result_t<PartitionFunction, PartitionArgsTuple>;
    using type = typename types::partition_and_call_result<ConcreteSimdOp, partition_fun_type,
                                                           std::tuple<SimdOperatorCallArgs...>>::type;
  };
  template <class ConcreteSimdOp, class PartitionFunction, class PartitionArgsTuple, typename... SimdOperatorCallArgs>
  using partition_and_call_result_t =
    typename partition_and_call_result<ConcreteSimdOp, PartitionFunction, PartitionArgsTuple,
                                       SimdOperatorCallArgs...>::type;

  template <typename SimdExtensionTuple, template <typename...> class SimdOp, typename... SimdOpTs>
  struct simdops_tester_impl;
  template <typename... SimdExtension, template <typename...> class SimdOp, typename... SimdOpTs>
  struct simdops_tester_impl<std::tuple<SimdExtension...>, SimdOp, SimdOpTs...> {
    template <class ConcreteSimdOp, class SimdOperatorCtorArgsTuple, std::size_t... I>
    constexpr static decltype(auto) create_simdop(SimdOperatorCtorArgsTuple&& args, std::index_sequence<I...>) {
      return ConcreteSimdOp(std::get<I>(std::forward<SimdOperatorCtorArgsTuple>(args))...);
    }
    template <class ConcreteSimdOp, class CallArgsTuple, std::size_t... I>
    constexpr static decltype(auto) call(ConcreteSimdOp& op, CallArgsTuple&& args, std::index_sequence<I...>) {
      return op(std::get<I>(std::forward<CallArgsTuple>(args))...);
    }
    template <class PartitionFunction, class PartitionArgsTuple, std::size_t... I>
    constexpr static decltype(auto) exec_partition(PartitionFunction&& partition_fun, size_t partition_count,
                                                   size_t partition, PartitionArgsTuple&& partition_args,
                                                   std::index_sequence<I...>) {
      return partition_fun(partition_count, partition,
                           std::get<I>(std::forward<PartitionArgsTuple>(partition_args))...);
    }
    template <class ConcreteSimdOp, class PartitionFunction, class PartitionArgsTuple, typename... SimdOperatorCallArgs>
    constexpr static decltype(auto) partition_and_call(ConcreteSimdOp& op, PartitionFunction&& partition_fun,
                                                       size_t partition_count, size_t partition,
                                                       PartitionArgsTuple&& partition_args,
                                                       SimdOperatorCallArgs&&... simd_op_function_args) {
      auto params = exec_partition(std::forward<PartitionFunction>(partition_fun), partition_count, partition,
                                   std::forward<PartitionArgsTuple>(partition_args),
                                   std::make_index_sequence<std::tuple_size_v<PartitionArgsTuple>>{});
      return call(op, std::tuple_cat(params, std::forward_as_tuple(simd_op_function_args)...),
                  std::make_index_sequence<std::tuple_size_v<PartitionArgsTuple> + sizeof...(SimdOperatorCallArgs)>{});
    }

    template <class ConcreteSimdOp, class PartitionArgsTuple, class PartitionFunction, class SimdOperatorCtorArgsTuple,
              typename... SimdOperatorCallArgs>
    constexpr static decltype(auto) run_single_test_no_result_with_partitioning(
      PartitionArgsTuple&& partition_args, size_t partition_count, PartitionFunction&& partition_fun,
      SimdOperatorCtorArgsTuple&& simd_op_ctor_args, SimdOperatorCallArgs&&... simd_op_function_args) {
      std::vector<ConcreteSimdOp> operators;
      for (size_t partition = 0; partition < partition_count; ++partition) {
        auto simdop =
          create_simdop<ConcreteSimdOp>(std::forward<SimdOperatorCtorArgsTuple>(simd_op_ctor_args),
                                        std::make_index_sequence<std::tuple_size_v<SimdOperatorCtorArgsTuple>>{});
        partition_and_call(simdop, std::forward<PartitionFunction>(partition_fun), partition_count, partition,
                           std::forward<PartitionArgsTuple>(partition_args),
                           std::forward<SimdOperatorCallArgs>(simd_op_function_args)...);
        operators.push_back(simdop);
      }
      for (size_t partition = 1; partition < partition_count; ++partition) {
        operators[0].merge(operators[partition]);
      }
      return operators[0].finalize();
    }
    template <class ConcreteSimdOp, class PartitionArgsTuple, class PartitionFunction, class SimdOperatorCtorArgsTuple,
              typename... SimdOperatorCallArgs>
    constexpr static decltype(auto) run_single_test_with_result_with_partitioning(
      PartitionArgsTuple&& partition_args, size_t partition_count, PartitionFunction&& partition_fun,
      SimdOperatorCtorArgsTuple&& simd_op_ctor_args, SimdOperatorCallArgs&&... simd_op_function_args) {
      std::vector<ConcreteSimdOp> operators;
      std::vector<
        partition_and_call_result_t<ConcreteSimdOp, PartitionFunction, PartitionArgsTuple, SimdOperatorCallArgs...>>
        partition_and_call_results;
      for (size_t partition = 0; partition < partition_count; ++partition) {
        auto simdop =
          create_simdop<ConcreteSimdOp>(std::forward<SimdOperatorCtorArgsTuple>(simd_op_ctor_args),
                                        std::make_index_sequence<std::tuple_size_v<SimdOperatorCtorArgsTuple>>{});
        auto partition_and_call_result =
          partition_and_call(simdop, std::forward<PartitionFunction>(partition_fun), partition_count, partition,
                             std::forward<PartitionArgsTuple>(partition_args),
                             std::forward<SimdOperatorCallArgs>(simd_op_function_args)...);
        partition_and_call_results.push_back(partition_and_call_result);
        operators.push_back(simdop);
      }
      for (size_t partition = 1; partition < partition_count; ++partition) {
        operators[0].merge(operators[partition]);
      }
      return operators[0].finalize();
    }

    template <class ConcreteSimdOp, class PartitionArgsTuple, class PartitionFunction, class SimdOperatorCtorArgsTuple,
              typename... SimdOperatorCallArgs>
    constexpr static decltype(auto) run_single_test_with_partitioning(PartitionArgsTuple&& partition_args,
                                                                      size_t partition_count,
                                                                      PartitionFunction&& partition_fun,
                                                                      SimdOperatorCtorArgsTuple&& simd_op_ctor_args,
                                                                      SimdOperatorCallArgs&&... simd_op_function_args) {
      if constexpr (std::is_void_v<partition_and_call_result_t<ConcreteSimdOp, PartitionFunction, PartitionArgsTuple,
                                                               SimdOperatorCallArgs...>>) {
        return run_single_test_no_result_with_partitioning<ConcreteSimdOp>(
          std::forward<PartitionArgsTuple>(partition_args), partition_count,
          std::forward<PartitionFunction>(partition_fun), std::forward<SimdOperatorCtorArgsTuple>(simd_op_ctor_args),
          std::forward<SimdOperatorCallArgs>(simd_op_function_args)...);
      } else {
        return run_single_test_with_result_with_partitioning<ConcreteSimdOp>(
          std::forward<PartitionArgsTuple>(partition_args), partition_count,
          std::forward<PartitionFunction>(partition_fun), std::forward<SimdOperatorCtorArgsTuple>(simd_op_ctor_args),
          std::forward<SimdOperatorCallArgs>(simd_op_function_args)...);
      }
    }

    template <typename PartitionArgsTuple, typename PartitionFunction, typename SimdOperatorCtorArgsTuple,
              typename... SimdOperatorCallArgs>
    constexpr static void run_extension_tests_with_partitioning(PartitionArgsTuple&& partition_args,
                                                                size_t partition_count,
                                                                PartitionFunction&& partition_fun,
                                                                SimdOperatorCtorArgsTuple&& simd_op_ctor_args,
                                                                SimdOperatorCallArgs&&... simd_op_function_args) {
      (run_single_test_with_partitioning<SimdOp<SimdExtension, SimdOpTs...>>(
         std::forward<PartitionArgsTuple>(partition_args), partition_count,
         std::forward<PartitionFunction>(partition_fun), std::forward<SimdOperatorCtorArgsTuple>(simd_op_ctor_args),
         std::forward<SimdOperatorCallArgs>(simd_op_function_args)...),
       ...);
    }
  };
}  // namespace details

template <class RT, typename T, template <typename...> class SimdOp>
struct simdops_tester {
  template <typename PartitionArgsTuple, typename PartitionFunction, typename SimdOperatorCtorArgsTuple,
            typename... SimdOperatorCallArgs>
  constexpr static auto run_tests_with_partitioning(PartitionArgsTuple&& partition_args, size_t partition_count,
                                                    PartitionFunction&& partition_fun,
                                                    SimdOperatorCtorArgsTuple&& simd_op_ctor_args,
                                                    SimdOperatorCallArgs&&... simd_op_function_args) {
    using SimdExtensionTuple = typename RT::template simdStyles<T>;
    return details::simdops_tester_impl<SimdExtensionTuple, SimdOp>::run_extension_tests_with_partitioning(
      std::forward<PartitionArgsTuple>(partition_args), partition_count, std::forward<PartitionFunction>(partition_fun),
      std::forward<SimdOperatorCtorArgsTuple>(simd_op_ctor_args),
      std::forward<SimdOperatorCallArgs>(simd_op_function_args)...);
  }
};

template <class SimdExtension>
struct Select {
 private:
  int el_count;
  int const add;

 public:
  Select(int _add) : el_count(0), add(_add) {
    std::cout << "Select<" << type_name<SimdExtension>() << ">::ctor(" << add << ")" << std::endl;
  }
  auto operator()(typename SimdExtension::base_type* out, typename SimdExtension::base_type const* in, size_t count,
                  typename SimdExtension::base_type const mulitply) {
    std::cout << "Run operator()(" << (void*)out << ", " << (void*)in << ", " << count << ", " << mulitply << ")"
              << " -> Out: " << (void*)(out + count) << " In: " << (void*)(in + count) << std::endl;
    for (int i = 0; i < count; ++i) {
      out[i] = (in[i] + add) * mulitply;
    }
    el_count += count;
  }
  auto merge(Select const& other) {
    std::cout << "Run merge()" << std::endl;
    el_count += other.el_count;
  }
  auto finalize() {
    std::cout << "Run finalize() -> " << el_count << std::endl;
    return el_count;
  }
};
}  // namespace tuddbs
#endif
int main() {
  auto out_mem = new int[130];
  auto in_mem = new int[130];
  for (size_t i = 0; i < 130; ++i) {
    in_mem[i] = i;
  }

  using tester = simdops_tester<Runtime, int, Select>;
  tester::run_tests_with_partitioning(
    std::make_tuple(out_mem, in_mem, 130), 4,
    [](size_t partition_count, size_t partition_id, auto out, auto in, size_t element_count) {
      auto partition_offset = (element_count / partition_count) * partition_id;
      auto partition_elements = ((element_count % partition_count) == 0) ? element_count / partition_count
                                : (partition_id != (partition_count - 1))
                                  ? element_count / partition_count
                                  : (element_count / partition_count) + (element_count % partition_count);
      return std::make_tuple(out + partition_offset, in + partition_offset, partition_elements);
    },
    std::make_tuple(12), 2);
}
