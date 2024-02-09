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

#include <cstddef>
#include <tuple>
#include <variant>

#include "tslintrin.hpp"

namespace tuddbs {
  template <template <typename> class Functor, class Runtime>
  class operator_testing_factory {
   public:
    template <typename T, typename ExtensionTuple>
    struct functor_variant;
    template <typename T, typename... Extension>
    struct functor_variant<T, std::tuple<Extension...>> {
      using type = std::variant<Functor<tsl::simd<T, Extension>>...>;
    };
    template <typename T>
    using variant_type = typename functor_variant<T, typename Runtime::available_extensions_tuple>::type;
    template <typename T>
    using functors_type = std::array<variant_type<T>, std::tuple_size_v<typename Runtime::available_extensions_tuple>>;

   private:
    template <typename T, size_t... Idx, typename... Args>
    static auto instantiate_simdops_impl(std::index_sequence<Idx...>, Args... args) -> functors_type<T> {
      return {
        Functor<tsl::simd<T, std::tuple_element_t<Idx, typename Runtime::available_extensions_tuple>>>(args...)...};
    }

   public:
    template <typename T, typename... Args>
    static auto instantiate_simdops(Args... args) -> functors_type<T> {
      return instantiate_simdops_impl<T>(
        std::make_index_sequence<std::tuple_size_v<typename Runtime::available_extensions_tuple>>{}, args...);
    }
  };

  template <typename T, template <typename> class Functor, class Runtime, typename... Args>
  auto instantiate_simdops(Args... args) ->
    typename operator_testing_factory<Functor, Runtime>::template functors_type<T> {
    return operator_testing_factory<Functor, Runtime>::template instantiate_simdops<T>(args...);
  }

}  // namespace tuddbs
#endif
