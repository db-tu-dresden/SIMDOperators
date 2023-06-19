// ------------------------------------------------------------------- //
/*
   This file is part of the SimdOperators Project.
   Copyright (c) 2022 Johannes Pietrzyk.
   
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

/*
 * @file types.hpp
 * @author jpietrzyk
 * @date 14.07.22
 * @brief A brief description.
 *
 * @details A detailed description.
 */

#ifndef SIMDOPERATORS_DBOPS_UTILS_INCLUDE_COLUMN_HPP
#define SIMDOPERATORS_DBOPS_UTILS_INCLUDE_COLUMN_HPP

#include <cstddef>
#include <cstdint>
#if (__cplusplus == 202002L)
#include <concepts>
#endif

#include <tslintrin.hpp>

namespace tuddbs{
//alias
using size_t = std::size_t;

//traits
template <typename T, typename TupleType>
struct all_of_specific_type;
template <typename T, typename... Us>
struct all_of_specific_type<T, std::tuple<Us...>> : std::conjunction<std::is_same<T, Us>...> {};

//concepts
#if (__cplusplus == 202002L)
template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;
template<typename T>
concept DataProviderType =
  requires {
      typename T::base_type;
  } &&
  requires(T const & o){
      { o.element_count() } -> std::same_as<size_t>;
  };
template<typename T>
concept DataSourceType = DataProviderType<T> &&
  requires(T const & o) {
    { o.data() } -> std::same_as<typename T::base_type const *>;
  };
template<typename T>
concept DataSinkType = DataProviderType<T> &&
  requires(T const & o) {
    { o.data() } -> std::same_as<typename T::base_type const *>;
  };
#else 
#define Arithmetic typename
#define DataProviderType typename
#define DataSourceType typename
#define DataSinkType typename
#endif


//Helper Structs
//todo: This should be in the TVL!!!
namespace details {
using namespace tsl;
template<VectorProcessingStyle Vec, Arithmetic... Ts, std::size_t... I>
auto broadcast_from_tuple_impl(std::tuple<Ts...> const & tup, std::index_sequence<I...>) {
  return std::make_tuple(tsl::set1<Vec>(std::get<I>(tup))...);
}
}
using namespace tsl;
template<VectorProcessingStyle Vec, Arithmetic... Ts>
constexpr auto broadcast_from_tuple(std::tuple<Ts...> tup) {
  static_assert(all_of_specific_type<typename Vec::base_type, std::tuple<Ts...>>::value, "Parameters has to be of same type as the specified Vector type.");
  return details::broadcast_from_tuple_impl<Vec>(tup, std::make_index_sequence<sizeof...(Ts)>{});
}
}

#endif//SIMDOPERATORS_DBOPS_UTILS_INCLUDE_COLUMN_HPP
