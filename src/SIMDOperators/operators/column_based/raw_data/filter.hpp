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
 * @file filter.hpp
 * @author jpietrzyk
 * @date 14.07.22
 * @brief A brief description.
 *
 * @details A detailed description.
 */

#ifndef SIMDOPERATORS_DBOPS_OPERATORS_INCLUDE_COLUMN_BASED_FILTER_HPP
#define SIMDOPERATORS_DBOPS_OPERATORS_INCLUDE_COLUMN_BASED_FILTER_HPP

#include <type_traits>

#include <tvlintrin.hpp>
#include <types.hpp>

namespace tuddbs {
using namespace tvl;
template<template<class, class> class Comparator, DataSinkType ResultType, VectorProcessingStyle Vec, Arithmetic... ParamTypes, ImplementationDegreeOfFreedom Idof = tvl::workaround>
  auto select(
    DataSourceType auto const & p_input_container,
    std::tuple<ParamTypes...> p_predicates,
    const size_t p_result_count_estimate = 0
  ) {
    static_assert(all_of_specific_type<typename Vec::base_type, std::tuple<ParamTypes...>>::value, "Predicates has to be of same type as the specified Vector type.");
    using ScalarProcessingStyle = tvl::simd<typename Vec::base_type, tvl::scalar>;

    auto const result_element_count = (p_result_count_estimate == 0) ? p_input_container.element_count() : p_result_count_estimate;
    auto result = new ResultType(result_element_count, Vec::vector_alignment());

    auto input_data_ptr = p_input_container.data();
    auto input_element_count = p_input_container.element_count();

    auto input_data_end_ptr = input_data_ptr + input_element_count;
    auto remainder = input_element_count % Vec::vector_element_count();

    auto const input_data_vectorized_end_ptr = input_data_end_ptr - remainder;

    auto predicates_vec = broadcast_from_tuple<Vec>(p_predicates);
    auto position_vec = []<typename Vec::base_type... Idx>(std::integer_sequence<typename Vec::base_type, Idx...>){return tvl::set<Vec>(Idx...);}(std::make_integer_sequence<typename Vec::base_type, Vec::vector_element_count()>{});
    auto position_incrementor = tvl::set1<Vec>(Vec::vector_element_count());
    auto simd_comparator_lambda = [&]<std::size_t... Idx>(typename Vec::register_type data, std::index_sequence<Idx...>){return Comparator<Vec, Idof>::apply(data, std::get<Idx>(predicates_vec)...);};
    auto comparator_lambda = [&]<std::size_t... Idx>(typename ScalarProcessingStyle::register_type data, std::index_sequence<Idx...>){return Comparator<ScalarProcessingStyle, Idof>::apply(data, std::get<Idx>(p_predicates)...);};


    for(; input_data_ptr<input_data_vectorized_end_ptr; input_data_ptr+=Vec::vector_element_count()) {
      auto result_mask = simd_comparator_lambda(tvl::load<Vec>(input_data_ptr), std::make_index_sequence<sizeof...(ParamTypes)>{});

    }
    for(; input_data_ptr<input_data_end_ptr; ++input_data_ptr) {
      auto result_bit = comparator_lambda(tvl::load<ScalarProcessingStyle>(input_data_ptr), std::make_index_sequence<sizeof...(ParamTypes)>{});
    }

    return result;

  }

}

#endif //SIMDOPERATORS_DBOPS_OPERATORS_INCLUDE_COLUMN_BASED_FILTER_HPP
