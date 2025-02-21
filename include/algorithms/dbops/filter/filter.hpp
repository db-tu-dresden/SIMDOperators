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
 * @file filter.hpp
 * @brief
 */

#pragma once

#include <climits>
#include <cstddef>
#include <tuple>
#include <type_traits>

#include "algorithms/dbops/dbops_hints.hpp"
#include "algorithms/dbops/filter/filter_hints.hpp"
#include "algorithms/utils/hinting.hpp"
#include "iterable.hpp"
#include "tsl.hpp"

namespace tuddbs {
  template <tsl::VectorProcessingStyle _SimdStyle, template <class, class> class CompareFun,
            class HintSet = OperatorHintSet<hints::intermediate::bit_mask>, typename Idof = tsl::workaround>
  class Generic_Filter {
   public:
    using SimdStyle = _SimdStyle;
    using ResultType =
      std::conditional_t<has_hint<HintSet, hints::intermediate::position_list>, size_t, typename SimdStyle::imask_type>;
    using DataSinkType = ResultType *;
    using base_type = typename SimdStyle::base_type;
    using result_base_type = typename SimdStyle::imask_type;

    using CTorParamTupleT = std::tuple<typename SimdStyle::base_type const>;
    static constexpr bool ProducesBitmask = has_hint<HintSet, hints::intermediate::bit_mask>;

    constexpr size_t byte_count(SimdOpsIterable auto p_start, SimdOpsIterableOrSizeT auto p_end) {
      auto it_end = iter_end(p_start, p_end);
      auto dist = it_end - p_start;
      return dist * sizeof(ResultType);
    }

   private:
    using ScalarT = tsl::simd<typename SimdStyle::base_type, tsl::scalar>;
    using UnsignedSimdT = typename SimdStyle::template transform_extension<typename SimdStyle::offset_base_type>;
    using CountSimdT = typename SimdStyle::template transform_extension<size_t>;
    using CountSimdRegisterT = typename SimdStyle::template transform_type<size_t>;

   private:
    typename SimdStyle::base_type const m_predicate_scalar;
    typename SimdStyle::register_type const m_predicate_reg;

   public:
    Generic_Filter(typename SimdStyle::base_type const p_predicate)
      : m_predicate_scalar(p_predicate), m_predicate_reg(tsl::set1<SimdStyle>(p_predicate)) {}

    ~Generic_Filter() = default;

   public:
    template <class HS = HintSet>
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    enable_if_has_hint_t<HS, hints::operators::filter::count_bits, hints::intermediate::bit_mask> =
                      {}) noexcept -> std::tuple<DataSinkType, size_t> {
      // Get the end of the SIMD iteration
      auto const simd_end = simd_iter_end<SimdStyle>(p_data, p_end);

      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      // Get the result pointer
      auto result = reinterpret_iterable<DataSinkType>(p_result);

      auto const m_increment = tsl::set1<UnsignedSimdT>(1);

      auto valid_count = tsl::set1<CountSimdT, Idof>(0);

      typename SimdStyle::register_type data;
      // Iterate over the data simdified and apply the Filter for equality
      for (; p_data != simd_end; p_data += SimdStyle::vector_element_count(), ++result) {
        // Load data from the source
        if constexpr (has_hint<HintSet, hints::memory::aligned>) {
          data = tsl::load<SimdStyle, Idof>(p_data);
        } else {
          data = tsl::loadu<SimdStyle, Idof>(p_data);
        }
        // Compare the data with the predicate producing a mask type (either
        // register or integral type)
        auto mask = CompareFun<SimdStyle, Idof>::apply(data, m_predicate_reg);
        // Store the result as an integral value into the data sink
        tsl::store_imask<SimdStyle, Idof>(result, tsl::to_integral<SimdStyle>(mask));

        // Increment the valid count
        auto count_increment =
          tsl::reinterpret<SimdStyle, UnsignedSimdT, Idof>(tsl::maskz_mov<SimdStyle>(mask, m_increment));
        if constexpr (sizeof(typename SimdStyle::base_type) < sizeof(size_t)) {
          for (auto const &inc : tsl::convert_up<SimdStyle, CountSimdT, Idof>(count_increment)) {
            valid_count = tsl::add<CountSimdT, Idof>(valid_count, inc);
          }
        } else {
          valid_count = tsl::add<CountSimdT, Idof>(valid_count, count_increment);
        }
      }
      // Get the simdified valid elements count
      auto scalar_valid_count = tsl::hadd<CountSimdT, Idof>(valid_count);
      if (p_data != end) {
        size_t valid_count_scalar = 0;
        auto remainder_result = tsl::integral_all_false<SimdStyle>();
        // Iterate over the remainder of the data
        auto shift = 0;

        for (; p_data != end; ++p_data, ++shift) {
          auto const res =
            static_cast<typename SimdStyle::imask_type>(CompareFun<ScalarT, Idof>::apply(*p_data, m_predicate_scalar));
          remainder_result = tsl::insert_mask<SimdStyle, Idof>(remainder_result, res, shift);
          valid_count_scalar += res;
        }
        // Store the remainder result
        tsl::store_imask<SimdStyle, Idof>(result, remainder_result);
        ++result;
        // Increment the valid count
        scalar_valid_count += valid_count_scalar;
      }
      return std::make_tuple(result, scalar_valid_count);
    }

    template <class HS = HintSet>
    auto operator()(
      SimdOpsIterable auto p_result, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
      enable_if_has_hints_t<HS, hints::operators::filter::count_bits, hints::intermediate::dense_bit_mask> =
        {}) noexcept -> std::tuple<DataSinkType, size_t> {
      constexpr auto const bits_per_mask = sizeof(typename SimdStyle::imask_type) * CHAR_BIT;
      // Get the end of the SIMD iteration
      auto const batched_end = batched_iter_end<bits_per_mask>(p_data, p_end);

      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      // Get the result pointer
      auto result = reinterpret_iterable<DataSinkType>(p_result);

      auto const m_increment = tsl::set1<UnsignedSimdT>(1);

      auto valid_count = tsl::set1<CountSimdT, Idof>(0);
      typename SimdStyle::register_type data;

      // Iterate over the data simdified and apply the Filter for equality
      for (; p_data != batched_end; ++result) {
        auto mask = tsl::integral_all_false<SimdStyle, Idof>();
        for (size_t i = 0; i < bits_per_mask;
             i += SimdStyle::vector_element_count(), p_data += SimdStyle::vector_element_count()) {
          // Load data from the source
          if constexpr (has_hint<HintSet, hints::memory::aligned>) {
            data = tsl::load<SimdStyle, Idof>(p_data);
          } else {
            data = tsl::loadu<SimdStyle, Idof>(p_data);
          }
          auto part_mask = CompareFun<SimdStyle, Idof>::apply(data, m_predicate_reg);
          // Increment the valid count
          auto count_increment =
            tsl::reinterpret<SimdStyle, UnsignedSimdT, Idof>(tsl::maskz_mov<SimdStyle>(mask, m_increment));
          if constexpr (sizeof(typename SimdStyle::base_type) < sizeof(size_t)) {
            for (auto const &inc : tsl::convert_up<SimdStyle, CountSimdT, Idof>(count_increment)) {
              valid_count = tsl::add<CountSimdT, Idof>(valid_count, inc);
            }
          } else {
            valid_count = tsl::add<CountSimdT, Idof>(valid_count, count_increment);
          }
          mask = tsl::insert_mask<SimdStyle, Idof>(mask, tsl::to_integral<SimdStyle>(part_mask), i);
          // Store the result as an integral value into the data sink
          tsl::store_imask<SimdStyle, Idof>(result, mask);
        }
      }
      // Get the simdified valid elements count
      auto scalar_valid_count = tsl::hadd<CountSimdT, Idof>(valid_count);
      if (p_data != end) {
        size_t valid_count_scalar = 0;
        auto remainder_result = tsl::integral_all_false<SimdStyle>();
        // Iterate over the remainder of the data
        auto shift = 0;

        for (; p_data != end; ++p_data, ++shift) {
          auto const res =
            static_cast<typename SimdStyle::imask_type>(CompareFun<ScalarT, Idof>::apply(*p_data, m_predicate_scalar));
          remainder_result = tsl::insert_mask<SimdStyle, Idof>(remainder_result, res, shift);
          valid_count_scalar += res;
        }
        // Store the remainder result
        tsl::store_imask<SimdStyle, Idof>(result, remainder_result);
        ++result;
        // Increment the valid count
        scalar_valid_count += valid_count_scalar;
      }
      return std::make_tuple(result, scalar_valid_count);
    }

    template <class HS = HintSet>
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    enable_if_has_hints_mutual_excluding_t<HS, std::tuple<hints::intermediate::bit_mask>,
                                                           std::tuple<hints::operators::filter::count_bits>> =
                      {}) noexcept -> DataSinkType {
      // Get the end of the SIMD iteration
      auto const simd_end = simd_iter_end<SimdStyle>(p_data, p_end);

      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      // Get the result pointer
      auto result = reinterpret_iterable<DataSinkType>(p_result);

      typename SimdStyle::register_type data;
      // Iterate over the data simdified and apply the Filter for equality
      for (; p_data != simd_end; p_data += SimdStyle::vector_element_count(), ++result) {
        // Load data from the source
        if constexpr (has_hint<HintSet, hints::memory::aligned>) {
          data = tsl::load<SimdStyle, Idof>(p_data);
        } else {
          data = tsl::loadu<SimdStyle, Idof>(p_data);
        }
        // Compare the data with the predicate producing a mask type (either
        // register or integral type)
        auto mask = CompareFun<SimdStyle, Idof>::apply(data, m_predicate_reg);
        // Store the result as an integral value into the data sink
        tsl::store_imask<SimdStyle, Idof>(result, tsl::to_integral<SimdStyle>(mask));
      }
      // Get the simdified valid elements count
      if (p_data != end) {
        auto remainder_result = tsl::integral_all_false<SimdStyle>();
        // Iterate over the remainder of the data
        auto shift = 0;

        for (; p_data != end; ++p_data, ++shift) {
          auto const res =
            static_cast<typename SimdStyle::imask_type>(CompareFun<ScalarT, Idof>::apply(*p_data, m_predicate_scalar));
          remainder_result = tsl::insert_mask<SimdStyle, Idof>(remainder_result, res, shift);
        }
        // Store the remainder result
        tsl::store_imask<SimdStyle, Idof>(result, remainder_result);
        ++result;
        // Increment the valid count
      }
      return result;
    }

    template <class HS = HintSet>
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    enable_if_has_hints_mutual_excluding_t<HS, std::tuple<hints::intermediate::dense_bit_mask>,
                                                           std::tuple<hints::operators::filter::count_bits>> =
                      {}) noexcept -> DataSinkType {
      constexpr auto const bits_per_mask = sizeof(typename SimdStyle::imask_type) * CHAR_BIT;
      // Get the end of the SIMD iteration
      auto const batched_end = batched_iter_end<bits_per_mask>(p_data, p_end);

      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      // Get the result pointer
      auto result = reinterpret_iterable<DataSinkType>(p_result);

      typename SimdStyle::register_type data;
      // Iterate over the data simdified and apply the Filter for equality
      for (; p_data != batched_end; ++result) {
        auto mask = tsl::integral_all_false<SimdStyle, Idof>();
        for (size_t i = 0; i < bits_per_mask;
             i += SimdStyle::vector_element_count(), p_data += SimdStyle::vector_element_count()) {
          // Load data from the source
          if constexpr (has_hint<HintSet, hints::memory::aligned>) {
            data = tsl::load<SimdStyle, Idof>(p_data);
          } else {
            data = tsl::loadu<SimdStyle, Idof>(p_data);
          }
          auto part_mask = CompareFun<SimdStyle, Idof>::apply(data, m_predicate_reg);
          mask = tsl::insert_mask<SimdStyle, Idof>(mask, tsl::to_integral<SimdStyle>(part_mask), i);
          // Store the result as an integral value into the data sink
          tsl::store_imask<SimdStyle, Idof>(result, mask);
        }
      }
      if (p_data != end) {
        auto remainder_result = tsl::integral_all_false<SimdStyle>();
        // Iterate over the remainder of the data
        auto shift = 0;

        for (; p_data != end; ++p_data, ++shift) {
          auto const res =
            static_cast<typename SimdStyle::imask_type>(CompareFun<ScalarT, Idof>::apply(*p_data, m_predicate_scalar));
          remainder_result = tsl::insert_mask<SimdStyle, Idof>(remainder_result, res, shift);
        }
        // Store the remainder result
        tsl::store_imask<SimdStyle, Idof>(result, remainder_result);
        ++result;
      }
      return result;
    }

    template <class HS = HintSet>
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    ResultType start_position = 0,
                    enable_if_has_hint_t<HS, hints::intermediate::position_list> = {}) noexcept -> DataSinkType {
      using ResultSimdStyle = typename SimdStyle::template transform_extension<ResultType>;
      auto current_positions_reg = tsl::custom_sequence<ResultSimdStyle>(start_position);

      auto const position_increment_reg = tsl::set1<ResultSimdStyle>(ResultSimdStyle::vector_element_count());
      // Get the end of the SIMD iteration
      auto const simd_end = simd_iter_end<SimdStyle>(p_data, p_end);
      auto position_simd_end = (p_end - p_data) + start_position;
      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      // Get the result pointer
      auto result = reinterpret_iterable<DataSinkType>(p_result);

      typename SimdStyle::register_type data_reg;

      // Iterate over the data simdified and apply the Filter for equality
      for (; p_data != simd_end; p_data += SimdStyle::vector_element_count()) {
        // Load data from the source
        if constexpr (has_hint<HintSet, hints::memory::aligned>) {
          data_reg = tsl::load<SimdStyle, Idof>(p_data);
        } else {
          data_reg = tsl::loadu<SimdStyle, Idof>(p_data);
        }
        // Compare the data with the predicate producing a mask type (either
        // register or integral type)
        auto mask = tsl::to_integral<SimdStyle>(CompareFun<SimdStyle, Idof>::apply(data_reg, m_predicate_reg));

        if constexpr (sizeof(typename SimdStyle::base_type) == sizeof(ResultType)) {
          tsl::compress_store<ResultSimdStyle>(mask, result, current_positions_reg);
          result += tsl::mask_population_count<SimdStyle>(mask);
          current_positions_reg = tsl::add<ResultSimdStyle, Idof>(current_positions_reg, position_increment_reg);
        } else {
          for (size_t i = 0; i < SimdStyle::vector_element_count(); i += ResultSimdStyle::vector_element_count()) {
            auto current_mask = tsl::extract_mask<ResultSimdStyle, Idof>(mask, i);
            tsl::compress_store<ResultSimdStyle>(current_mask, result, current_positions_reg);
            result += tsl::mask_population_count<ResultSimdStyle>(current_mask);
            current_positions_reg = tsl::add<ResultSimdStyle, Idof>(current_positions_reg, position_increment_reg);
          }
        }
      }
      // Get the simdified valid elements count
      if (p_data != end) {
        for (; p_data != end; ++p_data, ++position_simd_end) {
          if (CompareFun<ScalarT, Idof>::apply(*p_data, m_predicate_scalar)) {
            *result = position_simd_end;
            ++result;
          }
        }
      }
      return result;
    }

    auto merge(Generic_Filter const &other) noexcept -> void {}

    auto finalize() const noexcept -> void {}
  };

  template <tsl::VectorProcessingStyle _SimdStyle, template <class, class> class CompareFun,
            class HintSet = OperatorHintSet<hints::intermediate::bit_mask>, typename Idof = tsl::workaround>
  class Generic_Range_Filter {
   public:
    using SimdStyle = _SimdStyle;
    using ResultType =
      std::conditional_t<has_hint<HintSet, hints::intermediate::position_list>, size_t, typename SimdStyle::imask_type>;
    using DataSinkType = ResultType *;
    using base_type = typename SimdStyle::base_type;
    using result_base_type = typename SimdStyle::imask_type;

    using CTorParamTupleT = std::tuple<typename SimdStyle::base_type const, typename SimdStyle::base_type const>;
    static constexpr bool ProducesBitmask = has_hint<HintSet, hints::intermediate::bit_mask>;
    constexpr size_t byte_count(SimdOpsIterable auto p_start, SimdOpsIterableOrSizeT auto p_end) {
      auto it_end = iter_end(p_start, p_end);
      auto dist = it_end - p_start;
      return dist * sizeof(ResultType);
    }

   private:
    using ScalarT = tsl::simd<typename SimdStyle::base_type, tsl::scalar>;
    using UnsignedSimdT = typename SimdStyle::template transform_extension<typename SimdStyle::offset_base_type>;
    using CountSimdT = typename SimdStyle::template transform_extension<size_t>;
    using CountSimdRegisterT = typename SimdStyle::template transform_type<size_t>;

   private:
    typename SimdStyle::base_type const m_lower_predicate_scalar;
    typename SimdStyle::base_type const m_upper_predicate_scalar;
    typename SimdStyle::register_type const m_lower_predicate_reg;
    typename SimdStyle::register_type const m_upper_predicate_reg;

   public:
    Generic_Range_Filter(typename SimdStyle::base_type const p_lower_predicate,
                         typename SimdStyle::base_type const p_upper_predicate)
      : m_lower_predicate_scalar(p_lower_predicate),
        m_upper_predicate_scalar(p_upper_predicate),
        m_lower_predicate_reg(tsl::set1<SimdStyle>(p_lower_predicate)),
        m_upper_predicate_reg(tsl::set1<SimdStyle>(p_upper_predicate)) {}

    ~Generic_Range_Filter() = default;

   public:
    template <class HS = HintSet>
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    enable_if_has_hint_t<HS, hints::operators::filter::count_bits, hints::intermediate::bit_mask> =
                      {}) noexcept -> std::tuple<DataSinkType, size_t> {
      // Get the end of the SIMD iteration
      auto const simd_end = simd_iter_end<SimdStyle>(p_data, p_end);

      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      // Get the result pointer
      auto result = reinterpret_iterable<DataSinkType>(p_result);

      auto const m_increment = tsl::set1<UnsignedSimdT>(1);

      auto valid_count = tsl::set1<CountSimdT, Idof>(0);

      typename SimdStyle::register_type data;

      // Iterate over the data simdified and apply the Filter for equality
      for (; p_data != simd_end; p_data += SimdStyle::vector_element_count(), ++result) {
        // Load data from the source
        if constexpr (has_hint<HintSet, hints::memory::aligned>) {
          data = tsl::load<SimdStyle, Idof>(p_data);
        } else {
          data = tsl::loadu<SimdStyle, Idof>(p_data);
        }
        // Compare the data with the predicate producing a mask type (either
        // register or integral type)
        auto mask = CompareFun<SimdStyle, Idof>::apply(data, m_lower_predicate_reg, m_upper_predicate_reg);
        // Store the result as an integral value into the data sink
        tsl::store_imask<SimdStyle, Idof>(result, tsl::to_integral<SimdStyle>(mask));

        // Increment the valid count
        auto count_increment =
          tsl::reinterpret<SimdStyle, UnsignedSimdT, Idof>(tsl::maskz_mov<SimdStyle>(mask, m_increment));
        if constexpr (sizeof(typename SimdStyle::base_type) < sizeof(size_t)) {
          for (auto const &inc : tsl::convert_up<SimdStyle, CountSimdT, Idof>(count_increment)) {
            valid_count = tsl::add<CountSimdT, Idof>(valid_count, inc);
          }
        } else {
          valid_count = tsl::add<CountSimdT, Idof>(valid_count, count_increment);
        }
      }
      // Get the simdified valid elements count
      auto scalar_valid_count = tsl::hadd<CountSimdT, Idof>(valid_count);
      if (p_data != end) {
        size_t valid_count_scalar = 0;
        auto remainder_result = tsl::integral_all_false<SimdStyle>();
        // Iterate over the remainder of the data
        auto shift = 0;

        for (; p_data != end; ++p_data, ++shift) {
          auto const res = static_cast<typename SimdStyle::imask_type>(
            CompareFun<ScalarT, Idof>::apply(*p_data, m_lower_predicate_scalar, m_upper_predicate_scalar));
          remainder_result = tsl::insert_mask<SimdStyle, Idof>(remainder_result, res, shift);
          valid_count_scalar += res;
        }
        // Store the remainder result
        tsl::store_imask<SimdStyle, Idof>(result, remainder_result);
        ++result;
        // Increment the valid count
        scalar_valid_count += valid_count_scalar;
      }
      return std::make_tuple(result, scalar_valid_count);
    }

    template <class HS = HintSet>
    auto operator()(
      SimdOpsIterable auto p_result, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
      enable_if_has_hints_t<HS, hints::operators::filter::count_bits, hints::intermediate::dense_bit_mask> =
        {}) noexcept -> std::tuple<DataSinkType, size_t> {
      constexpr auto const bits_per_mask = sizeof(typename SimdStyle::imask_type) * CHAR_BIT;
      // Get the end of the SIMD iteration
      auto const batched_end = batched_iter_end<bits_per_mask>(p_data, p_end);

      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      // Get the result pointer
      auto result = reinterpret_iterable<DataSinkType>(p_result);

      auto const m_increment = tsl::set1<UnsignedSimdT>(1);

      auto valid_count = tsl::set1<CountSimdT, Idof>(0);
      typename SimdStyle::register_type data;

      // Iterate over the data simdified and apply the Filter for equality
      for (; p_data != batched_end; ++result) {
        auto mask = tsl::integral_all_false<SimdStyle, Idof>();
        for (size_t i = 0; i < bits_per_mask;
             i += SimdStyle::vector_element_count(), p_data += SimdStyle::vector_element_count()) {
          // Load data from the source
          if constexpr (has_hint<HintSet, hints::memory::aligned>) {
            data = tsl::load<SimdStyle, Idof>(p_data);
          } else {
            data = tsl::loadu<SimdStyle, Idof>(p_data);
          }
          auto part_mask = CompareFun<SimdStyle, Idof>::apply(data, m_lower_predicate_reg, m_upper_predicate_reg);
          // Increment the valid count
          auto count_increment =
            tsl::reinterpret<SimdStyle, UnsignedSimdT, Idof>(tsl::maskz_mov<SimdStyle>(mask, m_increment));
          if constexpr (sizeof(typename SimdStyle::base_type) < sizeof(size_t)) {
            for (auto const &inc : tsl::convert_up<SimdStyle, CountSimdT, Idof>(count_increment)) {
              valid_count = tsl::add<CountSimdT, Idof>(valid_count, inc);
            }
          } else {
            valid_count = tsl::add<CountSimdT, Idof>(valid_count, count_increment);
          }
          mask = tsl::insert_mask<SimdStyle, Idof>(mask, tsl::to_integral<SimdStyle>(part_mask), i);
          // Store the result as an integral value into the data sink
          tsl::store_imask<SimdStyle, Idof>(result, mask);
        }
      }
      // Get the simdified valid elements count
      auto scalar_valid_count = tsl::hadd<CountSimdT, Idof>(valid_count);
      if (p_data != end) {
        size_t valid_count_scalar = 0;
        auto remainder_result = tsl::integral_all_false<SimdStyle>();
        // Iterate over the remainder of the data
        auto shift = 0;

        for (; p_data != end; ++p_data, ++shift) {
          auto const res = static_cast<typename SimdStyle::imask_type>(
            CompareFun<ScalarT, Idof>::apply(*p_data, m_lower_predicate_scalar, m_upper_predicate_scalar));
          remainder_result = tsl::insert_mask<SimdStyle, Idof>(remainder_result, res, shift);
          valid_count_scalar += res;
        }
        // Store the remainder result
        tsl::store_imask<SimdStyle, Idof>(result, remainder_result);
        ++result;
        // Increment the valid count
        scalar_valid_count += valid_count_scalar;
      }
      return std::make_tuple(result, scalar_valid_count);
    }

    template <class HS = HintSet>
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    enable_if_has_hints_mutual_excluding_t<HS, std::tuple<hints::intermediate::bit_mask>,
                                                           std::tuple<hints::operators::filter::count_bits>> =
                      {}) noexcept -> DataSinkType {
      // Get the end of the SIMD iteration
      auto const simd_end = simd_iter_end<SimdStyle>(p_data, p_end);

      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      // Get the result pointer
      auto result = reinterpret_iterable<DataSinkType>(p_result);

      typename SimdStyle::register_type data;

      // Iterate over the data simdified and apply the Filter for equality
      for (; p_data != simd_end; p_data += SimdStyle::vector_element_count(), ++result) {
        // Load data from the source
        if constexpr (has_hint<HintSet, hints::memory::aligned>) {
          data = tsl::load<SimdStyle, Idof>(p_data);
        } else {
          data = tsl::loadu<SimdStyle, Idof>(p_data);
        }
        // Compare the data with the predicate producing a mask type (either
        // register or integral type)
        auto mask = CompareFun<SimdStyle, Idof>::apply(data, m_lower_predicate_reg, m_upper_predicate_reg);
        // Store the result as an integral value into the data sink
        tsl::store_imask<SimdStyle, Idof>(result, tsl::to_integral<SimdStyle>(mask));
      }
      // Get the simdified valid elements count
      if (p_data != end) {
        auto remainder_result = tsl::integral_all_false<SimdStyle>();
        // Iterate over the remainder of the data
        auto shift = 0;

        for (; p_data != end; ++p_data, ++shift) {
          auto const res = static_cast<typename SimdStyle::imask_type>(
            CompareFun<ScalarT, Idof>::apply(*p_data, m_lower_predicate_scalar, m_upper_predicate_scalar));
          remainder_result = tsl::insert_mask<SimdStyle, Idof>(remainder_result, res, shift);
        }
        // Store the remainder result
        tsl::store_imask<SimdStyle, Idof>(result, remainder_result);
        ++result;
        // Increment the valid count
      }
      return result;
    }

    template <class HS = HintSet>
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    enable_if_has_hints_mutual_excluding_t<HS, std::tuple<hints::intermediate::dense_bit_mask>,
                                                           std::tuple<hints::operators::filter::count_bits>> =
                      {}) noexcept -> DataSinkType {
      constexpr auto const bits_per_mask = sizeof(typename SimdStyle::imask_type) * CHAR_BIT;
      // Get the end of the SIMD iteration
      auto const batched_end = batched_iter_end<bits_per_mask>(p_data, p_end);

      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      // Get the result pointer
      auto result = reinterpret_iterable<DataSinkType>(p_result);

      typename SimdStyle::register_type data;

      // Iterate over the data simdified and apply the Filter for equality
      for (; p_data != batched_end; ++result) {
        auto mask = tsl::integral_all_false<SimdStyle, Idof>();
        for (size_t i = 0; i < bits_per_mask;
             i += SimdStyle::vector_element_count(), p_data += SimdStyle::vector_element_count()) {
          // Load data from the source
          if constexpr (has_hint<HintSet, hints::memory::aligned>) {
            data = tsl::load<SimdStyle, Idof>(p_data);
          } else {
            data = tsl::loadu<SimdStyle, Idof>(p_data);
          }
          auto part_mask = CompareFun<SimdStyle, Idof>::apply(data, m_lower_predicate_reg, m_upper_predicate_reg);
          mask = tsl::insert_mask<SimdStyle, Idof>(mask, tsl::to_integral<SimdStyle>(part_mask), i);
          // Store the result as an integral value into the data sink
          tsl::store_imask<SimdStyle, Idof>(result, mask);
        }
      }
      if (p_data != end) {
        auto remainder_result = tsl::integral_all_false<SimdStyle>();
        // Iterate over the remainder of the data
        auto shift = 0;

        for (; p_data != end; ++p_data, ++shift) {
          auto const res = static_cast<typename SimdStyle::imask_type>(
            CompareFun<ScalarT, Idof>::apply(*p_data, m_lower_predicate_scalar, m_upper_predicate_scalar));
          remainder_result = tsl::insert_mask<SimdStyle, Idof>(remainder_result, res, shift);
        }
        // Store the remainder result
        tsl::store_imask<SimdStyle, Idof>(result, remainder_result);
        ++result;
      }
      return result;
    }

    template <class HS = HintSet>
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    ResultType start_position = 0,
                    enable_if_has_hint_t<HS, hints::intermediate::position_list> = {}) noexcept -> DataSinkType {
      using ResultSimdStyle = typename SimdStyle::template transform_extension<ResultType>;
      auto current_positions_reg = tsl::custom_sequence<ResultSimdStyle>(start_position);

      auto const position_increment_reg = tsl::set1<ResultSimdStyle>(ResultSimdStyle::vector_element_count());
      // Get the end of the SIMD iteration
      auto const simd_end = simd_iter_end<SimdStyle>(p_data, p_end);
      auto position_simd_end = iter_distance(p_data, simd_end) + start_position;
      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      // Get the result pointer
      auto result = reinterpret_iterable<DataSinkType>(p_result);

      typename SimdStyle::register_type data_reg;

      // Iterate over the data simdified and apply the Filter for equality
      for (; p_data != simd_end; p_data += SimdStyle::vector_element_count()) {
        // Load data from the source
        if constexpr (has_hint<HintSet, hints::memory::aligned>) {
          data_reg = tsl::load<SimdStyle, Idof>(p_data);
        } else {
          data_reg = tsl::loadu<SimdStyle, Idof>(p_data);
        }
        // Compare the data with the predicate producing a mask type (either
        // register or integral type)
        auto mask = tsl::to_integral<SimdStyle>(
          CompareFun<SimdStyle, Idof>::apply(data_reg, m_lower_predicate_reg, m_upper_predicate_reg));

        if constexpr (sizeof(typename SimdStyle::base_type) == sizeof(ResultType)) {
          tsl::compress_store<ResultSimdStyle>(mask, result, current_positions_reg);
          result += tsl::mask_population_count<SimdStyle>(mask);
          current_positions_reg = tsl::add<ResultSimdStyle, Idof>(current_positions_reg, position_increment_reg);
        } else {
          for (size_t i = 0; i < SimdStyle::vector_element_count(); i += ResultSimdStyle::vector_element_count()) {
            auto current_mask = tsl::extract_mask<ResultSimdStyle, Idof>(mask, i);
            tsl::compress_store<ResultSimdStyle>(current_mask, result, current_positions_reg);
            result += tsl::mask_population_count<ResultSimdStyle>(current_mask);
            current_positions_reg = tsl::add<ResultSimdStyle, Idof>(current_positions_reg, position_increment_reg);
          }
        }
      }
      // Get the simdified valid elements count
      if (p_data != end) {
        for (; p_data != end; ++p_data, ++position_simd_end) {
          if (CompareFun<ScalarT, Idof>::apply(*p_data, m_lower_predicate_scalar, m_upper_predicate_scalar)) {
            *result = position_simd_end;
            ++result;
          }
        }
      }
      return result;
    }

    auto merge(Generic_Range_Filter const &other) noexcept -> void {}

    auto finalize() const noexcept -> void {}
  };

  template <tsl::VectorProcessingStyle _SimdStyle, class HintSet = OperatorHintSet<hints::intermediate::bit_mask>,
            typename Idof = tsl::workaround>
  using Filter_EQ = Generic_Filter<_SimdStyle, tsl::functors::equal, HintSet, Idof>;
  template <tsl::VectorProcessingStyle _SimdStyle, class HintSet = OperatorHintSet<hints::intermediate::bit_mask>,
            typename Idof = tsl::workaround>
  using Filter_NEQ = Generic_Filter<_SimdStyle, tsl::functors::nequal, HintSet, Idof>;
  template <tsl::VectorProcessingStyle _SimdStyle, class HintSet = OperatorHintSet<hints::intermediate::bit_mask>,
            typename Idof = tsl::workaround>
  using Filter_LT = Generic_Filter<_SimdStyle, tsl::functors::less_than, HintSet, Idof>;
  template <tsl::VectorProcessingStyle _SimdStyle, class HintSet = OperatorHintSet<hints::intermediate::bit_mask>,
            typename Idof = tsl::workaround>
  using Filter_GT = Generic_Filter<_SimdStyle, tsl::functors::greater_than, HintSet, Idof>;
  template <tsl::VectorProcessingStyle _SimdStyle, class HintSet = OperatorHintSet<hints::intermediate::bit_mask>,
            typename Idof = tsl::workaround>
  using Filter_LE = Generic_Filter<_SimdStyle, tsl::functors::less_than_or_equal, HintSet, Idof>;
  template <tsl::VectorProcessingStyle _SimdStyle, class HintSet = OperatorHintSet<hints::intermediate::bit_mask>,
            typename Idof = tsl::workaround>
  using Filter_GE = Generic_Filter<_SimdStyle, tsl::functors::greater_than_or_equal, HintSet, Idof>;

  template <tsl::VectorProcessingStyle _SimdStyle, class HintSet = OperatorHintSet<hints::intermediate::bit_mask>,
            typename Idof = tsl::workaround>
  using Filter_BWI = Generic_Range_Filter<_SimdStyle, tsl::functors::between_inclusive, HintSet, Idof>;

}  // namespace tuddbs
