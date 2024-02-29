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

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_FILTER_FILTER_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_FILTER_FILTER_HPP

#include <climits>
#include <cstddef>
#include <tuple>
#include <type_traits>

#include "algorithms/dbops/dbops_hints.hpp"
#include "algorithms/utils/hinting.hpp"
#include "iterable.hpp"
#include "tslintrin.hpp"

namespace tuddbs {
  template <tsl::VectorProcessingStyle _SimdStyle, class HintSet = OperatorHintSet<hints::intermediate::bit_mask>,
            typename Idof = tsl::workaround>
  class Filter {
   public:
    using SimdStyle = _SimdStyle;

    using ValidElementType =
      std::conditional_t<has_hint<HintSet, hints::intermediate::position_list>, size_t, typename SimdStyle::imask_type>;
    using ValidElementIterableType = ValidElementType *;

    using PositionalSimdStyle = typename SimdStyle::template transform_extension<size_t>;

    using base_type = typename SimdStyle::base_type;
    using DataSinkType = base_type *;

   public:
    explicit Filter() = default;
    ~Filter() = default;

   public:
    template <class HS = HintSet>
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    SimdOpsIterable auto p_valid_masks, [[maybe_unused]] SimdOpsIterable auto p_valid_masks_end, activate_for_bit_mask<HS> = {}) const noexcept -> DataSinkType {
      // Get the end of the SIMD iteration
      auto const simd_end = simd_iter_end<SimdStyle>(p_data, p_end);

      // Get the end of the data
      auto const end = iter_end(p_data, p_end);
      // Get the result pointer
      auto valid_masks = reinterpret_iterable<ValidElementIterableType>(p_valid_masks);
      auto result = reinterpret_iterable<DataSinkType>(p_result);

      typename SimdStyle::register_type data;
      for (; p_data != simd_end; p_data += SimdStyle::vector_element_count(), ++valid_masks) {
        if constexpr (has_hint<HintSet, hints::memory::aligned>) {
          data = tsl::load<SimdStyle, Idof>(p_data);
        } else {
          data = tsl::loadu<SimdStyle, Idof>(p_data);
        }
        auto const current_mask = tsl::load_imask<SimdStyle, Idof>(valid_masks);
        tsl::compress_store<SimdStyle, Idof>(current_mask, result, data);
        result += tsl::mask_population_count<SimdStyle, Idof>(current_mask);
      }

      if (p_data != end) {
        auto current_mask = tsl::load_imask<SimdStyle, Idof>(valid_masks);
        int position = 0;
        for (; p_data != end; ++p_data, ++position) {
          if (tsl::test_mask<SimdStyle, Idof>(current_mask, position)) {
            *result = *p_data;
            ++result;
          }
        }
      }
      return result;
    }

    template <class HS = HintSet>
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    SimdOpsIterable auto p_valid_masks, [[maybe_unused]] SimdOpsIterable auto p_valid_masks_end, activate_for_dense_bit_mask<HS> = {}) const noexcept
      -> DataSinkType {
      constexpr auto const bits_per_mask = sizeof(typename SimdStyle::imask_type) * CHAR_BIT;
      // Get the end of the SIMD iteration
      auto const batched_end = batched_iter_end<bits_per_mask>(p_data, p_end);

      // Get the end of the data
      auto const end = iter_end(p_data, p_end);

      auto valid_masks = reinterpret_iterable<ValidElementIterableType>(p_valid_masks);
      // Get the result pointer
      auto result = reinterpret_iterable<DataSinkType>(p_result);

      typename SimdStyle::register_type data;
      for (; p_data != batched_end; ++valid_masks) {
        auto const mask = tsl::load_imask<SimdStyle, Idof>(valid_masks);
        for (size_t i = 0; i < bits_per_mask;
             i += SimdStyle::vector_element_count(), p_data += SimdStyle::vector_element_count()) {
          if constexpr (has_hint<HintSet, hints::memory::aligned>) {
            data = tsl::load<SimdStyle, Idof>(p_data);
          } else {
            data = tsl::loadu<SimdStyle, Idof>(p_data);
          }
          auto const current_mask = tsl::extract_mask<SimdStyle, Idof>(mask, i);
          tsl::compress_store<SimdStyle, Idof>(current_mask, result, data);
          result += tsl::mask_population_count<SimdStyle, Idof>(current_mask);
        }
      }
      if (p_data != end) {
        auto current_mask = tsl::load_imask<SimdStyle, Idof>(valid_masks);
        int position = 0;
        for (; p_data != end; ++p_data, ++position) {
          if (tsl::test_mask(current_mask, position)) {
            *result = *p_data;
            ++result;
          }
        }
      }
      return result;
    }

    template <class HS = HintSet>
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data,
                    SimdOpsIterable auto p_end, SimdOpsIterable auto p_position_list, SimdOpsIterable auto p_position_list_end,
                    activate_for_position_list<HS> = {}) const noexcept -> DataSinkType {
      auto positions = reinterpret_iterable<ValidElementIterableType>(p_position_list);
      // Get the end of the data
      auto const end = iter_end(positions, p_position_list_end);
      // Get the result pointer
      auto result = reinterpret_iterable<DataSinkType>(p_result);

      if constexpr (sizeof(typename SimdStyle::base_type) == sizeof(typename PositionalSimdStyle::base_type)) {
        // Get the end of the SIMD iteration
        auto const simd_end = simd_iter_end<PositionalSimdStyle>(positions, p_position_list_end);
        typename PositionalSimdStyle::register_type current_positions;
        for (; positions != simd_end; positions += PositionalSimdStyle::vector_element_count(),
                                      result += PositionalSimdStyle::vector_element_count()) {
          if constexpr (has_hint<HintSet, hints::memory::aligned>) {
            current_positions = tsl::load<PositionalSimdStyle, Idof>(positions);
          } else {
            current_positions = tsl::loadu<PositionalSimdStyle, Idof>(positions);
          }
          auto const data = tsl::gather<SimdStyle, Idof>(p_data, current_positions);
          if constexpr (has_hint<HintSet, hints::memory::aligned>) {
            tsl::store<SimdStyle, Idof>(result, data);
          } else {
            tsl::storeu<SimdStyle, Idof>(result, data);
          }
        }

      } else {  // base_type is smaller than position_type (size_t)

        // Get the end of the SIMD iteration
        auto const batched_end = batched_iter_end<SimdStyle::vector_element_count()>(positions, p_position_list_end);

        constexpr auto const registers_per_batch =
          sizeof(typename PositionalSimdStyle::base_type) / sizeof(typename SimdStyle::base_type);

        std::array<typename SimdStyle::base_type, registers_per_batch> data_array;
        typename PositionalSimdStyle::register_type current_positions;
        for (; positions != batched_end; result += SimdStyle::vector_element_count()) {
          for (size_t i = 0; i < registers_per_batch; ++i, positions += PositionalSimdStyle::vector_element_count()) {
            if constexpr (has_hint<HintSet, hints::memory::aligned>) {
              current_positions = tsl::load<PositionalSimdStyle, Idof>(positions);
            } else {
              current_positions = tsl::loadu<PositionalSimdStyle, Idof>(positions);
            }
            data_array[i] = tsl::gather<PositionalSimdStyle, Idof>(p_data, current_positions);
          }
          auto const data = tsl::convert_down<PositionalSimdStyle, SimdStyle, Idof>(data_array);
          if constexpr (has_hint<HintSet, hints::memory::aligned>) {
            tsl::store<SimdStyle, Idof>(result, data);
          } else {
            tsl::storeu<SimdStyle, Idof>(result, data);
          }
        }
      }
      if (positions != end) {
        for (; positions != end; ++positions, ++result) {
          *result = p_data[*positions];
        }
      }
      return result;
    }

    template <tsl::VectorProcessingStyle OtherSimdStlye, class OtherHintSet, typename OtherIdof>
    auto merge(Filter<OtherSimdStlye, OtherHintSet, OtherIdof> const &other) noexcept -> void {}

    auto finalize() const noexcept -> void {}
  };

}  // namespace tuddbs

#endif