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
 * @file union.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_UNION_UNION_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_UNION_UNION_HPP

#include <climits>
#include <cstddef>
#include <tuple>
#include <type_traits>

#include "algorithms/dbops/dbops_hints.hpp"
#include "algorithms/utils/hinting.hpp"
#include "iterable.hpp"
#include "tsl.hpp"

namespace tuddbs {
  template <tsl::VectorProcessingStyle _SimdStyle, class HintSet = OperatorHintSet<hints::intermediate::bit_mask>,
            typename Idof = tsl::workaround>
  class Union {
   public:
    using SimdStyle = _SimdStyle;
    // using ProcessingBaseType = size_t;
    // using ProcessingSimdStyle = typename SimdStyle::template transform_extension<ProcessingBaseType>;
    // using ProcessingIterableType = ProcessingBaseType *;

    using ResultType =
      std::conditional_t<has_hint<HintSet, hints::intermediate::position_list>, size_t, typename SimdStyle::imask_type>;
    using DataSinkType = ResultType *;

   public:
    explicit Union() = default;
    ~Union() = default;

   public:
    constexpr size_t byte_count(SimdOpsIterable auto p_start, SimdOpsIterableOrSizeT auto p_end) {
      auto it_end = iter_end(p_start, p_end);
      auto dist = it_end - p_start;
      return dist * sizeof(ResultType);
    }

    template <class HS = HintSet>
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_left_data,
                    SimdOpsIterableOrSizeT auto p_left_end, SimdOpsIterable auto p_right_data,
                    activate_for_bit_mask<HS> = {}) const noexcept -> DataSinkType {
      // Get the end of the SIMD iteration
      auto const simd_end = simd_iter_end<SimdStyle>(p_left_data, p_left_end);

      auto const end = iter_end(p_left_data, p_left_end);

      typename SimdStyle::register_type left_reg;
      typename SimdStyle::register_type right_reg;
      for (; p_left_data != simd_end; p_left_data += SimdStyle::vector_element_count(),
                                      p_right_data += SimdStyle::vector_element_count(),
                                      p_result += SimdStyle::vector_element_count()) {
        if constexpr (has_hint<HintSet, hints::memory::aligned>) {
          left_reg = tsl::load<SimdStyle, Idof>(p_left_data);
          right_reg = tsl::load<SimdStyle, Idof>(p_right_data);
        } else {
          left_reg = tsl::loadu<SimdStyle, Idof>(p_left_data);
          right_reg = tsl::loadu<SimdStyle, Idof>(p_right_data);
        }
        auto const result = tsl::binary_or<SimdStyle, Idof>(left_reg, right_reg);
        if constexpr (has_hint<HintSet, hints::memory::aligned>) {
          tsl::store<SimdStyle, Idof>(p_result, result);
        } else {
          tsl::storeu<SimdStyle, Idof>(p_result, result);
        }
      }
      if (p_left_data != end) {
        for (; p_left_data != end; ++p_left_data, ++p_right_data, ++p_result) {
          *p_result = *p_left_data | *p_right_data;
        }
      }
      return reinterpret_iterable<DataSinkType>(p_result);
    }
    template <class HS = HintSet>
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_left_data,
                    SimdOpsIterableOrSizeT auto p_left_end, SimdOpsIterable auto p_right_data,
                    activate_for_dense_bit_mask<HS> = {}) const noexcept -> DataSinkType {
      // Get the end of the SIMD iteration
      auto const simd_end = simd_iter_end<SimdStyle>(p_left_data, p_left_end);

      auto const end = iter_end(p_left_data, p_left_end);

      typename SimdStyle::register_type left_reg;
      typename SimdStyle::register_type right_reg;
      for (; p_left_data != simd_end; p_left_data += SimdStyle::vector_element_count(),
                                      p_right_data += SimdStyle::vector_element_count(),
                                      p_result += SimdStyle::vector_element_count()) {
        if constexpr (has_hint<HintSet, hints::memory::aligned>) {
          left_reg = tsl::load<SimdStyle, Idof>(p_left_data);
          right_reg = tsl::load<SimdStyle, Idof>(p_right_data);
        } else {
          left_reg = tsl::loadu<SimdStyle, Idof>(p_left_data);
          right_reg = tsl::loadu<SimdStyle, Idof>(p_right_data);
        }
        auto const result = tsl::binary_or<SimdStyle, Idof>(left_reg, right_reg);
        if constexpr (has_hint<HintSet, hints::memory::aligned>) {
          tsl::store<SimdStyle, Idof>(p_result, result);
        } else {
          tsl::storeu<SimdStyle, Idof>(p_result, result);
        }
      }
      if (p_left_data != end) {
        for (; p_left_data != end; ++p_left_data, ++p_right_data, ++p_result) {
          *p_result = *p_left_data | *p_right_data;
        }
      }
      return reinterpret_iterable<DataSinkType>(p_result);
    }

    template <class HS = HintSet>
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_left_data,
                    SimdOpsIterableOrSizeT auto p_left_end, SimdOpsIterable auto p_right_data,
                    activate_for_position_list<HS> = {}) -> DataSinkType {
      throw std::runtime_error("Not implemented yet");
    }
  };
}  // namespace tuddbs

#endif  // SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_UNION_UNION_HPP