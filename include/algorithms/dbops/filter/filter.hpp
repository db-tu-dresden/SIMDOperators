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

    using base_type = typename SimdStyle::base_type;
    using DataSinkType = base_type *;

   public:
    explicit Filter() = default;
    ~Filter() = default;

   public:
    template <class HS = HintSet, enable_if_has_hint_t<HS, hints::intermediate::bit_mask>>
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end,
                    SimdOpsIterable auto p_valid_masks) -> void {
      // Get the end of the SIMD iteration
      auto const simd_end = simd_iter_end<SimdStyle>(p_data, p_end);

      // Get the end of the data
      auto const end = iter_end(p_data, p_end);
      // Get the result pointer
      auto valid_masks = reinterpret_iterable<ValidElementIterableType>(p_valid_masks);
      auto result = reinterpret_iterable<DataSinkType>(p_result);

      for (; p_data != simd_end;
           p_data += SimdStyle::vector_element_count(), p_result += SimdStyle::vector_element_count(), ++valid_masks) {
        auto const data = tsl::load<SimdStyle, Idof>(p_data);
        tsl::load_imask<SimdStyle>(valid_masks)
      }
    }
  };

}  // namespace tuddbs

#endif