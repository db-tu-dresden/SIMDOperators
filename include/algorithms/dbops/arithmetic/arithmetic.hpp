
// ------------------------------------------------------------------- //
/*
   This file is part of the SimdOperators Project.
   Author(s): Alexander Krause.

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
 * @file multiply.hpp
 * @brief
 */

#ifndef SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_ARITHMETIC_MULTIPLY_HPP
#define SIMDOPS_INCLUDE_ALGORITHMS_DBOPS_ARITHMETIC_MULTIPLY_HPP

#include <climits>
#include <cstddef>
#include <deque>
#include <iterable.hpp>
#include <tuple>
#include <type_traits>

#include "algorithms/dbops/dbops_hints.hpp"
#include "tsl.hpp"

namespace tuddbs {
  template <tsl::VectorProcessingStyle _SimdStyle, class HintSet = OperatorHintSet<hints::arithmetic::add>>
  class Arithmetic {
   public:
    using SimdStyle = _SimdStyle;
    using reg_t = typename SimdStyle::register_type;

    explicit Arithmetic() {}

    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data1, SimdOpsIterableOrSizeT auto p_end1,
                    SimdOpsIterable auto p_data2) {
      const size_t scalar_remainder = (p_end1 - p_data1) % SimdStyle::vector_element_count();
      for (; p_data1 + SimdStyle::vector_element_count() <= p_end1; p_data1 += SimdStyle::vector_element_count(),
                                                                    p_data2 += SimdStyle::vector_element_count(),
                                                                    p_result += SimdStyle::vector_element_count()) {
        const auto vals1 = tsl::loadu<SimdStyle>(p_data1);
        const auto vals2 = tsl::loadu<SimdStyle>(p_data2);
        tsl::storeu<SimdStyle>(p_result, calc(vals1, vals2));
      }
      for (size_t i = 0; i < scalar_remainder; ++i) {
        *p_result++ = calc<tsl::simd<typename SimdStyle::base_type, tsl::scalar>>(*p_data1++, *p_data2++);
      }
    }

   private:
    template <class PS = SimdStyle>
    typename PS::register_type calc(typename PS::register_type vals1, typename PS::register_type vals2) const {
      if constexpr (has_hint<HintSet, tuddbs::hints::arithmetic::add>) {
        return tsl::add<PS>(vals1, vals2);
      } else if constexpr (has_hint<HintSet, tuddbs::hints::arithmetic::sub>) {
        return tsl::sub<PS>(vals1, vals2);
      } else if constexpr (has_hint<HintSet, tuddbs::hints::arithmetic::mul>) {
        return tsl::mul<PS>(vals1, vals2);
      } else if constexpr (has_hint<HintSet, tuddbs::hints::arithmetic::div>) {
        return tsl::div<PS>(vals1, vals2);
      } else {
        throw std::runtime_error("No supported arithmetic operation found");
      }
    }
  };

  template <typename SimdStyle>
  using col_adder_t = tuddbs::Arithmetic<SimdStyle, tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::add>>;

  template <typename SimdStyle>
  using col_subtractor_t = tuddbs::Arithmetic<SimdStyle, tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::sub>>;

  template <typename SimdStyle>
  using col_multiplier_t = tuddbs::Arithmetic<SimdStyle, tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::mul>>;

  template <typename SimdStyle>
  using col_divider_t = tuddbs::Arithmetic<SimdStyle, tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::div>>;
}  // namespace tuddbs

#endif
