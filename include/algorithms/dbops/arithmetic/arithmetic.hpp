
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
    using base_t = typename SimdStyle::base_type;

    explicit Arithmetic() {}

    /* Reducing Operations on a single column, e.g., sum, avg */
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data, SimdOpsIterableOrSizeT auto p_end) {
      const auto simd_end = tuddbs::simd_iter_end<SimdStyle>(p_data, p_end);
      const auto scalar_end = tuddbs::iter_end(p_data, p_end);

      reg_t res_vec = tsl::set1<SimdStyle>(0);
      if constexpr (std::is_floating_point_v<base_t>) {
        // This is a SIMDified version of the Kahan summation
        reg_t error_vec = tsl::set1<SimdStyle>(0);
        for (; p_data != simd_end; p_data += SimdStyle::vector_element_count()) {
          reg_t vals = tsl::loadu<SimdStyle>(p_data);
          vals = tsl::sub<SimdStyle>(vals, error_vec);
          reg_t buffer = tsl::add<SimdStyle>(res_vec, vals);
          error_vec = tsl::sub<SimdStyle>(tsl::sub<SimdStyle>(buffer, res_vec), vals);
          res_vec = buffer;
        }
      } else {
        for (; p_data != simd_end; p_data += SimdStyle::vector_element_count()) {
          res_vec = tsl::add<SimdStyle>(res_vec, tsl::loadu<SimdStyle>(p_data));
        }
      }

      // base_t res_scalar = tsl::hadd<SimdStyle>(res_vec);
      base_t res_scalar = 0;

      if constexpr (std::is_floating_point_v<base_t>) {
        base_t error = static_cast<base_t>(0.0);
        for (; p_data != scalar_end; p_data++) {
          // This is a scalar version of the Kahan summation
          base_t y = *p_data - error;
          base_t t = res_scalar + y;
          error = (t - res_scalar) - y;
          res_scalar = t;
        }
        // Scalar remainder for Kahan
        const auto vec_res = tsl::hadd<SimdStyle>(res_vec);
        base_t y = vec_res - error;
        res_scalar += y;
      } else {
        for (; p_data != scalar_end; p_data++) {
          res_scalar += *p_data;
        }
        res_scalar += tsl::hadd<SimdStyle>(res_vec);
      }

      if constexpr (has_hint<HintSet, tuddbs::hints::arithmetic::sum>) {
        *p_result = res_scalar;
      } else if constexpr (has_hint<HintSet, tuddbs::hints::arithmetic::average>) {
        const size_t element_count = scalar_end - p_data;
        if constexpr (std::is_floating_point_v<base_t>) {
          *p_result = res_scalar / element_count;
        } else {
          *p_result = static_cast<double>(res_scalar) / element_count;
        }
      } else {
        throw std::runtime_error("Unknown single-column arithmetic. No suitable hint was provided.");
      }
    }

    /* Combining two columns element-wise, e.g., add, sub, div, mul */
    auto operator()(SimdOpsIterable auto p_result, SimdOpsIterable auto p_data1, SimdOpsIterableOrSizeT auto p_end1,
                    SimdOpsIterable auto p_data2) {
      const auto simd_end = tuddbs::simd_iter_end<SimdStyle>(p_data1, p_end1);
      const auto scalar_end = tuddbs::iter_end(p_data1, p_end1);
      for (; p_data1 != simd_end; p_data1 += SimdStyle::vector_element_count(),
                                  p_data2 += SimdStyle::vector_element_count(),
                                  p_result += SimdStyle::vector_element_count()) {
        const auto vals1 = tsl::loadu<SimdStyle>(p_data1);
        const auto vals2 = tsl::loadu<SimdStyle>(p_data2);
        tsl::storeu<SimdStyle>(p_result, calc(vals1, vals2));
      }
      // Process the scalar remainder
      for (; p_data1 != scalar_end; p_data1++, p_data2++) {
        *p_result++ = calc<tsl::simd<typename SimdStyle::base_type, tsl::scalar>>(*p_data1, *p_data2);
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

  template <typename SimdStyle>
  using col_sum_t = tuddbs::Arithmetic<SimdStyle, tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::sum>>;

  template <typename SimdStyle>
  using col_average_t = tuddbs::Arithmetic<SimdStyle, tuddbs::OperatorHintSet<tuddbs::hints::arithmetic::average>>;
}  // namespace tuddbs

#endif
