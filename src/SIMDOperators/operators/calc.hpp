// ------------------------------------------------------------------- //
/*
   This file is part of the SimdOperators Project.
   Copyright (c) 2022 SimdOperators Team.
   
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

#ifndef SRC_SIMDOPERATORS_CALC_HPP
#define SRC_SIMDOPERATORS_CALC_HPP

#include <iostream>

#include <SIMDOperators/utils/preprocessor.h>
#include <SIMDOperators/utils/AlignmentHelper.hpp>
#include <SIMDOperators/datastructures/column.hpp>
#include <SIMDOperators/utils/constexpr/MemberDetector.h>

namespace tuddbs {

    template<typename ProcessingStyle, template<typename...> typename CalcOperator>
    class calc_binary_core {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;

        using reg_t = typename ps::register_type;

        public:
        /// Used for MetaOperator to determine if operator has a state
        constexpr static bool is_stateful = false;

        constexpr static bool is_available = detector::has_static_method_apply_v<tsl::functors::loadu<ps, tsl::workaround>>
                                            && detector::has_static_method_apply_v<tsl::functors::storeu<ps, tsl::workaround>>;


        DBTUD_CXX_ATTRIBUTE_FORCE_INLINE
        static void
        apply(base_type * out1, size_t& out1_count, 
              const base_type * in1, size_t in1_element_count, 
              const base_type * in2, size_t in2_element_count) {
            if(in1_element_count == 0){
                return;
            }
            assert(in1_element_count == in2_element_count);

            size_t vector_count = in1_element_count / ps::vector_element_count();
            for (size_t i = 0; i < vector_count; ++i) {
                /// load data into vector registers
                reg_t data_vector_left  = tsl::loadu<ps>(in1);
                reg_t data_vector_right = tsl::loadu<ps>(in2);
                /// compute elementwise operation
                reg_t result_vec = CalcOperator<ps, tsl::workaround>::apply(data_vector_left, data_vector_right);

                /// store the result
                tsl::storeu<ps>(out1, result_vec);
                
                /// increment data pointer
                out1 += ps::vector_element_count();
                in1  += ps::vector_element_count();
                in2  += ps::vector_element_count();
            }
            out1_count += in1_element_count;
        }
    };


    template<typename ProcessingStyle, template <typename ...> typename CalcOperator>
    class calc_binary {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = col_t *;

        public:

        DBTUD_CXX_ATTRIBUTE_FORCE_INLINE
        static col_ptr
        apply(const col_ptr columnLhs, const col_ptr columnRhs){
            /// Get the alignments of the input columns
            auto alignmentLhs = AlignmentHelper<ps>::getAlignment(columnLhs->getRawDataPtr());
            auto alignmentRhs = AlignmentHelper<ps>::getAlignment(columnRhs->getRawDataPtr());


            size_t pre_elements_lhs = std::min(columnLhs->getPopulationCount(), alignmentLhs.getElementsUntilAlignment());
            size_t pre_elements_rhs = std::min(columnRhs->getPopulationCount(), alignmentRhs.getElementsUntilAlignment());
            size_t pre_elements = std::min(pre_elements_lhs, pre_elements_rhs);

            auto result = new Column<base_type>(columnLhs->getPopulationCount(), ps::vector_size_B());

            auto lhsPtr = columnLhs->getRawDataPtr();
            auto rhsPtr = columnRhs->getRawDataPtr();
            auto resPtr = result->getRawDataPtr();

            /// Scalar preprocessing
            size_t res_count = 0;
            calc_binary_core<scalar, CalcOperator>::apply(resPtr, res_count, lhsPtr, pre_elements, rhsPtr, pre_elements);

            /// Vector processing
            size_t vector_count = (columnLhs->getPopulationCount() - pre_elements) / ps::vector_element_count();
            calc_binary_core<ps, CalcOperator>::apply(resPtr + pre_elements, res_count, 
                                               lhsPtr + pre_elements, vector_count * ps::vector_element_count(), 
                                               rhsPtr + pre_elements, vector_count * ps::vector_element_count());

            /// Scalar postprocessing
            calc_binary_core<scalar, CalcOperator>::apply(resPtr + res_count, res_count, 
                                                   lhsPtr + pre_elements + vector_count * ps::vector_element_count(), 
                                                   columnLhs->getPopulationCount() - pre_elements - vector_count * ps::vector_element_count(), 
                                                   rhsPtr + pre_elements + vector_count * ps::vector_element_count(), 
                                                   columnLhs->getPopulationCount() - pre_elements - vector_count * ps::vector_element_count());

            result->setPopulationCount(res_count);
            return result;
        }

    };

} // namespace tuddbs



#endif //SRC_SIMDOPERATORS_CALC_HPP



