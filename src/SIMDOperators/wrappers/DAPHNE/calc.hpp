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
#ifndef SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_CALC_HPP
#define SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_CALC_HPP
#include <iostream>

#include <SIMDOperators/utils/preprocessor.h>
#include <SIMDOperators/utils/AlignmentHelper.hpp>
#include <SIMDOperators/datastructures/column.hpp>
#include <SIMDOperators/operators/calc.hpp>

namespace tuddbs{
    template<typename ProcessingStyle, template <typename ...> typename CalcOperator>
    class daphne_calc {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = col_t *;
        using const_col_ptr = const col_t *;

        using reg_t = typename ps::register_type;

        public:

            DBTUD_CXX_ATTRIBUTE_FORCE_INLINE
            col_ptr operator()(const_col_ptr columnLhs, const_col_ptr columnRhs){
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

}; // namespace tuddbs

#endif //SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_CALC_HPP
