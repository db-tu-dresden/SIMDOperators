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


#ifndef SRC_SIMDOPERATORS_OPERATORS_METAOPERATOR_HPP
#define SRC_SIMDOPERATORS_OPERATORS_METAOPERATOR_HPP

#include <iostream>

#include <SIMDOperators/utils/preprocessor.h>
#include <SIMDOperators/utils/AlignmentHelper.hpp>
#include <SIMDOperators/datastructures/column.hpp>

namespace tuddbs{

    template<typename ProcessingStyle, typename Core>
    class MetaOperator {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = col_t *;
        using const_col_ptr = const col_t *;

        using reg_t = typename ps::register_type;
        using mask_t = typename ps::mask_type;
        using imask_t = typename ps::imask_type;

        public:
        // 1 Input -> 1 Output
        template<typename ... ApplyArgs>
        DBTUD_CXX_ATTRIBUTE_FORCE_INLINE
        static col_ptr
        apply(const_col_ptr column1, ApplyArgs ... args){
            auto alignment = AlignmentHelper<ps>::getAlignment(column1->getRawDataPtr());
            size_t pre_count = std::min(column1->getPopulationCount(), alignment.getElementsUntilAlignment());
            size_t vector_count = (column1->getPopulationCount() - pre_count) / ps::vector_element_count() * ps::vector_element_count();
            size_t post_count = column1->getPopulationCount() - pre_count - vector_count;

            auto result = new col_t(column1->getPopulationCount(), ps::vector_size_B());

            auto result_ptr = result->getRawDataPtr();
            auto column1_ptr = column1->getRawDataPtr();

            size_t out1_count = 0;
            size_t out1_overallCount = 0;

            if constexpr (Core::is_stateful){
                typename Core::state state;
                // Scalar
                Core::apply(result_ptr, out1_count, column1_ptr, pre_count, state, args...);
                result_ptr += out1_count;
                column1_ptr += pre_count;
                out1_overallCount += out1_count;
                // Vector
                Core::apply(result_ptr, out1_count, column1_ptr, vector_count, state, args...);
                result_ptr += out1_count;
                column1_ptr += vector_count;
                out1_overallCount += out1_count;
                // Scalar
                Core::apply(result_ptr, out1_count, column1_ptr, post_count, state, args...);
                out1_overallCount += out1_count;
            } else {
                // Scalar
                Core::apply(result_ptr, out1_count, column1_ptr, pre_count, args...);
                result_ptr += out1_count;
                column1_ptr += pre_count;
                out1_overallCount += out1_count;
                // Vector
                Core::apply(result_ptr, out1_count, column1_ptr, vector_count, args...);
                result_ptr += out1_count;
                column1_ptr += vector_count;
                out1_overallCount += out1_count;
                // Scalar
                Core::apply(result_ptr, out1_count, column1_ptr, post_count, args...);
                out1_overallCount += out1_count;
            }
            result->setPopulationCount(out1_overallCount);
            return result;

        } 

    };

}
#endif //SRC_SIMDOPERATORS_OPERATORS_METAOPERATOR_HPP