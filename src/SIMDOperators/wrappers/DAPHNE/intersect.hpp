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
#ifndef SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_INTERSECT_HPP
#define SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_INTERSECT_HPP
#include <iostream>

#include <SIMDOperators/utils/preprocessor.h>
#include <SIMDOperators/utils/AlignmentHelper.hpp>
#include <SIMDOperators/datastructures/column.hpp>
#include <SIMDOperators/operators/intersect_sorted.hpp>

namespace tuddbs{
    template<typename ProcessingStyle>
    class daphne_intersect {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = col_t *;
        using const_col_ptr = const col_t *;

        using reg_t = typename ps::register_type;

        public:

            DBTUD_CXX_ATTRIBUTE_FORCE_INLINE
            col_ptr operator()(const_col_ptr column_lhs, const_col_ptr column_rhs){
            
                base_type * lhs_ptr = const_cast<base_type *>(column_lhs->getRawDataPtr());
                base_type * rhs_ptr = const_cast<base_type *>(column_rhs->getRawDataPtr());

                int result_count;
                (column_rhs->getPopulationCount() < column_lhs->getPopulationCount())? result_count = column_rhs->getPopulationCount() : result_count = column_lhs->getPopulationCount();

                auto result = Column<base_type>::create(result_count, ps::vector_size_B());

                auto result_ptr = result->getRawDataPtr();

                auto batch_size = ps::vector_element_count();

                typename tuddbs::intersect_sorted<ps>::State state = {.result_ptr = result_ptr, .p_Data1Ptr = lhs_ptr, .p_CountData1 = batch_size, .p_Data2Ptr = rhs_ptr, .p_CountData2 = batch_size};
                while(((state.p_Data1Ptr - lhs_ptr + ps::vector_element_count()) < column_lhs->getPopulationCount()) && ((state.p_Data2Ptr - rhs_ptr + ps::vector_element_count()) < column_rhs->getPopulationCount())){
                    tuddbs::intersect_sorted<ps>{}(state);
                    
                    const size_t temp1 = lhs_ptr + column_lhs->getPopulationCount() - state.p_Data1Ptr;
                    const size_t temp2 = rhs_ptr + column_rhs->getPopulationCount() - state.p_Data2Ptr;

                    state.p_CountData1 = std::min(batch_size, temp1);
                    state.p_CountData2 = std::min(batch_size, temp2);
                }
                state.p_CountData1 = lhs_ptr + column_lhs->getPopulationCount() - state.p_Data1Ptr;
                state.p_CountData2 = rhs_ptr + column_rhs->getPopulationCount() - state.p_Data2Ptr;

                tuddbs::intersect_sorted<ps>::flush(state);

                result->setPopulationCount(state.result_ptr - result_ptr);

                return result;
            }


    };

}; // namespace tuddbs

#endif //SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_INTERSECT_HPP