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
#ifndef SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_SELECT_HPP
#define SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_SELECT_HPP
#include "SIMDOperators/utils/BasicStash.hpp"
#include <iostream>

#include <SIMDOperators/utils/preprocessor.h>
#include <SIMDOperators/utils/AlignmentHelper.hpp>
#include <SIMDOperators/datastructures/column.hpp>
#include <SIMDOperators/operators/select.hpp>

namespace tuddbs{
    template<typename ProcessingStyle, template<typename ...> typename CompareOperator>
    class daphne_select {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = col_t *;
        using const_col_ptr = const col_t *;

        using reg_t = typename ps::register_type;

        public:

            DBTUD_CXX_ATTRIBUTE_FORCE_INLINE
            col_ptr operator()(const_col_ptr column, const base_type& predicate){
                return select<ProcessingStyle, CompareOperator>::apply(column, predicate);
            }


    };

}; // namespace tuddbs

#endif //SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_SELECT_HPP
