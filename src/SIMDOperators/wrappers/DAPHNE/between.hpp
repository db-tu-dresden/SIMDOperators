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
#ifndef SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_BETWEEN_HPP
#define SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_BETWEEN_HPP

#include <iostream>

#include <SIMDOperators/wrappers/DAPHNE/select.hpp>
#include <SIMDOperators/wrappers/DAPHNE/intersect.hpp>

namespace tuddbs{
    template<typename ProcessingStyle>
    class between {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = col_t *;
        using const_col_ptr = const col_t *;

    
    public:

        DBTUD_CXX_ATTRIBUTE_FORCE_INLINE
        col_ptr operator()(const_col_ptr column, base_type lower_bound, base_type higher_bound){
            using op_t = tuddbs::daphne_select<ps, tsl::functors::less_than_or_equal>;
            op_t select_lower;
            col_ptr lesser_pos = select(column, lower_bound);

            using op_t = tuddbs::daphne_select<ps, tsl::functors::greater_than_or_equal>;
            op_t select_greater;
            col_ptr greater_pos = select(column, lower_bound);

            using op_t = tuddbs::daphne_intersect<ps>;
            op_t intersect;
            return intersect(lesser_pos, greater_pos);
        }


    };

}; //namespace tuddbs
#endif //SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_BETWEEN_HPP