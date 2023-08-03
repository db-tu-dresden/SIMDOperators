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
#ifndef SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_PROJECTIONPATH_HPP
#define SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_PROJECTIONPATH_HPP

#include <iostream>

#include <SIMDOperators/wrappers/DAPHNE/project.hpp>

namespace tuddbs{
    template<typename ProcessingStyle>
    class daphne_projection_path {
        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = col_t *;
        using const_col_ptr = const col_t *;

    
    public:

        template<typename... Args> 
        DBTUD_CXX_ATTRIBUTE_FORCE_INLINE
        col_ptr operator()(const_col_ptr column, Args... positions){
            using op_t = tuddbs::daphne_project<ps>;
            op_t project;
            const_col_ptr current_pos = nullptr;
            for (const_col_ptr pos : {positions...}) {
                if (!current_pos) {
                    current_pos = pos;
                    continue;
                }

                current_pos = project(pos, current_pos);
                
            }

            return project(column, current_pos);
        }


    };

}; //namespace tuddbs
#endif //SRC_SIMDOPERATORS_WRAPPERS_DAPHNE_PROJECTIONPATH_HPP