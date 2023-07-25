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
#ifndef SRC_SIMDOPERATORS_OPERATORS_PROJECT_HPP
#define SRC_SIMDOPERATORS_OPERATORS_PROJECT_HPP
#include <iostream>

#include <SIMDOperators/utils/preprocessor.h>
#include <SIMDOperators/utils/AlignmentHelper.hpp>
#include <SIMDOperators/datastructures/column.hpp>
#include <SIMDOperators/utils/BasicStash.hpp>

namespace tuddbs{
    template<typename ProcessingStyle, size_t BatchSizeInBytes>
    class project {
        static_assert(BatchSizeInBytes % ProcessingStyle::vector_size_B() == 0, "BatchSizeInBytes must be a multiple of the vector size!");

        using ps = ProcessingStyle;
        using base_type = typename ps::base_type;
        using scalar = tsl::simd<base_type, tsl::scalar>;

        using col_t = Column<base_type>;
        using col_ptr = col_t *;
        using const_col_ptr = const col_t *;

        using reg_t = typename ps::register_type;

        public:

            template<typename StateT>
            void operator()(StateT & state, const base_type * input_data) {
                const base_type * pos_ptr = state.data_ptr();

                base_type * result_ptr = state.result_ptr();

                for (size_t i = 0; i < state.element_count(); i += ps::vector_element_count()) {
                    /// load data into vector register
                    reg_t data_vector = tsl::load<ps>(pos_ptr);
                    /// gather the corresponding data
                    reg_t result_vector = tsl::gather<ps, ps>(input_data, data_vector);
                    /// store the resulting data
                    tsl::storeu<ps>(result_ptr, result_vector);
                    /// increment the position vector
                    pos_ptr += ps::vector_element_count();
                    /// increment the output data pointer
                    result_ptr += ps::vector_element_count();
                }
                state.result_ptr(result_ptr);
                state.data_ptr(pos_ptr);
            }
    };

}; // namespace tuddbs

#endif //SRC_SIMDOPERATORS_OPERATORS_PROJECT_HPP
